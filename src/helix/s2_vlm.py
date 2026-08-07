"""System 2 (S2) — the VLM that distills semantics into a latent vector.

Matches the Helix description:

  "System 2 (S2): An onboard internet-pretrained VLM operating at 7-9 Hz for scene
   understanding and language comprehension, enabling broad generalization across
   objects and contexts."

  "It processes monocular robot images and robot state information (consisting of
   wrist pose and finger positions) after projecting them into vision-language
   embedding space. Combined with natural language commands specifying desired
   behaviors, S2 distills all semantic task-relevant information into a single
   continuous latent vector, passed to S1 to condition its low-level actions."

Two modes are provided:

  * model_type="vlm"  — loads a real open-weights VLM (e.g. Qwen2-VL) via
    `transformers`. Text + images are tokenized with its AutoProcessor; the last
    hidden state is distilled into the latent vector. Supports chat generation
    and QLoRA training.
  * model_type="tiny" — a small self-contained transformer stack (vision tower +
    decoder-only language model) that keeps the exact same interface, so the
    pipeline is runnable without huge pretrained weights.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .latent import LatentProjector

try:
    from transformers import AutoModel, AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:  # pragma: no cover - older transformers
        AutoModelForImageTextToText = AutoModel
    _HF_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _HF_AVAILABLE = False


class VisionTower(nn.Module):
    """Patch-embedding vision tower shared by S2 (monocular camera image)."""

    def __init__(self, patch: int = 16, embed_dim: int = 768, image_size: int = 224):
        super().__init__()
        self.patch = patch
        self.n_patches = (image_size // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: Tensor) -> Tensor:
        """images: (B, 3, H, W) -> tokens (B, n_patches + 1, embed_dim)."""
        b = images.shape[0]
        x = self.patch_embed(images).flatten(2).transpose(1, 2)  # (B, N, D)
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)
        x = x + self.pos_embed
        return self.norm(x)


class LanguageBackbone(nn.Module):
    """Decoder-only transformer language model (tiny mode)."""

    def __init__(self, vocab: int, d_model: int = 768, n_layers: int = 8,
                 n_heads: int = 8, d_ff: int = 2048, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.tok_embed = nn.Embedding(vocab, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, batch_first=True, dropout=0.1, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        """token_ids: (B, T) -> last-token hidden states (B, T, d_model)."""
        b, t = token_ids.shape
        x = self.tok_embed(token_ids) * (self.d_model ** 0.5)
        x = x + self.pos_embed[:, :t]
        x = self.drop(x)
        return self.norm(self.blocks(x))


class RobotStateProjector(nn.Module):
    """Projects robot state (wrist pose + finger positions) into VLM embedding space."""

    def __init__(self, state_dim: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.mlp(state).unsqueeze(1)  # (B, 1, d_model)


class System2(nn.Module):
    """VLM backbone + robot state projector -> single latent vector.

    Forward produces the latent vector L of shape (B, latent_dim). During
    end-to-end training gradients flow from System 1 back through this vector.
    """

    def __init__(self, cfg: "System2Config"):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim

        if cfg.model_type == "vlm":
            if not _HF_AVAILABLE:
                raise ImportError("install 'transformers' to use model_type='vlm'")
            load = AutoModelForImageTextToText if AutoModelForImageTextToText is not AutoModel else AutoModel
            load = AutoModelForImageTextToText if AutoModelForImageTextToText is not AutoModel else AutoModel
            if cfg.lora:
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError:  # pragma: no cover
                    BitsAndBytesConfig = None
                kwargs = {}
                if BitsAndBytesConfig is not None:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                self.vlm = load.from_pretrained(cfg.vlm_name, torch_dtype="auto", **kwargs)
            else:
                self.vlm = load.from_pretrained(cfg.vlm_name, torch_dtype="auto")
            self.processor = AutoProcessor.from_pretrained(cfg.vlm_name, trust_remote_code=True)
            self.vlm_hidden = self.vlm.config.hidden_size
            self.vision = None
            self.lm = None
            self._device_override = None
        else:
            self.vlm = None
            self.processor = None
            self.vision = VisionTower(image_size=224, embed_dim=384, patch=16)
            self.lm = LanguageBackbone(vocab=256, d_model=384, n_layers=4)
            self.vlm_hidden = 384
        self.state_proj = RobotStateProjector(12, self.vlm_hidden)
        self.latent_head = LatentProjector(self.vlm_hidden, cfg.latent_dim)

    # ------------------------------------------------------------------ VLM I/O

    def prepare_inputs(
        self,
        texts: list[str],
        images: Tensor | None,
    ) -> dict[str, Tensor] | None:
        """Tokenize text + images with the VLM's AutoProcessor (off-graph).

        Returns the processor's tensors for the forward pass, or None in tiny mode
        (where plain char-token ids are used instead).
        """
        if self.processor is None:
            return None
        pil_images = None
        if images is not None:
            imgs = (images.permute(0, 2, 3, 1).cpu().numpy() * 255)
            imgs = imgs.astype("uint8")
            from PIL import Image
            pil_images = [Image.fromarray(im) for im in imgs]
        with torch.no_grad():
            return self.processor(
                text=list(texts), images=pil_images,
                padding=True, return_tensors="pt",
            )

    def chat(self, text: str, image: Tensor | None = None, max_new_tokens: int = 128) -> str:
        """Free-form conversation with the VLM (no robot actions).

        In tiny mode this is not meaningful — an error is raised.
        """
        if self.vlm is None:
            raise RuntimeError("chat() requires model_type='vlm'")
        if self.processor is None:
            raise RuntimeError("chat() requires AutoProcessor")
        import torch as _t
        content = [{"type": "text", "text": text}]
        if image is not None:
            content.insert(0, {"type": "image", "image": image.permute(1, 2, 0).cpu().numpy()})
        msgs = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=None, return_tensors="pt")
        inputs = {k: v.to(self.vlm.device) for k, v in inputs.items() if k != "image_grid_thw"}
        with _t.no_grad():
            out = self.vlm.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(out[0], skip_special_tokens=True)

    # ------------------------------------------------------------------ forward

    def forward(self, inputs: dict | None, robot_state: Tensor) -> Tensor:
        """-> latent (B, latent_dim).

        In vlm mode `inputs` is the dict produced by `prepare_inputs`; in tiny
        mode it is None and char-level token ids + images are embedded directly
        (kept for the tiny path via `token_ids`/`images` below).
        """
        if self.vlm is not None:
            if inputs is None:
                raise ValueError("vlm mode requires prepare_inputs() output")
            dev = self.vlm.device
            kw = {k: v.to(dev, dtype=torch.bfloat16 if "pixel_values" in k else torch.long)
                  for k, v in inputs.items() if k in ("input_ids", "attention_mask",
                                                      "pixel_values", "image_grid_thw")}
            out = self.vlm(**kw)
            hidden = out.last_hidden_state  # (B, T, H)
        else:
            images = inputs.get("images") if inputs else None
            token_ids = inputs.get("token_ids") if inputs else None
            if images is None:
                images = torch.zeros(robot_state.shape[0], 3, 224, 224, device=robot_state.device)
            vis = self.vision(images)
            txt = self.lm(token_ids)
            state = self.state_proj(robot_state)
            hidden = torch.cat([txt, vis, state], dim=1)

        last_hidden = hidden[:, -1]
        return self.latent_head(last_hidden)


from .config import System2Config  # noqa: E402  (type alias for docstrings)
