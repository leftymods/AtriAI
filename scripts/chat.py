"""Script: interactive chat with the VLM (System 2).

This is the "talk to the AI" part of AtriAI. With model_type='vlm' it uses the
real pretrained VLM (e.g. Qwen2-VL) so you can chat freely with text (and images
via --image). In tiny mode there is no real language model, so chat is disabled.

After training, the S2 heads/adapters from --checkpoint can be loaded to steer the
latent vector — chat() stays on the raw pretrained VLM for conversation.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from helix.config import Config
from helix.model import HelixVLA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/helix_default.yaml")
    ap.add_argument("--image", default=None, help="path to a camera image (optional)")
    ap.add_argument("--prompt", default=None, help="single prompt, then exit")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if cfg.system2.model_type != "vlm":
        print("Chat requires model_type='vlm' in the config (a real VLM).")
        sys.exit(1)

    model = HelixVLA(cfg).eval()
    s2 = model.s2

    image_t = None
    if args.image:
        from PIL import Image
        import torchvision.transforms.functional as F
        image_t = F.to_tensor(Image.open(args.image).convert("RGB"))

    def ask(text: str):
        reply = s2.chat(text, image_t)
        print(f"\n[you]  {text}")
        print(f"[ai]   {reply.splitlines()[-1] if reply else ''}")

    if args.prompt:
        ask(args.prompt)
        return

    print("Chat with the VLM. Type 'exit' to quit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ("exit", "quit"):
            break
        ask(text)


if __name__ == "__main__":
    main()
