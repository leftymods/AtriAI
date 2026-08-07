# AtriAI — репродукция архитектуры Helix (Figure AI)

System 2 (VLM, «мозг») + System 1 (визомоторная политика 200 Гц, «руки»),
соединённые единым латентным вектором. Описание архитектуры: figure.ai/news/helix.

## Установка

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy pyyaml pillow tqdm pytest
pip install "transformers>=4.45" accelerate peft bitsandbytes  # для реальной VLM
```

## Режимы

- `tiny` (по умолчанию) — свой маленький трансформер вместо VLM, запуск без весов.
- `vlm` — реальная открытая VLM (например `Qwen/Qwen2-VL-7B-Instruct`).

Включить VLM в `configs/helix_default.yaml`:

```yaml
system2:
  model_type: "vlm"
  vlm_name: "Qwen/Qwen2-VL-7B-Instruct"
  lora: true          # QLoRA: 7B обучается на одной 3090 (~7 ГБ)
```

## Общение с ИИ (чат)

```bash
python scripts/chat.py --config configs/helix_default.yaml
python scripts/chat.py --prompt "Что на картинке?" --image photo.jpg
```

## Обучение

Данные — эпизоды телеоперации в `.npz` (в папке `--data`), формат:
`images` (T,H,W,3), `states` (T,S), `actions` (T,36), `text` (инструкция).

```bash
# проверка каркаса
python scripts/sanity.py

# обучение на синтетике (smoke-test)
python scripts/train.py --steps 10

# обучение на реальных данных
python scripts/train.py --data /path/to/teleop --steps 250000

# предобучение vision backbone S1 в симуляции
python scripts/sim_pretrain.py --steps 5000
```

Чекпоинты: `checkpoints/s1.pt`, `checkpoints/s2_lora/` (адаптеры PEFT),
`checkpoints/s2_heads.pt` (LatentProjector + StateProjector).

## Запуск на роботе (2 GPU — как у Figure)

`scripts/deploy.py` запускает S2 асинхронно (7–9 Гц) на `cuda:0` и S1
в реальном времени (200 Гц) на `cuda:1`, общаясь через shared latent.

```bash
python scripts/deploy.py --command "Pick up the desert item"
```

## Тесты

```bash
python -m pytest -q
```
