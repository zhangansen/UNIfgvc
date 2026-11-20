# UniFGVC — Unified Fine-Grained Visual Classification

A few-shot fine-grained visual classification framework extending the CDV-Captioner approach with a **CLIP + DINOv2 multimodal encoder**.

## What this is

This repo implements and extends a pipeline for fine-grained visual classification (FGVC) across multiple standard benchmarks. The core idea: instead of comparing images by visual embeddings alone, we generate *contrastive text descriptions* of discriminative attributes using a VLM, and fuse that signal with visual features.

**Our extension over the base paper**: we replace the paper's CLIP-only visual encoder with a CLIP + DINOv2 concatenated encoder, giving richer local texture and structure features alongside CLIP's semantic embeddings.

## Ablation Modes

| Mode | Features | Dim | What it tests |
|---|---|---|---|
| `clip_only` | CLIP visual | 512 | Paper baseline |
| `clip_cdv` | CLIP visual + CDV text | 1024 | Paper full method |
| `clip_dino` | CLIP + DINOv2 visual | 1280 | Our visual extension |
| `full` | CLIP + DINOv2 + CDV text | 1792 | Our full method |

## Supported Datasets

| Dataset | Classes | Domain |
|---|---|---|
| CUB-200-2011 | 200 | Birds |
| Stanford Cars | 196 | Cars |
| FGVC-Aircraft | 100 | Aircraft |
| Oxford-IIIT Pets | 37 | Pets |
| Stanford Dogs | 120 | Dogs |
| Food-101 | 101 | Food |

## Models

- **CLIP ViT-B/32** — semantic visual features (512-dim)
- **DINOv2-Base** — local texture / structure features (768-dim)
- **Qwen2.5-VL-7B** via the **SiliconFlow API** (OpenAI-compatible) — CDV caption generation

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/unifgvc
cd unifgvc
pip install -r requirements.txt
```

Set your **SiliconFlow API key** (required — Qwen-VL runs as a cloud API):

**On Kaggle** — add `QWEN_VL_API_KEY` to your [Kaggle Secrets](https://www.kaggle.com/settings) (Settings → API → Add Secret). The code picks it up automatically.

**Locally** — set the environment variable:
```bash
export QWEN_VL_API_KEY=sk-xxxxxxxx
```

Create a key at [cloud.siliconflow.com/account/ak](https://cloud.siliconflow.com/account/ak). Defaults: model `Qwen/Qwen2.5-VL-7B-Instruct` on the SiliconFlow OpenAI-compatible endpoint (`https://api.siliconflow.cn/v1`). Switch provider or model with the `QWEN_VL_BASE_URL` / `QWEN_VL_MODEL` env vars (e.g. back to Alibaba DashScope with `qwen-vl-max-latest`) — no code changes needed.

> The Hugging Face token is now **optional** — it is only used to download the CLIP / DINOv2 encoders (not gated). If you are behind a restricted network, `export HF_TOKEN=hf_xxx` (from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) still works with the same Kaggle-Secrets / env-var / `huggingface-cli login` resolution as before.

## Usage

### Step 1 — Build the gallery

```bash
python scripts/build_gallery.py \
    --dataset cub \
    --data_root /path/to/cub \
    --out_dir results/cub \
    --k_shot 4
```

Use `--fast_cdv` to skip the full 3-stage CDV pipeline (single-prompt fallback, ~4s/image vs ~10s/image).

### Step 2 — Evaluate

```bash
python scripts/evaluate.py \
    --dataset cub \
    --data_root /path/to/cub \
    --gallery_dir results/cub \
    --out_dir results/cub \
    --k_shot 4 \
    --n_test_per_class 3
```

Results are written to `results/cub/eval_results.json` and `results/cub/eval_results.csv`.

### Running all datasets

```bash
for DATASET in cub cars aircraft pets dogs food101; do
    python scripts/build_gallery.py \
        --dataset $DATASET \
        --data_root /data/$DATASET \
        --out_dir results/$DATASET \
        --k_shot 4

    python scripts/evaluate.py \
        --dataset $DATASET \
        --data_root /data/$DATASET \
        --gallery_dir results/$DATASET \
        --out_dir results/$DATASET \
        --k_shot 4
done
```

## Dataset Setup

| Dataset | Download |
|---|---|
| CUB-200-2011 | [Kaggle](https://www.kaggle.com/datasets/wenewone/cub2002011) |
| Stanford Cars | [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) |
| FGVC-Aircraft | [VGG](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) |
| Oxford Pets | [VGG](https://www.robots.ox.ac.uk/~vgg/data/pets/) |
| Stanford Dogs | [Stanford](http://vision.stanford.edu/aditya86/ImageNetDogs/) |
| Food-101 | [Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101) |

## Output Format

**JSON** (`eval_results.json`):
```json
{
  "dataset": "cub",
  "k_shot": 4,
  "n_test_per_class": 3,
  "results": {
    "clip_only": 0.6123,
    "clip_cdv":  0.7045,
    "clip_dino": 0.7312,
    "full":      0.7891
  }
}
```

**CSV** (`eval_results.csv`) — one row per dataset + mode, append-friendly across runs:
```
dataset,k_shot,n_test_per_class,fast_cdv,seed,mode,top1_acc
cub,4,3,False,42,clip_only,0.6123
cub,4,3,False,42,clip_cdv,0.7045
...
```

## Repo Structure

```
unifgvc/
├── configs/default.yaml       # all experiment knobs
├── datasets/                  # dataset loaders (CUB, Cars, Aircraft, Pets, Dogs, Food101)
├── models/
│   ├── encoders.py            # CLIP, DINOv2, fusion
│   └── cdv_captioner.py       # 3-stage Qwen CDV pipeline
├── gallery/
│   ├── builder.py             # builds & checkpoints the image cache
│   └── retrieval.py           # cosine sim + per-class mean aggregation
├── eval/
│   ├── evaluator.py           # ablation runner
│   └── metrics.py             # accuracy + table formatting
└── scripts/
    ├── build_gallery.py       # CLI: build the gallery
    └── evaluate.py            # CLI: run eval + save CSV/JSON
```

## Key Design Decisions

**Why concatenate then L2-normalise?** Concatenation preserves the individual signal from each modality before normalisation collapses it. Each modality is already L2-normed before concatenation, so no single modality dominates by scale.

**Why per-class mean aggregation?** Averaging cosine similarities over K support images gives a more stable prototype than nearest-neighbour, especially for visually diverse classes.

**Why 3-stage CDV?** Stage 1 generates attributes that are *explicitly contrastive* — written with knowledge of the visually hardest neighbours. A generic caption misses the subtle differences; a contrastive one focuses exactly on them.

## Citation

If you use this code, please cite the original CDV/UniFGVC paper (add citation here) and this repo.
