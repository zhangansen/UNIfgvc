"""
models/encoders.py — CLIP visual, DINOv2 visual, CLIP text, and multimodal fusion.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor,
    AutoModel, AutoImageProcessor,
)

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DINO_MODEL_ID = "facebook/dinov2-base"

_clip_model     = None
_clip_processor = None
_dino_model     = None
_dino_processor = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_encoders(
    clip_model_id: str = CLIP_MODEL_ID,
    dino_model_id: str = DINO_MODEL_ID,
):
    global _clip_model, _clip_processor, _dino_model, _dino_processor

    if _clip_model is None:
        _clip_model     = CLIPModel.from_pretrained(clip_model_id, device_map="auto").eval()
        _clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        print(f"CLIP loaded  | dim={_clip_model.config.projection_dim}")

    if _dino_model is None:
        _dino_model     = AutoModel.from_pretrained(dino_model_id, device_map="auto").eval()
        _dino_processor = AutoImageProcessor.from_pretrained(dino_model_id)
        print(f"DINOv2 loaded | hidden_size={_dino_model.config.hidden_size}")


def _unwrap_clip(out):
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0, :]
    raise ValueError(f"Cannot unwrap CLIP output: {type(out)}")


@torch.no_grad()
def encode_clip_visual(image: Image.Image) -> torch.Tensor:
    """CLIP image embedding, L2-normed → (512,)"""
    inp = _clip_processor(images=image, return_tensors="pt").to(DEVICE)
    out = _clip_model.get_image_features(**inp)
    return F.normalize(_unwrap_clip(out), dim=-1).squeeze(0).cpu()


@torch.no_grad()
def encode_clip_text(text: str) -> torch.Tensor:
    """CLIP text embedding, L2-normed → (512,)"""
    inp = _clip_processor(
        text=[text], return_tensors="pt", padding=True, truncation=True
    ).to(DEVICE)
    out = _clip_model.get_text_features(**inp)
    return F.normalize(_unwrap_clip(out), dim=-1).squeeze(0).cpu()


@torch.no_grad()
def encode_dino_visual(image: Image.Image) -> torch.Tensor:
    """DINOv2 CLS-token embedding, L2-normed → (768,)"""
    inp = _dino_processor(images=image, return_tensors="pt").to(DEVICE)
    out = _dino_model(**inp)
    cls = out.last_hidden_state[:, 0, :]
    return F.normalize(cls, dim=-1).squeeze(0).cpu()


def fuse(
    clip_vis: torch.Tensor,
    dino_vis: torch.Tensor | None,
    cdv_text: torch.Tensor | None,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """
    Concatenate available modalities and L2-normalise.

    Ablation modes:
        clip_only    → dino_vis=None, cdv_text=None  →  512-dim
        clip_cdv     → dino_vis=None                 → 1024-dim
        clip_dino    → cdv_text=None                 → 1280-dim
        full         → all three                     → 1792-dim
    """
    w1, w2, w3 = weights
    parts = [w1 * clip_vis]
    if dino_vis is not None:
        parts.append(w2 * dino_vis)
    if cdv_text is not None:
        parts.append(w3 * cdv_text)
    return F.normalize(torch.cat(parts, dim=-1), dim=-1)
