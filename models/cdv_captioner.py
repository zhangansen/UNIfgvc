"""
models/cdv_captioner.py — 3-stage CDV captioner using the Qwen-VL API
(DashScope, OpenAI-compatible mode).

Drop-in replacement for the previous local Qwen2.5-VL version:
  • stage1..3 / fast_cdv_caption / full_cdv_caption keep identical
    prompts and signatures — gallery/builder.py and eval/ need no change.
  • load_qwen() now initialises the API client instead of loading weights.

Config (env-overridable):
    QWEN_VL_API_KEY          required (fallbacks: SILICONFLOW_API_KEY,
                             DASHSCOPE_API_KEY) — SiliconFlow default provider
    QWEN_VL_MODEL            default "Qwen/Qwen2.5-VL-7B-Instruct"
    QWEN_VL_BASE_URL         default SiliconFlow OpenAI-compatible endpoint
    QWEN_MAX_IMAGE_SIDE      default 1024 (client-side resize before upload)
"""

import base64
import io
import os

from PIL import Image
from openai import OpenAI

# ── Configuration (env-overridable) ─────────────────────────────────────────
QWEN_VL_MODEL       = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
QWEN_VL_BASE_URL    = os.environ.get(
    "QWEN_VL_BASE_URL",
    "https://api.siliconflow.cn/v1",
)
QWEN_MAX_IMAGE_SIDE = int(os.environ.get("QWEN_MAX_IMAGE_SIDE", "1024"))

_client: OpenAI | None = None

_SYS_PROMPT = (
    "You are an expert visual analyst specialising in fine-grained image classification. "
    "Your task is to identify discriminative visual attributes that distinguish "
    "a target subject from visually similar categories."
)


def load_qwen(model_id: str = QWEN_VL_MODEL):
    """Initialise the Qwen-VL API client (replaces local weight loading)."""
    global _client, QWEN_VL_MODEL
    if _client is not None:
        return
    api_key = (os.environ.get("QWEN_VL_API_KEY")
               or os.environ.get("SILICONFLOW_API_KEY")
               or os.environ.get("DASHSCOPE_API_KEY"))
    if not api_key:
        raise EnvironmentError(
            "\n\nNo Qwen-VL API key found. Set ONE of these env vars:\n"
            "  QWEN_VL_API_KEY / SILICONFLOW_API_KEY   (SiliconFlow, default provider)\n"
            "    -> create at https://cloud.siliconflow.com/account/ak\n"
            "  DASHSCOPE_API_KEY                       (Alibaba DashScope)\n"
            "    -> create at https://bailian.console.aliyun.com/\n"
        )
    QWEN_VL_MODEL = model_id
    _client = OpenAI(
        api_key=api_key,
        base_url=QWEN_VL_BASE_URL,
        timeout=120,       # seconds per request (multi-image Stage 1 can be slow)
        max_retries=5,     # SDK auto-retries 429 / 5xx / connection errors
    )
    print(f"Qwen-VL API client ready | model={model_id} | base={QWEN_VL_BASE_URL}")


def _pil_to_data_uri(img: Image.Image,
                     max_side: int = QWEN_MAX_IMAGE_SIDE) -> str:
    """Encode a PIL image as a base64 JPEG data URI.

    Client-side resize keeps the upload small; the service re-tokenises
    anyway. <=1024px is roughly 1-1.5k vision tokens per image.
    """
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale),
                          round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _convert_content(content):
    """Translate qwen_vl_utils-style content into OpenAI-compatible parts."""
    out = []
    for part in content:
        if isinstance(part, str):
            out.append({"type": "text", "text": part})
        elif part.get("type") == "image":
            img = part["image"]
            uri = img if isinstance(img, str) else _pil_to_data_uri(img)
            out.append({"type": "image_url", "image_url": {"url": uri}})
        else:
            out.append(part)
    return out


def _run_qwen(messages: list, max_new_tokens: int = 256) -> str:
    if _client is None:
        raise RuntimeError("Call load_qwen() before using the captioner.")
    api_messages = [
        {"role": m["role"],
         "content": _convert_content(m["content"])
                    if isinstance(m["content"], list) else m["content"]}
        for m in messages
    ]
    resp = _client.chat.completions.create(
        model=QWEN_VL_MODEL,
        messages=api_messages,
        max_tokens=max_new_tokens,
        # Deterministic captions (was do_sample=False).
        # If a newer model rejects these with HTTP 400, remove them.
        temperature=0.0,
        top_p=1.0,
    )
    return (resp.choices[0].message.content or "").strip()


# ── Stage 1: Discriminative Region Discovery ──────────────────────────────────
def stage1_discover(
    target_img: Image.Image,
    ref_imgs: list[Image.Image],
    ref_names: list[str],
) -> str:
    """Identify which attributes discriminate the target from confusable classes."""
    content = [{"type": "image", "image": target_img}]
    for img in ref_imgs:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": (
        f"Image 1 is the TARGET subject. "
        f"Images 2–{1+len(ref_imgs)} are from visually similar but DIFFERENT categories: "
        f"{', '.join(ref_names)}.\n"
        "List 3–5 specific visual attributes that best discriminate the TARGET. "
        "Reply with ONLY a comma-separated list of short attribute names."
    )})
    return _run_qwen(
        [{"role": "system", "content": _SYS_PROMPT},
         {"role": "user",   "content": content}],
        max_new_tokens=80,
    )


# ── Stage 2: Region Attribute Description ────────────────────────────────────
def stage2_describe(target_img: Image.Image, attributes: str) -> str:
    """Describe each discriminative attribute in visual detail."""
    return _run_qwen([
        {"role": "system", "content": _SYS_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": target_img},
            {"type": "text", "text": (
                f"Describe these visual attributes of the subject in the image: {attributes}.\n"
                "For each attribute write one precise sentence about its colour, shape, "
                "pattern, and size. Format: 'Attribute name: description.'"
            )},
        ]},
    ], max_new_tokens=200)


# ── Stage 3: Attribute Feature Summarization ──────────────────────────────────
def stage3_summarize(attr_descriptions: str, class_name: str) -> str:
    """Condense attribute descriptions into one dense discriminative paragraph."""
    return _run_qwen([
        {"role": "system", "content": _SYS_PROMPT},
        {"role": "user", "content": (
            f"Attribute descriptions for a {class_name}:\n{attr_descriptions}\n\n"
            "Write ONE dense paragraph (3–5 sentences) capturing the most discriminative "
            "visual features of this category for visual retrieval. "
            "Focus on appearance features visible in photographs."
        )},
    ], max_new_tokens=256)


# ── Fast single-prompt fallback ───────────────────────────────────────────────
def fast_cdv_caption(target_img: Image.Image) -> str:
    return _run_qwen([
        {"role": "system", "content": _SYS_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": target_img},
            {"type": "text", "text": (
                "Analyze this image. "
                "Step 1: identify the category. "
                "Step 2: list 5–8 discriminative visual attributes. "
                "Step 3: write ONE dense paragraph of discriminative attributes "
                "useful for visual retrieval classification."
            )},
        ]},
    ], max_new_tokens=256)


# ── Full 3-stage orchestrator ────────────────────────────────────────────────
def full_cdv_caption(
    target_img: Image.Image,
    class_name: str,
    ref_imgs: list[Image.Image],
    ref_names: list[str],
    fast: bool = False,
) -> str:
    """
    Returns the final discriminative CDV paragraph.

    fast=True  → single-prompt fallback (1 API call/image)
    fast=False → full 3-stage pipeline (3 API calls/image)
    """
    if fast:
        return fast_cdv_caption(target_img)

    attributes = stage1_discover(target_img, ref_imgs, ref_names)
    attr_descs = stage2_describe(target_img, attributes)
    return stage3_summarize(attr_descs, class_name)
