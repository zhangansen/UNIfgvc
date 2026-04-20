"""
models/cdv_captioner.py — 3-stage CDV captioner using Qwen2.5-VL.
"""

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

_qwen_model     = None
_qwen_processor = None

_SYS_PROMPT = (
    "You are an expert visual analyst specialising in fine-grained image classification. "
    "Your task is to identify discriminative visual attributes that distinguish "
    "a target subject from visually similar categories."
)


def load_qwen(model_id: str = QWEN_MODEL_ID):
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb_cfg, device_map="auto"
    ).eval()
    _qwen_processor = AutoProcessor.from_pretrained(model_id)
    print(f"Qwen2.5-VL ({model_id}) loaded in 4-bit")


def _run_qwen(messages: list, max_new_tokens: int = 256) -> str:
    text = _qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    kwargs = dict(text=[text], padding=True, return_tensors="pt")
    if image_inputs:
        kwargs["images"] = image_inputs
    if video_inputs:
        kwargs["videos"] = video_inputs
    inputs = _qwen_processor(**kwargs).to(_qwen_model.device)
    with torch.no_grad():
        gen_ids = _qwen_model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    return _qwen_processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


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


# ── Full 3-stage orchestrator ─────────────────────────────────────────────────
def full_cdv_caption(
    target_img: Image.Image,
    class_name: str,
    ref_imgs: list[Image.Image],
    ref_names: list[str],
    fast: bool = False,
) -> str:
    """
    Returns the final discriminative CDV paragraph.

    fast=True  → single-prompt fallback (~4s/image)
    fast=False → full 3-stage pipeline (~10s/image)
    """
    if fast:
        return fast_cdv_caption(target_img)

    attributes = stage1_discover(target_img, ref_imgs, ref_names)
    attr_descs = stage2_describe(target_img, attributes)
    return stage3_summarize(attr_descs, class_name)
