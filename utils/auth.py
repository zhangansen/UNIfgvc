"""
utils/auth.py — Hugging Face token setup.

Tries in order:
    1. Kaggle Secrets (when running on Kaggle)
    2. HF_TOKEN environment variable
    3. ~/.huggingface/token (set by `huggingface-cli login`)

Raises a clear error if none are found.
"""

import os


def setup_hf_token(secret_name: str = "HF_TOKEN") -> str:
    """
    Resolve and export the HF token. Returns the token string.
    Call this once at the top of any script before loading models.
    """
    token = None

    # ── 1. Kaggle Secrets ─────────────────────────────────────────────────────
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(secret_name)
        print("HF_TOKEN loaded from Kaggle Secrets.")
    except Exception:
        pass  # not running on Kaggle, or secret not set

    # ── 2. Environment variable ───────────────────────────────────────────────
    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            print("HF_TOKEN loaded from environment variable.")

    # ── 3. ~/.huggingface/token (huggingface-cli login) ───────────────────────
    if not token:
        hf_token_file = os.path.expanduser("~/.huggingface/token")
        if os.path.exists(hf_token_file):
            with open(hf_token_file) as f:
                token = f.read().strip()
            if token:
                print("HF_TOKEN loaded from ~/.huggingface/token.")

    if not token:
        raise EnvironmentError(
            "\n\nHugging Face token not found. Please do ONE of the following:\n"
            "  • Kaggle:  Add HF_TOKEN to your Kaggle Secrets\n"
            "  • Local:   export HF_TOKEN=hf_xxxxxxxxxxxx\n"
            "  • Local:   run `huggingface-cli login`\n"
            "The token is used to download the CLIP / DINOv2 encoders.\n"
            "Get your token at: https://huggingface.co/settings/tokens\n"
        )

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token  # older transformers compat
    return token


def setup_vlm_api_key() -> str:
    """
    Resolve and export the Qwen-VL API key (provider-agnostic). Returns the key.

    Default provider is SiliconFlow; DashScope also works by setting
    QWEN_VL_BASE_URL + QWEN_VL_MODEL accordingly (see models/cdv_captioner.py).

    Tries in order:
        1. Kaggle Secrets — secret name QWEN_VL_API_KEY (when on Kaggle)
        2. Environment: QWEN_VL_API_KEY → SILICONFLOW_API_KEY → DASHSCOPE_API_KEY

    Raises a clear error if none is found.
    """
    key = None

    # ── 1. Kaggle Secrets ─────────────────────────────────────────────────────
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("QWEN_VL_API_KEY")
        print("QWEN_VL_API_KEY loaded from Kaggle Secrets.")
    except Exception:
        pass  # not running on Kaggle, or secret not set

    # ── 2. Environment variables (provider chain) ────────────────────────────
    if not key:
        for var in ("QWEN_VL_API_KEY", "SILICONFLOW_API_KEY", "DASHSCOPE_API_KEY"):
            key = os.environ.get(var)
            if key:
                print(f"Qwen-VL API key loaded from {var}.")
                break

    if not key:
        raise EnvironmentError(
            "\n\nQwen-VL API key not found. Please do ONE of the following:\n"
            "  • Kaggle:      Add QWEN_VL_API_KEY to your Kaggle Secrets\n"
            "  • SiliconFlow: export QWEN_VL_API_KEY=sk-xxxxxxxx\n"
            "    (default provider; key from https://cloud.siliconflow.com/account/ak)\n"
            "  • DashScope:   export DASHSCOPE_API_KEY=sk-xxxxxxxx and set\n"
            "    QWEN_VL_BASE_URL / QWEN_VL_MODEL to the DashScope values\n"
        )

    os.environ["QWEN_VL_API_KEY"] = key
    return key
