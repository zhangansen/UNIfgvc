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
            "The token is required to download Qwen2.5-VL (gated model).\n"
            "Get your token at: https://huggingface.co/settings/tokens\n"
        )

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token  # older transformers compat
    return token
