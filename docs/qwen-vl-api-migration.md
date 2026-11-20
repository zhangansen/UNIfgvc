# 迁移方案：本地 Qwen2.5-VL → Qwen-VL API（DashScope）

> 本文档记录"本地 Qwen2.5-VL → Qwen-VL API"迁移：现状分析、选型、逐文件改动、成本与风险。
> 方案撰写：2026-11-19。**第一轮实施：2026-11-20**（DashScope）→ **第二轮实施：2026-11-20**（默认 provider 改为硅基流动 SiliconFlow，见 §0.1）。§0/§0.1 是权威变更记录，日后查看代码时据此对照。

---

## 0. 实施记录（2026-11-20，已全部应用到代码）

| 文件 | 改动内容 |
|---|---|
| `models/cdv_captioner.py` | **整文件替换**。删除本地 transformers 4-bit 推理（`Qwen2_5_VLForConditionalGeneration` / `BitsAndBytesConfig` / `qwen_vl_utils`），改用 `openai` SDK 调 DashScope OpenAI 兼容端点。新增：`load_qwen()` 初始化客户端并校验 `DASHSCOPE_API_KEY`；`_pil_to_data_uri()`（≤1024px JPEG base64）；`_convert_content()` 消息格式转换；`_run_qwen()` → `chat.completions.create`（`max_tokens` / `temperature=0` / SDK 内建 5 次重试）。**三个 stage 与 fast 模式的 prompt 逐字节未动**，对外函数签名全部不变。 |
| `utils/auth.py` | ① `setup_hf_token()` 报错文案由 "required to download Qwen2.5-VL (gated model)" 改为 "used to download the CLIP / DINOv2 encoders"（逻辑未变）；② 文件末尾新增 `setup_dashscope_api_key()`（Kaggle Secrets → 环境变量，缺失时明确报错）。 |
| `utils/__init__.py` | 导出行扩为 `from .auth import setup_hf_token, setup_dashscope_api_key`。 |
| `scripts/build_gallery.py` | import 行加入 `setup_dashscope_api_key`；`main()` 里 `setup_hf_token()` 之后调用 `setup_dashscope_api_key()`（共 2 行）。 |
| `scripts/evaluate.py` | 同上（共 2 行）。 |
| `configs/default.yaml` | `qwen_model_id` 改为 `"qwen-vl-max-latest"`；新增 `qwen_api_base`、`qwen_max_image_side: 1024`。注：脚本目前不读取该 yaml（实际生效靠环境变量/模块常量），此处仅保持文档一致。 |
| `requirements.txt` | 新增 `openai>=1.30.0`；移除 `bitsandbytes`、`qwen-vl-utils`、`hf_transfer`；`transformers` / `accelerate` 保留（CLIP/DINOv2 编码器仍用）。 |
| `README.md` | Models 一节：`Qwen2.5-VL-7B-Instruct (4-bit)` → `Qwen-VL API (DashScope, default qwen-vl-max-latest)`；密钥配置一节：改为 `DASHSCOPE_API_KEY` 为主，HF token 降级为"仅编码器下载、可选"。 |

**明确未改动**：`gallery/builder.py`、`gallery/retrieval.py`、`eval/evaluator.py`、`eval/metrics.py`、`models/encoders.py`、`models/__init__.py`（`load_qwen` 保留原名，故无需改）、`datasets/*`。

**实施后验证状态**（2026-11-20）：① 5 个改动的 .py 文件 `py_compile` 通过；② `openai` 已安装（3.55.1，`NO_PROXY="*" pip install` 直连）；③ `models/cdv_captioner.py` 以独立模块导入成功（默认模型/端点常量正确），无 key 时 `load_qwen()` 按预期抛出明确报错；④ **真实 API 冒烟测试待拿到 `DASHSCOPE_API_KEY` 后执行**（见 §10）。

**明确未实施**（可选优化，见 §7）：eval 端 CDV 去重缓存（`clip_cdv`/`full` 两模式仍会对同一张测试图各调一次 VLM）。

---

## 0.1 第二轮变更（2026-11-20）：默认 provider 由 DashScope 改为硅基流动（SiliconFlow）

起因：用户要求改调硅基流动 API（开源 7B 在硅基远便宜于百炼）。**第一轮的全部传输层设计不变**（openai SDK / OpenAI 兼容模式 / data URI 传图 / 重试），只换默认端点、模型名与 key 解析。

| 文件 | 改动内容 |
|---|---|
| `models/cdv_captioner.py` | ① 默认 `QWEN_VL_MODEL`：`qwen-vl-max-latest` → `Qwen/Qwen2.5-VL-7B-Instruct`（硅基用 HF 风格带斜杠的 id；另有 `Pro/Qwen/Qwen2.5-VL-7B-Instruct` 加速变体可选）；② 默认 `QWEN_VL_BASE_URL` → `https://api.siliconflow.cn/v1`；③ `load_qwen()` 的 key 解析改为链式：`QWEN_VL_API_KEY` → `SILICONFLOW_API_KEY` → `DASHSCOPE_API_KEY`（保留 DashScope 只需设回 base_url + model + 对应 key）；④ 文件头 docstring 与报错信息同步更新。prompt、stage 逻辑、`max_tokens`/`temperature` 等全部未动。 |
| `utils/auth.py` | `setup_dashscope_api_key()` **删除**，替换为 provider 无关的 `setup_vlm_api_key()`：Kaggle Secret 名固定 `QWEN_VL_API_KEY`；本地按同一 env 链解析；导出为 `QWEN_VL_API_KEY`。 |
| `utils/__init__.py` | 导出改为 `setup_vlm_api_key`。 |
| `scripts/build_gallery.py`、`scripts/evaluate.py` | import 与调用由 `setup_dashscope_api_key` 改为 `setup_vlm_api_key`（各 2 行，位置不变）。 |
| `configs/default.yaml` | `qwen_model_id` → `"Qwen/Qwen2.5-VL-7B-Instruct"`；`qwen_api_base` → 硅基端点；新增 `qwen_api_key_env` 说明行。 |
| `README.md` | Models 一节与密钥配置一节改为 SiliconFlow（key 控制台 https://cloud.siliconflow.com/account/ak）。 |

**硅基流动要点**（与 DashScope 的差异）：
- 端点：国内 `https://api.siliconflow.cn/v1`（官方英文文档用 `.com` 国际端点，勿混用 key）；key 在 https://cloud.siliconflow.com/account/ak 创建
- 模型 id 是 HF 风格：`Qwen/Qwen2.5-VL-7B-Instruct`（不是百炼的 `qwen-vl-max` 命名）；同系列还有 32B/72B 可直接替换模型名使用
- base64 传图必须带 MIME 前缀 `data:image/jpeg;base64,...`——现有 `_pil_to_data_uri()` 已满足（常见坑：缺前缀会导致"模型不认图"）
- 图片计费：Qwen 系列按像素折算 token，1024×1024 ≈ 1369 token（与原估算一致）；`detail=low` 可压到 256 token/图但会缩到 448×448，细粒度分类不建议
- 价格：第三方追踪显示 7B 输入/输出约 $0.05/M 量级或更低（部分渠道标注免费），**远低于**百炼同模型；以硅基控制台价格页为准

**回切 DashScope 的方法**（零代码改动）：
```bash
export QWEN_VL_MODEL=qwen-vl-max-latest
export QWEN_VL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export DASHSCOPE_API_KEY=sk-xxx
```

---

## 1. 现状：VLM 在项目中的位置

当前 CDV 描述由**本地 transformers 加载的 Qwen2.5-VL-7B-Instruct（4-bit）**生成，涉及 VLM 的代码只有一个文件：

| 文件 | 现状 | API 化后 |
|---|---|---|
| `models/cdv_captioner.py` | 加载本地 4-bit 模型，`_run_qwen()` 本地 generate | **重写传输层**（本文档 §4，唯一核心改动） |
| `models/__init__.py` | 导出 `load_qwen, full_cdv_caption` | 不改（保留 `load_qwen` 函数名） |
| `scripts/build_gallery.py` | L66 调 `load_qwen()` | 不改（函数语义变为"初始化 API 客户端"） |
| `scripts/evaluate.py` | L85 调 `load_qwen()` | 不改（同上） |
| `gallery/builder.py` | 只调 `full_cdv_caption()`，不接触 VLM | **不改**。checkpoint 机制原样可用 |
| `eval/evaluator.py` | 查询图片在 `clip_cdv`/`full` 两个模式下各调一次 `full_cdv_caption` | 功能上不改也能跑；**建议**做去重缓存（§7，省一半 API 费） |
| `models/encoders.py` | CLIP / DINOv2 本地小模型 | 不改（与 VLM 无关） |
| `utils/auth.py` | 只解析 HF token | 新增 DashScope key 解析函数（§5） |
| `configs/default.yaml` | `qwen_model_id` 指向 HF 仓库 id | 改为 API 模型名（§6；注意该 yaml 目前只是文档，脚本并不读取它） |
| `requirements.txt` | 含 bitsandbytes / qwen-vl-utils | 增 `openai`，删本地 VLM 依赖（§6） |
| `README.md` | 说明本地 Qwen + HF token | 更新为 API + DASHSCOPE_API_KEY（§6） |

**关键结论**：三个 CDV stage 的 prompt 文本保持逐字节不变，只替换"消息怎么发出去"的传输层，保证新旧描述风格可比、`gallery/builder.py` 与 `eval/` 完全无感知。

---

## 2. API 选型

采用 **阿里云百炼（DashScope）的 OpenAI 兼容模式**，用官方 `openai` Python SDK 调用：

| 项 | 值 |
|---|---|
| Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1`（北京区） |
| 鉴权 | 环境变量 `DASHSCOPE_API_KEY`（控制台 https://bailian.console.aliyun.com → API-KEY 管理） |
| SDK | `openai>=1.30`（复用其自动重试：429/5xx/连接错误，指数退避） |
| 图片格式 | `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`，本地 PIL 图转 base64 data URI；**单条消息多图**（Stage 1 发 4 张）已支持 |

选 OpenAI 兼容模式而非 `dashscope` 原生 SDK 的原因：接口形态通用，以后不改代码即可切到 SiliconFlow、自建 vLLM 等任何 OpenAI 兼容端点（只改 base_url + key）。

### 模型名选择（`QWEN_VL_MODEL`）

| 模型 ID | 说明 | 适用 |
|---|---|---|
| `qwen-vl-max-latest` | 商用旗舰，别名跟随官方升级 | **默认推荐** |
| `qwen-vl-plus-latest` | 商用高性价比 | 大批量跑数 |
| `qwen2.5-vl-72b-instruct` | 开源权重快照，与本地 7B 同家族 | 想和旧结果最接近时用 |
| `qwen-vl-max-2025-01-25`（示例） | 带日期的固定快照 | 论文实验可比性：跑分时**固定快照**，避免 `-latest` 漂移 |
| `qwen3-vl-*` 系列 | 更新的一代 | 有社区报告其在 OpenAI 兼容 chat 端点 404（与官方文档矛盾，可能分区域/分时间）。**用前先在控制台/小请求验证** |

参考：[Qwen API via OpenAI Chat Completions](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)、[Qwen-VL OpenAI 兼容文档](https://www.alibabacloud.com/help/en/model-studio/qwen-vl-compatible-with-openai.md)、[视觉理解文档](https://www.alibabacloud.com/help/en/model-studio/visual-understanding-hk)。

---

## 3. 参数映射（本地 → API）

| 本地 generate() | API chat.completions.create() |
|---|---|
| `max_new_tokens=N` | `max_tokens=N`（各 stage 数值不变：80 / 200 / 256） |
| `do_sample=False` | `temperature=0, top_p=1`（描述尽量确定性，保证可复现） |
| `qwen_vl_utils` 消息 `{"type": "image", "image": PIL}` | `{"type": "image_url", "image_url": {"url": data_uri}}`（转换函数内置） |
| 4-bit 显存加载 | 无。VLM 零显存，本机只剩 CLIP+DINOv2（约 1–2 GB，CPU 也能跑） |

注意：个别新模型（如 thinking 系列）不接受自定义 temperature，若返回 400 且报错提及 temperature/top_p，去掉这两个参数重试即可（在代码注释中已标注）。

---

## 4. 核心改动：`models/cdv_captioner.py`（已于 2026-11-20 应用，以下代码即当前文件内容）

```python
"""
models/cdv_captioner.py — 3-stage CDV captioner using the Qwen-VL API
(DashScope, OpenAI-compatible mode).

Drop-in replacement for the local Qwen2.5-VL version:
  • stage1..3 / fast_cdv_caption / full_cdv_caption keep identical
    prompts and signatures — gallery/builder.py and eval/ need no change.
  • load_qwen() now initialises the API client instead of loading weights.
"""

import base64
import io
import os

from PIL import Image
from openai import OpenAI

# ── Configuration (env-overridable) ─────────────────────────────────────────
QWEN_VL_MODEL       = os.environ.get("QWEN_VL_MODEL", "qwen-vl-max-latest")
QWEN_VL_BASE_URL    = os.environ.get(
    "QWEN_VL_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
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
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "\n\nDASHSCOPE_API_KEY not set. Get one at the Bailian console:\n"
            "  https://bailian.console.aliyun.com/  (API-KEY management)\n"
            "Then: export DASHSCOPE_API_KEY=sk-xxxxxxxx\n"
        )
    QWEN_VL_MODEL = model_id
    _client = OpenAI(
        api_key=api_key,
        base_url=QWEN_VL_BASE_URL,
        timeout=120,       # seconds per request (multi-image Stage 1 can be slow
        max_retries=5,     # SDK auto-retries 429 / 5xx / connection errors
    )
    print(f"Qwen-VL API client ready | model={model_id} | base={QWEN_VL_BASE_URL}")


def _pil_to_data_uri(img: Image.Image,
                     max_side: int = QWEN_MAX_IMAGE_SIDE) -> str:
    """Encode a PIL image as a base64 JPEG data URI.

    Client-side resize keeps upload small; the service re-tokenises anyway.
    <=1024px is roughly 1-1.5k vision tokens per image.
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
        # deterministic captions (was do_sample=False).
        # If a newer model rejects these with HTTP 400, remove them.
        temperature=0.0,
        top_p=1.0,
    )
    return (resp.choices[0].message.content or "").strip()


# ── Stage 1: Discriminative Region Discovery ─────────────────────────────────
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


# ── Stage 3: Attribute Feature Summarization ─────────────────────────────────
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


# ── Fast single-prompt fallback ──────────────────────────────────────────────
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
```

要点：
- **prompt 文本与原文件逐字节一致**（连破折号 `–` 都保留），确保新旧描述可比。
- stage 函数仍按 `qwen_vl_utils` 风格组装消息，`_run_qwen()` 内部统一转换，diff 最小。
- `load_qwen` 保留原名 → `models/__init__.py`、两个脚本零改动。

---

## 5. 次要改动

### 5.1 `utils/auth.py` — 追加函数（不改现有 `setup_hf_token`）

```python
def setup_dashscope_api_key(secret_name: str = "DASHSCOPE_API_KEY") -> str:
    """Resolve the DashScope API key: Kaggle Secrets → env var. Exports to env."""
    token = None
    # 1. Kaggle Secrets (when running on Kaggle)
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(secret_name)
        print("DASHSCOPE_API_KEY loaded from Kaggle Secrets.")
    except Exception:
        pass
    # 2. Environment variable
    if not token:
        token = os.environ.get(secret_name)
        if token:
            print("DASHSCOPE_API_KEY loaded from environment variable.")
    if not token:
        raise EnvironmentError(
            "\n\nDASHSCOPE_API_KEY not found. Create one at:\n"
            "  https://bailian.console.aliyun.com/  (API-KEY management)\n"
            "Then: export DASHSCOPE_API_KEY=sk-xxxxxxxx\n"
        )
    os.environ[secret_name] = token
    return token
```

并在 `utils/__init__.py` 导出它。同时把 `setup_hf_token()` 报错文案中 "required to download Qwen2.5-VL (gated model)" 改为 "used to download CLIP / DINOv2 encoders"（Qwen 不再从 HF 下载，CLIP/DINO 非 gated）。

### 5.2 `scripts/build_gallery.py` 与 `scripts/evaluate.py` — 各加 2 行（可选但建议）

```python
from utils import setup_hf_token, setup_dashscope_api_key   # 扩充原 import
...
    setup_hf_token()
    setup_dashscope_api_key()   # 新增：在 load_qwen() 之前
```

（不加也能跑：`load_qwen()` 自带缺 key 报错；加它是为了 Kaggle Secret 自动捡取。）

### 5.3 `configs/default.yaml`

```yaml
# ── Models ────────────────────────────────────────────────────────────────
clip_model_id: "openai/clip-vit-base-patch32"
dino_model_id: "facebook/dinov2-base"
qwen_model_id: "qwen-vl-max-latest"        # was "Qwen/Qwen2.5-VL-7B-Instruct"
qwen_api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
qwen_max_image_side: 1024
```

> 注意：当前脚本并不读取该 yaml（全靠 argparse 默认值 + 模块常量/环境变量）。yaml 改动只是让文档与实际一致；若想让 yaml 真正生效需另行接线（非本次必须）。

### 5.4 `requirements.txt`

```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.40.0      # 仍需：CLIP / DINOv2 编码器
accelerate>=0.27.0        # 仍需：编码器 device_map
openai>=1.30.0            # 新增：Qwen-VL API（DashScope OpenAI 兼容模式）
Pillow>=10.0.0
scipy>=1.11.0
numpy>=1.24.0
pyyaml>=6.0
# 删除：bitsandbytes / qwen-vl-utils / hf_transfer —— 仅本地 VLM 后端需要
```

附带收益：bitsandbytes 没有官方 Windows wheel，是本机（Windows 11）跑本地 4-bit 的最大障碍，API 化后直接消除；显存需求从约 10 GB 降到 1–2 GB（仅编码器），CPU 亦可。

### 5.5 `README.md`

- "Models" 一节：`Qwen2.5-VL-7B-Instruct (4-bit)` → `Qwen-VL API (DashScope, OpenAI-compatible, default qwen-vl-max-latest)`。
- 安装/配置一节：新增 `DASHSCOPE_API_KEY` 设置说明（Kaggle Secret 或环境变量），HF token 说明降级为"仅编码器下载用，非 gated，可不设"。

---

## 6. 不需要改的文件（明确排除，避免误伤）

- `gallery/builder.py` —— 只依赖 `full_cdv_caption(image, class_name, ref_imgs, ref_names, fast=...)` 签名；checkpoint schema（`cdv_texts` / `done_set` 等）不变，**断点续跑原样生效**。
- `gallery/retrieval.py`、`eval/metrics.py` —— 纯张量计算。
- `eval/evaluator.py` —— 功能上不改即可运行（见 §7 的可选优化）。
- `datasets/` —— 与 VLM 无关。

---

## 7. 可选优化（本次未实施）：eval 端 CDV 去重

现状（`eval/evaluator.py`）：默认跑 4 个 ablation 模式，其中 `clip_cdv` 和 `full` **各自独立**对同一张测试图调 `full_cdv_caption` → 同一张图描述生成 2 遍，API 调用与费用 ×2；且 eval 循环没有 checkpoint，中途断网该模式从头再来。

最小改法：在 `run_ablation` 的模式循环**之前**，对测试集每张图先算好 `cdv_t`（含 CLIP text 编码），存入 `{img_path: cdv_t}` 字典；`_query_vec_for_mode` 里 `clip_cdv`/`full` 直接查表。这样每张测试图只调一次 VLM（3-stage 下省一半调用），并把 VLM 失败风险集中到前置一段（可顺带把描述文本缓存到磁盘 json，键为 `(img_path, fast, model)`，重跑免费）。此改动不影响精度结果，只影响耗时与费用。

---

## 8. 成本与调用次数估算（以 CUB 为例：200 类 × 4-shot = 800 张支持图，600 张测试图）

| 阶段 | 3-stage 模式 | fast 模式 |
|---|---|---|
| build gallery | 800 × 3 = **2400 次** | 800 × 1 = 800 次 |
| eval（默认 4 模式，未去重） | 600 × 2 × 3 = **3600 次** | 600 × 2 = 1200 次 |
| eval（去重后，§7） | 1800 次 | 600 次 |

- Stage 1 单次请求带 4 张图（目标 + 3 个易混类参考），按 ≤1024px 每图约 1–1.5k 视觉 token 估算，单次输入约 5–7k token；Stage 2 单图；Stage 3 纯文本。
- 计费以控制台价格页为准（`qwen-vl-plus-latest` 明显便宜于 max，大批量调试期建议先用 plus，正式跑分再换 max/固定快照）。

---

## 9. 本机（Windows 11 + Clash）环境注意事项

1. **网络**：`api.siliconflow.cn` 与 `dashscope.aliyuncs.com` 都是国内域名，直连即可。系统代理（Clash，`127.0.0.1:7897`）通常按规则对国内域名走 DIRECT；若请求挂起，给该进程设 `NO_PROXY=api.siliconflow.cn`（bash）或 `$env:NO_PROXY="api.siliconflow.cn"`（PowerShell）。
2. **装 openai 包**：本机 pypi 经代理会挂——用 `NO_PROXY="*" pip install "openai>=1.30"` 直连，或 `-i https://mirrors.ustc.edu.cn/pypi/simple`。
3. **控制台编码 GBK**：新代码 print 全部用 ASCII，无风险；后续 agent 若加中文日志，中文在 GBK 内安全，但**不要** print `✓ ✗ → •` 等 GBK 外符号；读写任何文件显式 `encoding="utf-8"`（本文档即 UTF-8 保存）。
4. **key 管理**：PowerShell 临时 `$env:DASHSCOPE_API_KEY="sk-..."`；bash `export DASHSCOPE_API_KEY=sk-...`。不要写进代码或提交到仓库。

---

## 10. 验证步骤（§0 + §0.1 改动已实施；步骤 1 已做语法级验证，2–5 待有 key 后依序执行）

```bash
# 1) 安装依赖（本机直连 pypi）—— 已完成：openai 3.55.1 装于 E:\miniconda3 base
NO_PROXY="*" pip install "openai>=1.30"

# 2) 设置 key（硅基流动，https://cloud.siliconflow.com/account/ak）
export QWEN_VL_API_KEY=sk-xxxxxxxx

# 3) 冒烟测试：单图 fast CDV（在仓库根目录、UTF-8 环境）
PYTHONUTF8=1 python -c "
import sys; sys.path.insert(0, '.')
from PIL import Image
from models.cdv_captioner import load_qwen, fast_cdv_caption
load_qwen()
img = Image.open(r'F:/path/to/some/test.jpg').convert('RGB')
print(fast_cdv_caption(img))
"
# 预期：打印一段英文判别性描述；401=key 错，404=模型名错（用 Qwen/Qwen2.5-VL-7B-Instruct，注意斜杠）

# 4) 小规模真实构建（选一个小数据集或先 --fast_cdv）
python scripts/build_gallery.py --dataset pets --data_root <pets路径> \
    --out_dir results/pets_api --k_shot 1 --fast_cdv
# 预期：正常出进度条与 checkpoint；中断重跑应显示 "Resumed checkpoint"

# 5) 全量 3-stage 构建 + 评测
python scripts/build_gallery.py --dataset pets --data_root <pets路径> --out_dir results/pets_api
python scripts/evaluate.py  --dataset pets --data_root <pets路径> \
    --gallery_dir results/pets_api --out_dir results/pets_api
```

---

## 11. 风险与兼容性清单

| 风险 | 说明 | 对策 |
|---|---|---|
| 新旧 checkpoint 混用 | `out_dir` 里若有本地模型时期建的 `gallery_checkpoint.pt`，续跑会混合两代模型的描述 | API 模式用**新的 out_dir**（如 `results/cub_api`），或删旧 checkpoint |
| 模型名不可用 | `qwen3-vl-*` 在 OpenAI 兼容端点有 404 的社区报告 | 默认用 `qwen-vl-max-latest` / `qwen-vl-plus-latest`；新模型先冒烟验证 |
| `-latest` 别名漂移 | 跑分期间官方升级会导致前后描述风格不一致 | 正式实验固定带日期快照 |
| temperature 被拒 | 个别 thinking 系模型不接受自定义采样参数 | 报 400 时去掉 `temperature/top_p` |
| 限流（429） | 3-stage 串行每图 3 次调用，峰值不高，但 eval 未去重时翻倍 | SDK `max_retries=5` 自动退避；长跑先做 §7 去重 |
| 输出语言漂移 | API 版 max 对 prompt 的遵守度可能与本地 7B 不同（如夹带中文） | prompt 未变；若出现，可在 `_SYS_PROMPT` 末尾加 "Respond in English only."（会破坏"逐字节一致"原则，需记录） |
| 长评测无断点 | eval 循环不落盘，中断丢失该模式进度 | §7 前置缓存后风险大幅降低 |

---

## 12. 参考资料

- [Qwen API（OpenAI 兼容 Chat）— 阿里云](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)
- [Qwen-VL OpenAI 兼容 — Alibaba Cloud](https://www.alibabacloud.com/help/en/model-studio/qwen-vl-compatible-with-openai.md)
- [视觉理解（VL）— Alibaba Cloud](https://www.alibabacloud.com/help/en/model-studio/visual-understanding-hk)
- 百炼控制台（申请 API Key）：https://bailian.console.aliyun.com/

---

## 13. Kaggle 部署速查（2026-11-20 补充）

### 前置准备
1. **Secrets**（Notebook 右侧 Add-ons → Secrets）：
   - `QWEN_VL_API_KEY` = 硅基流动 key（`setup_vlm_api_key()` 自动捡取）
   - `HF_TOKEN` = HF token（`setup_hf_token()` 缺失会**直接抛错**；CLIP/DINOv2 下载用，非 gated，任意 token 即可）
2. **Settings**：Internet 开启（需手机验证账号）；Accelerator 选 **GPU T4**（可选——VLM 已 API 化，只剩 CLIP/DINOv2 编码器，CPU 能跑但慢，bitsandbytes 依赖已删除、任何 GPU 都兼容）
3. **代码上传**：本地打包上传为 Dataset（Kaggle 页面 New Dataset），或放 GitHub 后 notebook 里 `!git clone`：
   ```bash
   cd f:/project/trainfree && tar --exclude=UNIfgvc/results -czf unifgvc.tar.gz UNIfgvc
   ```
4. **数据挂载**：README 列的公开数据集直接 "+ Add Input"。CUB 用 `wenewone/cub2002011`，加载器要求 `<data_root>/CUB_200_2011/images/...`，即 `--data_root /kaggle/input/cub2002011`（挂上后先 `!ls` 确认层级）

### Notebook cells
```bash
!pip install -q "openai>=1.30"          # 只补这一个；别装 -r requirements.txt（会重装 torch）

%cd /kaggle/working
!cp -r /kaggle/input/<代码dataset名>/UNIfgvc .
%cd UNIfgvc

# 冒烟：小数据集 + fast（约 1/5 成本）
!python scripts/build_gallery.py --dataset pets --data_root /kaggle/input/<pets> \
    --out_dir /kaggle/working/results/pets --k_shot 1 --fast_cdv

# 正式：CUB 全量三阶段
!python scripts/build_gallery.py --dataset cub --data_root /kaggle/input/cub2002011 \
    --out_dir /kaggle/working/results/cub --k_shot 4
!python scripts/evaluate.py --dataset cub --data_root /kaggle/input/cub2002011 \
    --gallery_dir /kaggle/working/results/cub --out_dir /kaggle/working/results/cub
```

### 运行注意
- **out_dir 必须放 `/kaggle/working/` 下**：gallery checkpoint（每 50 张落盘）随 notebook 保存持久化，session 断了重跑同 cell 会自动 `Resumed checkpoint`。单次 GPU session 上限约 9–12h，T4 每周 30h 免费额度
- **eval 无断点**：长评测建议 `--modes` 分次跑（如先 `clip_only clip_dino`，再 `clip_cdv full`），或先实施 §7 去重
- **Kaggle（海外机房）→ api.siliconflow.cn**：通常可直连；若慢/不通，设 `QWEN_VL_BASE_URL=https://api.siliconflow.com/v1`（国际端点——注意 cn/com 控制台账号与 key 可能独立，用哪个端点就用哪个站申请的 key）
- **HF 缓存**：每次 session 会重新下载 CLIP/DINOv2（约 1GB）。可选加速：cell 最前面 `import os; os.environ["HF_HOME"]="/kaggle/working/hf_cache"`（占 working 目录 20GB 配额）
- 结果下载：Run All 完成后右侧 Output 里打包下载 `results/`，或 `!cat results/cub/eval_results.csv`
