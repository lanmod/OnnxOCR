"""Download or export Qwen3.5-2B ONNX files for OnnxOCR.

The model files are always placed under:

    onnxocr/models/qwen_2b

Recommended quick path:

    python examples/qwen35_2b_onnx.py download --variant q4
    python examples/qwen35_2b_onnx.py verify
    python examples/qwen35_2b_onnx.py run-python

Full PyTorch-to-ONNX export path:

    python examples/qwen35_2b_onnx.py convert

Notes:
    - Qwen3.5-2B is a large multimodal model. Exporting it locally requires
      enough RAM/disk and recent versions of transformers/optimum.
    - The pre-converted ONNX repository is the most reliable way to obtain
      the ONNX files.
    - Pure Python inference requires onnxruntime>=1.26 because older versions
      do not support the contrib ops used by the Qwen3.5-2B q4 ONNX graphs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TARGET_DIR = Path("onnxocr") / "models" / "qwen_2b"
READY_MODELSCOPE_REPO = "supersong/qwen2bonnx"
READY_HUGGINGFACE_REPO = "huggingworld/Qwen3.5-2B-ONNX"
READY_MODELSCOPE_SUBDIR = "models"
SOURCE_MODEL_REPO = "Qwen/Qwen3.5-2B"

COMMON_ALLOW_PATTERNS = [
    "README.md",
    "LICENSE*",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "images/demo.jpeg",
]

VARIANT_ALLOW_PATTERNS = {
    "q4": [
        "onnx/decoder_model_merged_q4.onnx*",
        "onnx/embed_tokens_q4.onnx*",
        "onnx/vision_encoder_fp16.onnx*",
    ],
    "q4f16": [
        "onnx/decoder_model_merged_q4f16.onnx*",
        "onnx/embed_tokens_q4f16.onnx*",
        "onnx/vision_encoder_q4f16.onnx*",
    ],
    "fp16": [
        "onnx/decoder_model_merged_fp16.onnx*",
        "onnx/embed_tokens_fp16.onnx*",
        "onnx/vision_encoder_fp16.onnx*",
    ],
    "full": [
        "onnx/*.onnx",
        "onnx/*.onnx_data*",
    ],
}

VERIFY_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "onnx/decoder_model_merged_q4.onnx",
    "onnx/decoder_model_merged_q4.onnx_data",
    "onnx/embed_tokens_q4.onnx",
    "onnx/embed_tokens_q4.onnx_data",
    "onnx/vision_encoder_fp16.onnx",
    "onnx/vision_encoder_fp16.onnx_data",
]

@dataclass(frozen=True)
class ExtractionTemplate:
    """Reusable OCR-to-JSON extraction template for a vertical scenario."""

    name: str
    description: str
    fields: list[str]
    rules: list[str]

    def build_prompt(self, ocr_text: str) -> str:
        schema = {field: "" for field in self.fields}
        rules = "\n".join(f"{index}. {rule}" for index, rule in enumerate(self.rules, start=1))
        return (
            f"你是{self.description}信息抽取器。下面是 OnnxOCR 的全量识别文本。\n\n"
            "请只输出一个 JSON 对象，不要解释，不要 Markdown，不要代码块。字段固定如下：\n"
            f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            f"规则：\n{rules}\n\n"
            f"OCR 文本：\n{ocr_text}\n"
        )


class Qwen35ONNX:
    """Pure Python Qwen3.5-2B ONNX inference helper."""

    def __init__(self, model_dir: str | Path = TARGET_DIR, providers: list[str] | None = None) -> None:
        ensure_dependency("numpy", "pip install numpy")
        ensure_dependency("tokenizers", "pip install tokenizers")
        ensure_dependency("onnxruntime", "pip install 'onnxruntime>=1.26.0'")

        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np = np
        self.ort = ort
        self.model_dir = Path(model_dir)
        self.providers = providers or ["CPUExecutionProvider"]

        if verify_files(self.model_dir, "q4") != 0:
            raise RuntimeError(f"Qwen3.5-2B ONNX files are not ready: {self.model_dir}")
        if _version_tuple(ort.__version__) < (1, 26, 0):
            raise RuntimeError(
                f"onnxruntime {ort.__version__} is too old for Qwen3.5-2B q4 ONNX. "
                "Install onnxruntime>=1.26.0."
            )

        onnx_dir = self.model_dir / "onnx"
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.embed_session = ort.InferenceSession(str(onnx_dir / "embed_tokens_q4.onnx"), providers=self.providers)
        self.decoder_session = ort.InferenceSession(
            str(onnx_dir / "decoder_model_merged_q4.onnx"),
            providers=self.providers,
        )
        self.vision_session = None

    def generate(
        self,
        prompt: str,
        image_path: str | Path | None = None,
        max_new_tokens: int = 128,
        min_new_tokens: int = 16,
        stop_at_sentence: bool = True,
        min_pixels: int = 1024,
        max_pixels: int = 4096,
    ) -> tuple[str, bool]:
        image = Path(image_path) if image_path else None
        image_grid_thw = None
        pixel_values = None

        if image is not None:
            ensure_dependency("PIL", "pip install pillow")
            from PIL import Image

            pixel_values, image_grid_thw = _prepare_qwen_image(
                image,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                np=self.np,
                Image=Image,
            )

        input_ids = _build_qwen_input_ids(self.tokenizer, prompt, image_grid_thw)
        inputs_embeds = self.embed_session.run(
            None,
            {"input_ids": self.np.asarray([input_ids], dtype=self.np.int64)},
        )[0]

        if image is not None:
            if self.vision_session is None:
                self.vision_session = self.ort.InferenceSession(
                    str(self.model_dir / "onnx" / "vision_encoder_fp16.onnx"),
                    providers=self.providers,
                )
            image_features = self.vision_session.run(
                None,
                {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            )[0]
            image_token_id = self.tokenizer.token_to_id("<|image_pad|>")
            image_token_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == image_token_id]
            if len(image_token_positions) != image_features.shape[0]:
                raise RuntimeError(
                    "Image token count does not match vision feature count: "
                    f"{len(image_token_positions)} tokens vs {image_features.shape[0]} features"
                )
            inputs_embeds[0, image_token_positions, :] = image_features

        generated_ids, stopped = _generate_qwen_tokens(
            decoder_session=self.decoder_session,
            embed_session=self.embed_session,
            inputs_embeds=inputs_embeds,
            input_length=len(input_ids),
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            eos_token_id=self.tokenizer.token_to_id("<|im_end|>"),
            tokenizer=self.tokenizer,
            stop_at_sentence=stop_at_sentence,
            np=self.np,
        )
        return self.tokenizer.decode(generated_ids).strip(), stopped


class QwenScenarioExtractor:
    """Template-based OCR text extractor powered by Qwen3.5-2B ONNX."""

    def __init__(
        self,
        qwen: Qwen35ONNX | None = None,
        model_dir: str | Path = TARGET_DIR,
        max_new_tokens: int = 256,
    ) -> None:
        self.qwen = qwen or Qwen35ONNX(model_dir)
        self.max_new_tokens = max_new_tokens

    def extract(
        self,
        ocr_text: str,
        template: ExtractionTemplate,
        max_new_tokens: int | None = None,
    ) -> tuple[dict[str, Any], str]:
        raw, _ = self.qwen.generate(
            prompt=template.build_prompt(ocr_text),
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            min_new_tokens=32,
            stop_at_sentence=False,
        )
        parsed = parse_json_object(raw)
        normalized = {field: str(parsed.get(field, "") or "").strip() for field in template.fields}
        return normalized, raw


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        cleaned = match.group(0)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            value = json.loads(repair_json(cleaned))
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Qwen3.5-2B ONNX files under onnxocr/models/qwen_2b.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download pre-converted Qwen3.5-2B ONNX files.")
    download_parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="Download source. ModelScope is the default for this project.",
    )
    download_parser.add_argument(
        "--repo-id",
        default="",
        help=(
            "Pre-converted ONNX model repository. Defaults to supersong/qwen2bonnx on ModelScope, "
            "or huggingworld/Qwen3.5-2B-ONNX on HuggingFace."
        ),
    )
    download_parser.add_argument(
        "--repo-subdir",
        default=READY_MODELSCOPE_SUBDIR,
        help="ModelScope repository subdirectory that contains the model files.",
    )
    download_parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_ALLOW_PATTERNS),
        default="q4",
        help="ONNX weight variant to download. q4 is recommended for local testing.",
    )
    download_parser.add_argument("--target-dir", default=str(TARGET_DIR), help="Where ONNX files will be stored.")
    download_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload files even if partial cache files exist. Useful after a consistency-check failure.",
    )
    download_parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Concurrent downloads. Lower values are more stable on unreliable networks.",
    )

    convert_parser = subparsers.add_parser("convert", help="Export Qwen3.5-2B from PyTorch/HF to ONNX.")
    convert_parser.add_argument("--model-id", default=SOURCE_MODEL_REPO, help="Source Hugging Face model id or path.")
    convert_parser.add_argument("--target-dir", default=str(TARGET_DIR), help="Where exported ONNX files will be stored.")
    convert_parser.add_argument("--task", default="image-text-to-text", help="Optimum ONNX export task.")
    convert_parser.add_argument("--opset", default="18", help="ONNX opset version.")
    convert_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to the exporter if the local transformers version requires it.",
    )

    verify_parser = subparsers.add_parser("verify", help="Verify required Qwen3.5-2B ONNX files exist.")
    verify_parser.add_argument("--target-dir", default=str(TARGET_DIR), help="Directory to verify.")
    verify_parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_ALLOW_PATTERNS),
        default="q4",
        help="Variant to verify. The default q4 checks q4 decoder/embed and fp16 vision encoder.",
    )

    py_run_parser = subparsers.add_parser(
        "run-python",
        help="Run a local Qwen3.5-2B ONNX smoke test with pure Python and ONNXRuntime.",
    )
    py_run_parser.add_argument("--target-dir", default=str(TARGET_DIR), help="Directory containing ONNX files.")
    py_run_parser.add_argument("--image", default="", help="Optional image path. If omitted, pure text inference is used.")
    py_run_parser.add_argument("--prompt", default="\u4f60\u597d\uff0c\u4ecb\u7ecd\u4e00\u4e0b\u4f60\u81ea\u5df1\u3002", help="Prompt text.")
    py_run_parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum new tokens.")
    py_run_parser.add_argument("--min-new-tokens", type=int, default=16, help="Minimum tokens before sentence-boundary stopping.")
    py_run_parser.add_argument(
        "--no-stop-at-sentence",
        action="store_true",
        help="Disable early stopping when a complete sentence is generated.",
    )
    py_run_parser.add_argument("--min-pixels", type=int, default=1024, help="Minimum vision pixels.")
    py_run_parser.add_argument("--max-pixels", type=int, default=4096, help="Maximum vision pixels.")

    args = parser.parse_args(argv)
    if args.command == "download":
        return download_ready_onnx(
            repo_id=args.repo_id,
            target_dir=Path(args.target_dir),
            variant=args.variant,
            source=args.source,
            repo_subdir=args.repo_subdir,
            force_download=args.force_download,
            max_workers=args.max_workers,
        )
    if args.command == "convert":
        return export_with_optimum(
            model_id=args.model_id,
            target_dir=Path(args.target_dir),
            task=args.task,
            opset=args.opset,
            trust_remote_code=args.trust_remote_code,
        )
    if args.command == "verify":
        return verify_files(Path(args.target_dir), args.variant)
    if args.command == "run-python":
        return run_python_onnx(
            target_dir=Path(args.target_dir),
            image_path=args.image,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            stop_at_sentence=not args.no_stop_at_sentence,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    return 1


def download_ready_onnx(
    repo_id: str,
    target_dir: Path,
    variant: str,
    source: str = "modelscope",
    repo_subdir: str = READY_MODELSCOPE_SUBDIR,
    force_download: bool = False,
    max_workers: int = 2,
) -> int:
    if source == "modelscope":
        return download_from_modelscope(
            repo_id=repo_id or READY_MODELSCOPE_REPO,
            target_dir=target_dir,
            variant=variant,
            repo_subdir=repo_subdir,
        )

    return download_from_huggingface(
        repo_id=repo_id or READY_HUGGINGFACE_REPO,
        target_dir=target_dir,
        variant=variant,
        force_download=force_download,
        max_workers=max_workers,
    )


def download_from_modelscope(repo_id: str, target_dir: Path, variant: str, repo_subdir: str) -> int:
    ensure_dependency("modelscope", "pip install modelscope")
    from modelscope import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    repo_prefix = f"{repo_subdir.strip('/')}/" if repo_subdir else ""
    allow_patterns = [repo_prefix + pattern for pattern in COMMON_ALLOW_PATTERNS + VARIANT_ALLOW_PATTERNS[variant]]
    print(f"Downloading {repo_id}/{repo_subdir} ({variant}) from ModelScope to {target_dir.resolve()}")
    cache_dir = Path(
        snapshot_download(
            repo_id,
            allow_patterns=allow_patterns,
            local_dir=str(target_dir),
        )
    )
    source_dir = target_dir / repo_subdir if repo_subdir else cache_dir
    if source_dir.exists() and source_dir != target_dir:
        copied = copy_model_files(source_dir, target_dir, COMMON_ALLOW_PATTERNS + VARIANT_ALLOW_PATTERNS[variant])
        if copied == 0:
            print(f"No model files matched under: {source_dir}", file=sys.stderr)
            return 1
    return verify_files(target_dir, variant)


def download_from_huggingface(
    repo_id: str,
    target_dir: Path,
    variant: str,
    force_download: bool = False,
    max_workers: int = 2,
) -> int:
    ensure_dependency("huggingface_hub", "pip install huggingface_hub")
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = COMMON_ALLOW_PATTERNS + VARIANT_ALLOW_PATTERNS[variant]
    print(f"Downloading {repo_id} ({variant}) to {target_dir.resolve()}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        force_download=force_download,
        max_workers=max_workers,
    )
    return verify_files(target_dir, variant)


def copy_model_files(source_dir: Path, target_dir: Path, patterns: Iterable[str]) -> int:
    copied = 0
    for pattern in patterns:
        for source_file in source_dir.glob(pattern):
            if not source_file.is_file():
                continue
            relative_path = source_file.relative_to(source_dir)
            target_file = target_dir / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            copied += 1
    return copied


def export_with_optimum(model_id: str, target_dir: Path, task: str, opset: str, trust_remote_code: bool) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    ensure_executable_or_module("optimum-cli", "optimum.exporters.onnx", 'pip install "optimum[onnxruntime]" transformers')

    cmd = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        "--model",
        model_id,
        "--task",
        task,
        "--opset",
        str(opset),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.append(str(target_dir))

    print("Running ONNX export command:")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            "\nONNX export failed. Qwen3.5-2B is a recent multimodal architecture, so your "
            "local transformers/optimum versions may not support direct export yet.\n"
            "Use the reliable pre-converted path instead:\n"
            "  python examples/qwen35_2b_onnx.py download --variant q4\n",
            file=sys.stderr,
        )
        return exc.returncode
    print(f"Exported ONNX files to {target_dir.resolve()}")
    return 0


def verify_files(target_dir: Path, variant: str) -> int:
    if variant == "q4":
        files = VERIFY_FILES
    else:
        files = COMMON_ALLOW_PATTERNS[:5] + _patterns_to_representative_files(VARIANT_ALLOW_PATTERNS[variant])

    missing = [path for path in files if not (target_dir / path).exists()]
    if missing:
        print(f"Qwen3.5-2B ONNX directory: {target_dir.resolve()}")
        print("Missing files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print(f"Qwen3.5-2B ONNX files are ready in: {target_dir.resolve()}")
    return 0


def run_python_onnx(
    target_dir: Path,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    stop_at_sentence: bool,
    min_pixels: int,
    max_pixels: int,
) -> int:
    if verify_files(target_dir, "q4") != 0:
        return 1

    ensure_dependency("numpy", "pip install numpy")
    ensure_dependency("PIL", "pip install pillow")
    ensure_dependency("tokenizers", "pip install tokenizers")
    ensure_dependency("onnxruntime", "pip install 'onnxruntime>=1.26.0'")

    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from tokenizers import Tokenizer

    if _version_tuple(ort.__version__) < (1, 26, 0):
        print(
            f"onnxruntime {ort.__version__} is too old for this q4 graph. "
            "Please upgrade with: pip install -U 'onnxruntime>=1.26.0'",
            file=sys.stderr,
        )
        return 1

    image = Path(image_path) if image_path else None
    if image is not None and not image.exists():
        print(f"Image not found: {image}", file=sys.stderr)
        return 1

    tokenizer = Tokenizer.from_file(str(target_dir / "tokenizer.json"))
    if image is None:
        pixel_values = None
        image_grid_thw = None
        input_ids = _build_qwen_input_ids(tokenizer, prompt)
    else:
        pixel_values, image_grid_thw = _prepare_qwen_image(
            image,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            np=np,
            Image=Image,
        )
        input_ids = _build_qwen_input_ids(tokenizer, prompt, image_grid_thw)

    providers = ["CPUExecutionProvider"]
    onnx_dir = target_dir / "onnx"
    embed_session = ort.InferenceSession(str(onnx_dir / "embed_tokens_q4.onnx"), providers=providers)
    decoder_session = ort.InferenceSession(str(onnx_dir / "decoder_model_merged_q4.onnx"), providers=providers)

    input_array = np.asarray([input_ids], dtype=np.int64)
    inputs_embeds = embed_session.run(None, {"input_ids": input_array})[0]
    if image is not None:
        vision_session = ort.InferenceSession(str(onnx_dir / "vision_encoder_fp16.onnx"), providers=providers)
        image_features = vision_session.run(
            None,
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
        )[0]

        image_token_id = tokenizer.token_to_id("<|image_pad|>")
        image_token_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == image_token_id]
        if len(image_token_positions) != image_features.shape[0]:
            print(
                "Image token count does not match vision feature count: "
                f"{len(image_token_positions)} tokens vs {image_features.shape[0]} features",
                file=sys.stderr,
            )
            return 1
        inputs_embeds[0, image_token_positions, :] = image_features

    generated_ids, stopped_by_eos = _generate_qwen_tokens(
        decoder_session=decoder_session,
        embed_session=embed_session,
        inputs_embeds=inputs_embeds,
        input_length=len(input_ids),
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        eos_token_id=tokenizer.token_to_id("<|im_end|>"),
        tokenizer=tokenizer,
        stop_at_sentence=stop_at_sentence,
        np=np,
    )

    _safe_print(tokenizer.decode(generated_ids).strip())
    if generated_ids and not stopped_by_eos and len(generated_ids) >= max_new_tokens:
        print(
            f"\n[warning] Output reached --max-new-tokens={max_new_tokens}; "
            "increase this value if the sentence is still incomplete.",
            file=sys.stderr,
        )
    return 0


def _prepare_qwen_image(image_path: Path, min_pixels: int, max_pixels: int, np, Image) -> tuple:
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    image = Image.open(image_path).convert("RGB")
    resized_height, resized_width = _smart_resize(
        image.height,
        image.width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    pixel_values = np.asarray(image).astype(np.float32) / 255.0
    pixel_values = (pixel_values - 0.5) / 0.5
    pixel_values = np.transpose(pixel_values, (2, 0, 1))
    pixel_values = np.stack([pixel_values] * temporal_patch_size, axis=0)

    grid_t = 1
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    patches = pixel_values.reshape(
        grid_t,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    patches = patches.reshape(grid_t * grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    image_grid_thw = np.asarray([[grid_t, grid_h, grid_w]], dtype=np.int64)
    return patches.astype(np.float32), image_grid_thw


def _build_qwen_input_ids(tokenizer, prompt: str, image_grid_thw=None) -> list[int]:
    if image_grid_thw is None:
        image_content = ""
    else:
        merge_size = 2
        image_pad_count = int(image_grid_thw[0].prod() // (merge_size * merge_size))
        image_content = "<|vision_start|>" + ("<|image_pad|>" * image_pad_count) + "<|vision_end|>"
    text = (
        f"<|im_start|>user\n{image_content}{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    return tokenizer.encode(text).ids


def _generate_qwen_tokens(
    decoder_session,
    embed_session,
    inputs_embeds,
    input_length: int,
    max_new_tokens: int,
    min_new_tokens: int,
    eos_token_id: int | None,
    tokenizer,
    stop_at_sentence: bool,
    np,
) -> tuple[list[int], bool]:
    feeds = {
        "inputs_embeds": inputs_embeds.astype(np.float32),
        "attention_mask": np.ones((1, input_length), dtype=np.int64),
        "position_ids": np.tile(np.arange(input_length, dtype=np.int64), (3, 1, 1)),
    }
    for input_meta in decoder_session.get_inputs()[3:]:
        feeds[input_meta.name] = _zero_past_input(input_meta.shape, np)

    outputs = decoder_session.run(None, feeds)
    next_token_id = int(np.argmax(outputs[0][0, -1]))
    past = _collect_past_outputs(decoder_session, outputs)
    generated_ids: list[int] = []
    stopped_by_eos = False

    for _ in range(max_new_tokens):
        generated_ids.append(next_token_id)
        if eos_token_id is not None and next_token_id == eos_token_id:
            stopped_by_eos = True
            break
        if stop_at_sentence and len(generated_ids) >= min_new_tokens:
            decoded = tokenizer.decode(generated_ids).strip()
            if _ends_with_sentence_boundary(decoded):
                stopped_by_eos = True
                break

        token_embeds = embed_session.run(None, {"input_ids": np.asarray([[next_token_id]], dtype=np.int64)})[0]
        total_length = input_length + len(generated_ids)
        feeds = {
            "inputs_embeds": token_embeds.astype(np.float32),
            "attention_mask": np.ones((1, total_length), dtype=np.int64),
            "position_ids": np.asarray([[[total_length - 1]], [[total_length - 1]], [[total_length - 1]]], dtype=np.int64),
        }
        feeds.update(past)
        outputs = decoder_session.run(None, feeds)
        next_token_id = int(np.argmax(outputs[0][0, -1]))
        past = _collect_past_outputs(decoder_session, outputs)

    return generated_ids, stopped_by_eos


def _ends_with_sentence_boundary(text: str) -> bool:
    if not text:
        return False
    return text.rstrip().endswith(("\u3002", "\uff01", "\uff1f", ".", "!", "?"))


def _safe_print(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _zero_past_input(shape: list, np):
    concrete_shape = []
    for dim in shape:
        if dim == "batch_size":
            concrete_shape.append(1)
        elif dim == "past_sequence_length":
            concrete_shape.append(0)
        elif isinstance(dim, int):
            concrete_shape.append(dim)
        else:
            concrete_shape.append(0)
    return np.zeros(concrete_shape, dtype=np.float32)


def _collect_past_outputs(decoder_session, outputs) -> dict:
    output_by_name = {meta.name: value for meta, value in zip(decoder_session.get_outputs(), outputs)}
    past = {}
    for input_meta in decoder_session.get_inputs()[3:]:
        output_name = (
            input_meta.name.replace("past_key_values.", "present.")
            .replace("past_conv.", "present_conv.")
            .replace("past_recurrent.", "present_recurrent.")
        )
        past[input_meta.name] = output_by_name[output_name]
    return past


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    def round_by_factor(number: int) -> int:
        return round(number / factor) * factor

    def ceil_by_factor(number: float) -> int:
        return math.ceil(number / factor) * factor

    def floor_by_factor(number: float) -> int:
        return math.floor(number / factor) * factor

    resized_height = max(factor, round_by_factor(height))
    resized_width = max(factor, round_by_factor(width))
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, floor_by_factor(height / beta))
        resized_width = max(factor, floor_by_factor(width / beta))
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = max(factor, ceil_by_factor(height * beta))
        resized_width = max(factor, ceil_by_factor(width * beta))
    return resized_height, resized_width


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for part in version.split("."):
        number = ""
        for char in part:
            if not char.isdigit():
                break
            number += char
        if number:
            parts.append(int(number))
    return tuple(parts)


def _patterns_to_representative_files(patterns: Iterable[str]) -> list[str]:
    representatives = []
    for pattern in patterns:
        if pattern.endswith(".onnx*"):
            representatives.append(pattern.removesuffix("*"))
        elif pattern.endswith("*.onnx"):
            continue
    return representatives


def ensure_dependency(module_name: str, install_hint: str) -> None:
    try:
        __import__(module_name)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {module_name}. Install it with: {install_hint}") from exc


def ensure_executable_or_module(executable: str, module_name: str, install_hint: str) -> None:
    try:
        __import__(module_name.split(".", 1)[0])
    except ImportError as exc:
        raise SystemExit(f"Missing exporter: {executable}. Install it with: {install_hint}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
