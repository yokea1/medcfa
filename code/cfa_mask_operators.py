"""
MedCFA bbox-based counterfactual mask operators.

设计：CheXlocalize 给的是每个 (image, pathology) 对一个 bbox（不像 V*Bench 用 grid cell）。
这个模块就是 NVT `apply_operator` 的 bbox-adapted 包装。

三个 operator：
    - zero:           bbox 内填中性灰 (127,127,127)，最简单但引入 distribution-shift artifact
    - blur:           bbox 内 Gaussian blur，保留低频解剖背景、破坏高频病灶纹理
    - matched_patch:  用一张健康 CXR 同位置 patch 填充，视觉上仍然是真实 CXR（最严格的因果隔离）

依赖：
    pip install Pillow

示例：
    >>> from PIL import Image
    >>> from cfa_mask_operators import apply_bbox_mask, load_distractor
    >>> img = Image.open("path/to/cxr.jpg").convert("RGB")
    >>> distractor = load_distractor("path/to/healthy_cxr.jpg")
    >>> masked = apply_bbox_mask(img, (100, 100, 300, 300), "matched_patch", distractor)
    >>> masked.save("/tmp/masked.jpg")
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from _nvt_apply_operator import apply_operator

# --- 公开 API ---

VALID_OPERATORS = ("zero", "blur", "matched_patch")


def apply_bbox_mask(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    operator: str,
    distractor_image: Image.Image | None = None,
) -> Image.Image:
    """
    对单个 bbox 应用 mask operator。Bbox-adapted 版的 NVT apply_operator。

    Args:
        image: PIL Image（RGB）
        bbox: (x0, y0, x1, y1) pixel coords
        operator: 'zero' | 'blur' | 'matched_patch'
        distractor_image: matched_patch 时必填

    Returns:
        被 mask 的 image 副本（原 image 不变）

    Raises:
        ValueError: operator 不在 VALID_OPERATORS 里，或 matched_patch 缺 distractor
    """
    if operator not in VALID_OPERATORS:
        raise ValueError(
            f"operator must be one of {VALID_OPERATORS}, got '{operator}'"
        )
    if image.mode != "RGB":
        image = image.convert("RGB")
    return apply_operator(
        image=image,
        boxes=[bbox],
        cells=[0],
        operator=operator,
        distractor_image=distractor_image,
    )


def load_distractor(path: str | Path) -> Image.Image:
    """
    加载 distractor image（用于 matched_patch operator）。

    建议：选一张 CheXlocalize 里 **所有 pathology 都是 negative** 的图（i.e. 健康胸片），
    pre-compute 一次，所有 matched_patch 实验都用同一张。这样可复现。

    实现细节：CXR 经常是 16-bit grayscale，必须 convert('RGB')。
    """
    return Image.open(path).convert("RGB")


# --- Sanity helpers ---

def sanity_check_mask(
    original: Image.Image,
    masked: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    inside_ssim_max: float = 0.5,
    outside_ssim_min: float = 0.95,
) -> dict:
    """
    检查 mask 是否真的改了 bbox 内 + 没改 bbox 外。

    使用前确保 `pip install scikit-image numpy`

    Returns:
        dict with: ssim_inside, ssim_outside, pass_inside, pass_outside, pass_overall
    """
    import numpy as np
    from skimage.metrics import structural_similarity as ssim

    a = np.asarray(original.convert("L"), dtype=np.float32)
    b = np.asarray(masked.convert("L"), dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    x0, y0, x1, y1 = bbox
    h, w = a.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)

    # bbox 内 SSIM：应该低（说明 mask 真的改了像素）
    inside_a = a[y0:y1, x0:x1]
    inside_b = b[y0:y1, x0:x1]
    if inside_a.size == 0:
        ssim_inside = 1.0  # degenerate bbox
    else:
        ssim_inside = float(ssim(inside_a, inside_b, data_range=255.0))

    # bbox 外 SSIM：通过把 bbox 内置成同一值再算（直接算全图会被 bbox 拖低）
    mask = np.ones_like(a, dtype=bool)
    mask[y0:y1, x0:x1] = False
    a_out = a.copy()
    b_out = b.copy()
    a_out[~mask] = 0
    b_out[~mask] = 0
    ssim_outside = float(ssim(a_out, b_out, data_range=255.0))

    pass_inside = ssim_inside <= inside_ssim_max
    pass_outside = ssim_outside >= outside_ssim_min

    return {
        "ssim_inside": ssim_inside,
        "ssim_outside": ssim_outside,
        "pass_inside": pass_inside,
        "pass_outside": pass_outside,
        "pass_overall": pass_inside and pass_outside,
    }


# --- 自测 ---

if __name__ == "__main__":
    """
    Smoke test：
        python cfa_mask_operators.py path/to/any_image.jpg
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cfa_mask_operators.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = Image.open(img_path).convert("RGB")
    print(f"loaded {img_path}: size={img.size}")

    # Fake bbox in center
    w, h = img.size
    bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

    for op in VALID_OPERATORS:
        distractor = img.copy() if op == "matched_patch" else None
        out = apply_bbox_mask(img, bbox, op, distractor)
        out_path = f"/tmp/smoke_{op}.jpg"
        out.save(out_path)
        result = sanity_check_mask(img, out, bbox)
        print(f"[{op}] saved -> {out_path}  sanity={result}")

    print("\nSmoke test done. 用图片查看器打开 /tmp/smoke_*.jpg 确认效果。")
