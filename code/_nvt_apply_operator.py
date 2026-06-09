"""
NVT 原始 mask operator 参考实现（来自归档的 NVT 论文）。

来源：vstar_visual_necessity_audit.py:108-132（NVT V*Bench audit 代码）
说明：这是已经在 V*Bench 上跑通过 R015 实验、给出 +40 个百分点 selected-vs-random gap 的版本。
建议：**不要直接调用这个函数**，用 cfa_mask_operators.py 里的 bbox-adapted 版。

依赖：
    pip install Pillow
"""
from __future__ import annotations

from PIL import Image, ImageFilter


def apply_operator(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    cells: list[int],
    operator: str,
    distractor_image: Image.Image | None = None,
) -> Image.Image:
    """
    在 image 的指定 cells (索引到 boxes 列表) 上应用 mask operator。

    Args:
        image: 输入图像
        boxes: bbox 列表，每个 (x0, y0, x1, y1) pixel coords
        cells: 要操作的 cell 索引列表
        operator: 'zero' | 'blur' | 'matched_patch'
        distractor_image: matched_patch 时必填，作为像素来源

    Returns:
        被 mask 之后的 image 副本（原 image 不变）
    """
    out = image.copy()
    for cell in cells:
        box = boxes[cell]
        x0, y0, x1, y1 = box
        if operator == "zero":
            # NVT 实际填的是中性灰 (127,127,127)，不是纯黑
            # 选灰是因为 ImageNet 预处理后灰更接近 mean
            fill = Image.new("RGB", (x1 - x0, y1 - y0), (127, 127, 127))
            out.paste(fill, box)
        elif operator == "blur":
            # 半径自适应 bbox 大小，避免对小 bbox 过度 blur 或者对大 bbox 不够
            radius = max(6, min(x1 - x0, y1 - y0) // 6)
            crop = out.crop(box).filter(ImageFilter.GaussianBlur(radius=radius))
            out.paste(crop, box)
        elif operator == "matched_patch":
            if distractor_image is None:
                raise ValueError("matched_patch operator requires a distractor image")
            # 把 distractor 缩放到 image 大小，从相同 bbox 区域取一块替换
            # 关键设计：从 distractor 的"同位置"取，保证 spatial layout 相似
            resized = distractor_image.resize(image.size)
            out.paste(resized.crop(box), box)
        else:
            raise ValueError(f"Unknown operator: {operator}")
    return out
