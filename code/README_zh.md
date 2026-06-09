# code/ —— 参考代码用法

这个目录里有 **2 个文件**：

| 文件 | 用途 | 你要不要改 |
|---|---|---|
| `_nvt_apply_operator.py` | 来自归档 NVT 论文的原始 mask 实现 | **不要改**，只读 |
| `cfa_mask_operators.py` | MedCFA 用的 bbox-adapted 包装 + sanity check | 可以扩展，但保持 backward compat |

## 1. 直接 smoke test（验证代码能跑）

环境配好后（见 `../setup/README_zh.md`），先跑一个 smoke：

```bash
cd code/
python cfa_mask_operators.py /path/to/any_cxr_image.jpg
```

会在 `/tmp/` 下生成三张 `smoke_{zero,blur,matched_patch}.jpg`，用图片查看器打开看看 mask 效果对不对。

预期效果：
- `zero`：中央区域是中性灰
- `blur`：中央区域明显模糊，边界外清晰
- `matched_patch`：中央区域被另一块图像 patch 替换（smoke 里 distractor 就是原图本身，所以视觉上区别不大；真实数据上要换一张健康胸片）

如果 smoke 跑不通，**先排查 PIL 安装 + 图像是否能正常打开**，再 ping Qizhen。

## 2. 在你自己的 audit 脚本里调用

最小化使用方式：

```python
from PIL import Image
from cfa_mask_operators import apply_bbox_mask, load_distractor, sanity_check_mask

# 一次性 load distractor（建议在 main script 里只 load 一次）
distractor = load_distractor("path/to/one_healthy_chest_xray.jpg")

# 对每张 (image, pathology) 跑三个 operator
img = Image.open("path/to/positive_chexlocalize_image.jpg").convert("RGB")
bbox = (123, 45, 234, 178)  # 从 CheXlocalize bbox JSON 来

for op in ["zero", "blur", "matched_patch"]:
    masked = apply_bbox_mask(img, bbox, op, distractor)
    # ... 把 masked 喂给 VLM，记录答案 ...

    # 可选：sanity check
    sanity = sanity_check_mask(img, masked, bbox)
    if not sanity["pass_overall"]:
        # 记录 fail，最后过滤掉这条
        pass
```

## 3. Sanity check 阈值的含义

```python
sanity_check_mask(original, masked, bbox,
                  inside_ssim_max=0.5,     # bbox 内 SSIM 必须 ≤ 0.5（确认 mask 真改了）
                  outside_ssim_min=0.95)   # bbox 外 SSIM 必须 ≥ 0.95（确认 mask 没干扰其他区域）
```

- bbox **内** SSIM > 0.5：mask 效果太弱（图基本没变）→ **fail**
- bbox **外** SSIM < 0.95：mask 影响到了其他区域 → **fail**（多半是 paste 边界 bug）

两个都过才 `pass_overall=True`。**建议在 D2 sanity 阶段对 100 张测一下，确认 fail rate < 10%**。

## 4. 还需要补的代码（HANDOFF §10 工作清单里的 W3-W10）

你要自己写、放到 `scripts/` 下的（**不在这个包里**，由你创建）：

| 文件 | 用途 |
|---|---|
| `cfa_build_pairs.py` | 把 CheXlocalize bbox JSON 转 VQA pairs JSONL（schema 见 HANDOFF §11） |
| `cfa_sanity_full.py` | 跑完整 sanity（SSIM + torchxrayvision classifier confidence drop） |
| `cfa_model_adapters.py` | 6 个模型各自的 chat template 适配器（见 NVT 参考里的 Qwen-VL 模板） |
| `cfa_run_audit.py` | 主推理脚本：`--model X --operator Y --pairs ZZZ.jsonl` |
| `cfa_metrics.py` | 算 Acc / Δ-Flip / SCI / per-pathology breakdown |
| `run_audit_<model>_<operator>.sh` | bash wrapper，每个 model × operator 组合一个 |

写法可以参考 NVT 的 `vstar_visual_necessity_audit.py:main()`（在 handoff 之外的 archive 里，Qizhen 会单独发你完整版如果你需要）。

## 5. torchxrayvision sanity classifier 用法（D2 会用到）

```python
import torch
import torchxrayvision as xrv

# 加载预训练 CheXpert classifier
model = xrv.models.DenseNet(weights="densenet121-res224-chex")
model.eval()

# 推理（输入是 1x1xHxW 的 torch tensor，preprocess 见 torchxrayvision docs）
# 输出每个 pathology 的 confidence
```

完整 demo 见 `torchxrayvision` GitHub README。**用这个做 mask sanity 比你自己训一个 classifier 快 10 倍**。
