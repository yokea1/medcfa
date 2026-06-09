#!/usr/bin/env bash
# 下载 CheXlocalize 测试集 (668 张图 + bbox 标注)
#
# CheXlocalize 由 Stanford ML Group 维护。
# Test split 的标签 + bbox 是公开的，不需要 CheXpert PhysioNet 认证。
#
# 论文引用：
#   Saporta et al. "Benchmarking saliency methods for chest X-ray interpretation"
#   Nature Machine Intelligence 2022. arXiv: 2207.04106
#
# 使用：
#   source env.sh
#   bash download_chexlocalize.sh

set -euo pipefail

if [[ -z "${MEDCFA_DATA:-}" ]] || [[ "${MEDCFA_STORAGE:-}" == "<YOUR_ROOT_PATH>" ]]; then
  echo "ERROR: 先 source env.sh，并且确认 MEDCFA_STORAGE 不是占位符" >&2
  exit 1
fi

CHEX_DIR=$MEDCFA_DATA/chexlocalize
mkdir -p "$CHEX_DIR"
cd "$CHEX_DIR"

echo "[chexloc] 目标目录: $CHEX_DIR"

# ============================================================
# Fallback 1: HuggingFace mirror（首选，最快，无需手动登录）
# ============================================================
if [[ ! -d "$CHEX_DIR/raw" ]]; then
  echo "[chexloc] 尝试 HuggingFace mirror..."
  "$MEDCFA_PY" -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='StanfordAIMI/chexlocalize',
    repo_type='dataset',
    local_dir=os.path.join(os.environ['MEDCFA_DATA'], 'chexlocalize', 'raw'),
    local_dir_use_symlinks=False,
)
" 2>&1 && DOWNLOAD_OK=1 || DOWNLOAD_OK=0
fi

# ============================================================
# Fallback 2: 手动下载提示
# ============================================================
if [[ "${DOWNLOAD_OK:-0}" -eq 0 ]] && [[ ! -d "$CHEX_DIR/raw" ]]; then
  cat <<EOF >&2

[chexloc] ⚠️  HuggingFace mirror 失败。

请按以下 fallback 顺序处理：

  (a) 手动下载（建议）
      访问：https://stanfordaimi.azurewebsites.net/datasets
      搜索 "CheXlocalize"
      注册账号 → 接受 license → 下载 test split zip
      解压到：$CHEX_DIR/raw/

  (b) 改用 MS-CXR（需要 PhysioNet 认证）
      访问：https://physionet.org/content/ms-cxr/
      下载 image-sentence + bbox 标注
      解压到：$MEDCFA_DATA/ms_cxr/raw/
      然后告诉 Qizhen，需要改 cfa_build_pairs.py 适配 MS-CXR 格式

  (c) 改用 VinDr-CXR（开放，但 schema 不同）
      访问：https://vindr.ai/datasets/cxr
      Kaggle 也有 mirror

  (d) 卡住超过 30 分钟，停手 ping Qizhen

EOF
  exit 2
fi

# ============================================================
# Sanity check
# ============================================================
echo "[chexloc] sanity check..."
ACTUAL_IMG_COUNT=$(find "$CHEX_DIR/raw" \( -name "*.jpg" -o -name "*.png" -o -name "*.dcm" \) 2>/dev/null | wc -l)
echo "[chexloc] 找到 $ACTUAL_IMG_COUNT 张图（预期 ~668）"

if [[ "$ACTUAL_IMG_COUNT" -lt 100 ]]; then
  echo "[chexloc] ⚠️ 图像数量明显偏少，请检查下载是否完整" >&2
  exit 3
fi

echo "[chexloc] DONE. 目录布局："
ls -lh "$CHEX_DIR/raw/" | head -20
