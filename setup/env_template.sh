#!/usr/bin/env bash
# MedCFA env vars —— 模板文件
#
# 使用方法:
#   1. cp env_template.sh env.sh
#   2. 编辑 env.sh，把下面两个 <...> 填成你的实际路径
#   3. source env.sh
#
# 每次开新 shell 都要重新 source。建议加到 ~/.bashrc 自动加载。
#
# Hard rule: 所有 model cache / 数据 / 中间结果必须放在 MEDCFA_STORAGE 下，
# 不要污染你的 ~/.cache/* 或者其他共享 cache。

set -u

# ============================================================
# ⚠️ 必填：你的实际路径
# ============================================================

# 1. 远端根目录（至少 100GB 可写空间）
export MEDCFA_STORAGE=<YOUR_ROOT_PATH>
# 例：/home/yuke/projects/medcfa
# 例：/scratch/yuke/medcfa
# 例：/data/yuke_he/medcfa

# 2. Python 解释器路径（建议用专门的 conda env）
export MEDCFA_PY=<YOUR_PYTHON_PATH>
# 例：/home/yuke/miniconda3/envs/medcfa/bin/python

# ============================================================
# 下面不用改（除非你有特殊需求）
# ============================================================

# HuggingFace / torch / vLLM caches —— 全部指向 MEDCFA_STORAGE 下
export HF_HOME=$MEDCFA_STORAGE/cache/huggingface
export HF_DATASETS_CACHE=$MEDCFA_STORAGE/cache/datasets
export TRANSFORMERS_CACHE=$MEDCFA_STORAGE/cache/huggingface/transformers
export MODELSCOPE_CACHE=$MEDCFA_STORAGE/cache/modelscope
export TORCH_HOME=$MEDCFA_STORAGE/cache/torch
export XDG_CACHE_HOME=$MEDCFA_STORAGE/cache/xdg
export VLLM_CACHE_ROOT=$MEDCFA_STORAGE/cache/vllm

# 实验目录
export MEDCFA_DATA=$MEDCFA_STORAGE/data
export MEDCFA_RESULTS=$MEDCFA_STORAGE/results
export MEDCFA_LOGS=$MEDCFA_STORAGE/logs

# 第一次 source 时自动建目录
mkdir -p \
  "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$MODELSCOPE_CACHE" \
  "$TORCH_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" \
  "$MEDCFA_DATA" "$MEDCFA_RESULTS" "$MEDCFA_LOGS" 2>/dev/null || true

# GPU 默认两张都用
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# 加速 HF 下载
export HF_HUB_ENABLE_HF_TRANSFER=1

# 关闭 transformers 烦人的 warning
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

# ============================================================
# 可选：HF token（部分模型需要）
# ============================================================
# 申请：https://huggingface.co/settings/tokens（read 权限即可）
# 取消下一行注释，填上你的 token
# export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# ============================================================
# Banner
# ============================================================
echo "[medcfa env] MEDCFA_STORAGE=$MEDCFA_STORAGE"
echo "[medcfa env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[medcfa env] python=$MEDCFA_PY"

# Sanity check
if [[ "$MEDCFA_STORAGE" == "<YOUR_ROOT_PATH>" ]]; then
  echo "[medcfa env] ⚠️  WARN: MEDCFA_STORAGE 还是占位符，请先编辑 env.sh"
fi
if [[ "$MEDCFA_PY" == "<YOUR_PYTHON_PATH>" ]]; then
  echo "[medcfa env] ⚠️  WARN: MEDCFA_PY 还是占位符，请先编辑 env.sh"
fi
