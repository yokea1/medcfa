#!/usr/bin/env bash
set -u

export MEDCFA_STORAGE="/home/217885@student.upm.edu.my/medcfa"
export MEDCFA_PY="/home/217885@student.upm.edu.my/apps/miniforge3/envs/medcfa/bin/python"

export HF_HOME=$MEDCFA_STORAGE/cache/huggingface
export HF_DATASETS_CACHE=$MEDCFA_STORAGE/cache/datasets
export TRANSFORMERS_CACHE=$MEDCFA_STORAGE/cache/huggingface/transformers
export MODELSCOPE_CACHE=$MEDCFA_STORAGE/cache/modelscope
export TORCH_HOME=$MEDCFA_STORAGE/cache/torch
export XDG_CACHE_HOME=$MEDCFA_STORAGE/cache/xdg
export VLLM_CACHE_ROOT=$MEDCFA_STORAGE/cache/vllm

export MEDCFA_DATA=$MEDCFA_STORAGE/data
export MEDCFA_RESULTS=$MEDCFA_STORAGE/results
export MEDCFA_LOGS=$MEDCFA_STORAGE/logs

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$MODELSCOPE_CACHE" \
  "$TORCH_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" \
  "$MEDCFA_DATA" "$MEDCFA_RESULTS" "$MEDCFA_LOGS" 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

echo "[medcfa env] MEDCFA_STORAGE=$MEDCFA_STORAGE"
echo "[medcfa env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[medcfa env] python=$MEDCFA_PY"
