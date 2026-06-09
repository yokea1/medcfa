# 环境配置 —— 5 分钟上手

这一步你要做的事：

1. 选一个**你能写的远端目录**，作为本论文所有 cache / 数据 / 结果的根
2. 把这个路径填进 `env_template.sh`，重命名为 `env.sh`
3. 安装 Python 依赖
4. 下载数据

完成后你就可以照 `../DAY_1_zh.md` 开始 Day 1。

---

## 1. 选远端根目录（建议 ≥ 100 GB 可写空间）

任何你能 `mkdir` 的位置都行。例：

```bash
/home/yuke/projects/medcfa/                    # 个人家目录
/scratch/yuke/medcfa/                          # scratch 盘
/data/yuke_he/medcfa/                          # 实验室共享盘
```

记住这个路径。下面叫它 `<YOUR_ROOT>`。

## 2. 配 env_template.sh

复制模板：

```bash
cp env_template.sh env.sh
```

打开 `env.sh`，把两个 `<...>` 占位符替换成你的实际路径：

- `MEDCFA_STORAGE=<YOUR_ROOT>` → 填你第 1 步选的路径
- `MEDCFA_PY=<YOUR_PYTHON>` → 填你的 Python 解释器路径（通常是 conda env 里的 `bin/python`）

确认无误后：

```bash
source env.sh
```

应该看到 banner：

```
[medcfa env] MEDCFA_STORAGE=/your/path/here
[medcfa env] CUDA_VISIBLE_DEVICES=0,1
[medcfa env] python=/your/python/path
```

**之后每次开新 shell 都需要重新 `source env.sh`**。建议加到你的 `.bashrc`：

```bash
[ -f /path/to/medcfa/setup/env.sh ] && source /path/to/medcfa/setup/env.sh
```

## 3. 安装 Python 依赖

建议**新建一个 conda 环境**，避免污染你已有的项目：

```bash
conda create -n medcfa python=3.10 -y
conda activate medcfa
which python                       # 把这个路径填回 env.sh 的 MEDCFA_PY
pip install -r requirements.txt    # ~ 5 分钟
```

完整性 check：

```bash
python -c "
import torch, transformers, PIL
from huggingface_hub import snapshot_download
print('torch:', torch.__version__, '  cuda:', torch.cuda.is_available(), '  gpus:', torch.cuda.device_count())
print('transformers:', transformers.__version__)
print('ok')
"
```

应该看到 cuda True + GPU 数 ≥ 2。

## 4. 写入测试

```bash
mkdir -p $MEDCFA_STORAGE/data $MEDCFA_STORAGE/results $MEDCFA_STORAGE/logs
touch $MEDCFA_STORAGE/.write_test && rm $MEDCFA_STORAGE/.write_test
echo OK
```

写不进去就换个目录或者找 IT 调权限，**不要继续往下**。

## 5. 下载数据

```bash
bash download_chexlocalize.sh
```

如果 HuggingFace mirror 直连失败，脚本会打印 3 级 fallback 提示（手动下载、改用 MS-CXR、ping Qizhen）。

## 6. HuggingFace token（可选，但建议）

部分模型（如 MedGemma、HuatuoGPT-Vision）需要 HF gating 申请通过：

1. <https://huggingface.co/settings/tokens> 申请一个 **read** token
2. 在 `env.sh` 末尾加一行：
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```
3. 申请 model 访问：
   - `google/medgemma-4b-it`
   - `FreedomIntelligence/HuatuoGPT-Vision-7B`
   - `microsoft/llava-med-v1.5-mistral-7b`（部分版本需要 gating）

申请后通常几小时内通过。**Day 1 之前把 token 配好**，否则 Day 5 平行跑模型会卡住。

## 7. 完成确认

跑完上面所有步骤后，发 Qizhen 一句确认：

```
medcfa 环境已配好：
  storage: <YOUR_ROOT>
  python:  <YOUR_PYTHON>
  gpu:     2x A100 80GB
  data:    chexlocalize 已下载 (N=XXX images)
  hf_token: 已配 / 待申请
```

然后翻到 `../DAY_1_zh.md`，从 Hour 0 开始。
