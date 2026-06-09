# Day 1 —— 第一天逐小时计划（Yuke）

总计：约 **6 小时聚焦工作**。任何阻塞超过 30 分钟立刻 ping Qizhen，不要 spin。

> 这一天的目标**不是产生 paper**，而是把 pipeline 跑通，确认数据 / 代码 / 模型三件事**都可用**。Day 2 才开始真实工程。

---

## Hour 0–1：阅读 + 定向

1. 读 `README_zh.md`（整体说明）
2. 读 `HANDOFF_zh.md` §0、§0.5、§0.7、§9、§14、§20 —— 其他章节按需翻
3. 浏览 `code/README_zh.md` 看看包内已有什么代码
4. 浏览 `literature/related_work_zh.md`，标记 3 篇 ★ 论文明天读

读完后能用一句话回答下面 3 个问题：
- 这篇 paper 想证明 **什么**？
- 为什么不和 NVT (archived) 或 GEER (active) 重复？
- 实验数据**从哪里**来？

任何一题答不上，立刻停下来发消息给 Qizhen。

---

## Hour 1–2：环境 + 存储

1. 按 `setup/README_zh.md` 配好你自己的 `env.sh`（填 `MEDCFA_STORAGE` + `MEDCFA_PY`）。
2. `source setup/env.sh`，确认 banner 显示的路径正确，**不再出现 `<YOUR_*>` 占位符**。
3. 确认存储挂载存在且能写：
   ```bash
   mkdir -p $MEDCFA_STORAGE/data $MEDCFA_STORAGE/results $MEDCFA_STORAGE/logs
   touch $MEDCFA_STORAGE/.write_test && rm $MEDCFA_STORAGE/.write_test
   echo OK
   ```
4. 检查 GPU：`nvidia-smi` —— 期望看到 2× A100 80GB 全部空闲。
5. 检查 Python：
   ```bash
   $MEDCFA_PY --version                                                                  # 期望 Python 3.10+
   $MEDCFA_PY -c "import torch, transformers, PIL; print(torch.cuda.device_count())"     # 期望 2
   ```

写不进去或者环境跑不起来 → 停手，找运维 / Qizhen。

---

## Hour 2–3：数据下载

1. ```bash
   bash setup/download_chexlocalize.sh
   ```
   预期：CheXlocalize test split 下载到 `$MEDCFA_STORAGE/data/chexlocalize/raw/`。

2. 如果 HF mirror 失败，按 `download_chexlocalize.sh` 里打印的 fallback 链处理：
   - (a) Stanford 官网手动下载 + 解压
   - (b) 改用 MS-CXR（需 PhysioNet 认证）
   - (c) 改用 VinDr-CXR
   - (d) 卡 30 分钟以上 → ping Qizhen

3. 看看下到了什么：
   ```bash
   find $MEDCFA_STORAGE/data/chexlocalize/raw -type f | head -20
   ls -lh $MEDCFA_STORAGE/data/chexlocalize/raw/
   ```
   预期：~668 张 `.jpg`/`.png`/`.dcm`，1 个或多个 bbox 标注 JSON，train/test split 列表。

4. 打开一个 bbox JSON 在 Python repl 里看 schema，把发现写到 `$MEDCFA_STORAGE/data_inventory.md`（你自己建）：包含哪些字段、一个样例 entry 是什么样、image 文件命名规则。

---

## Hour 3–4：mask code 导入 + smoke

1. 建工作目录：
   ```bash
   mkdir -p $MEDCFA_STORAGE/work/{scripts,experiments,paper,logs}
   ```

2. 复制包内 mask code 到工作目录：
   ```bash
   cp code/cfa_mask_operators.py code/_nvt_apply_operator.py $MEDCFA_STORAGE/work/scripts/
   ```

3. Smoke test（包里已经写好，直接跑）：
   ```bash
   cd $MEDCFA_STORAGE/work/scripts
   $MEDCFA_PY cfa_mask_operators.py $MEDCFA_STORAGE/data/chexlocalize/raw/imgs/<任意一张>.jpg
   ```
   预期：在 `/tmp/` 下生成 `smoke_{zero,blur,matched_patch}.jpg` 三张图 + 打印每张的 sanity check 结果。

4. 用图片查看器打开 `/tmp/smoke_*.jpg` 三张：
   - `zero`：bbox 区域是中性灰矩形
   - `blur`：bbox 区域明显模糊
   - `matched_patch`：bbox 区域被另一块替换（smoke 里 distractor 是原图自己，所以看上去差不多——真实用法要换健康胸片）

5. 把 smoke 结果发给 Qizhen 看一眼（或者贴在 EOD 总结里）。

---

## Hour 4–5：VQA 配对生成骨架

1. 在 `$MEDCFA_STORAGE/work/scripts/` 下写 `cfa_build_pairs.py`，按 `HANDOFF_zh.md` §11 的 schema 把 CheXlocalize bbox JSON 转 VQA pairs JSONL。

2. 限量跑 50 张做 pilot：
   ```bash
   $MEDCFA_PY $MEDCFA_STORAGE/work/scripts/cfa_build_pairs.py \
     --bbox-json $MEDCFA_STORAGE/data/chexlocalize/raw/<bbox_file>.json \
     --image-root $MEDCFA_STORAGE/data/chexlocalize/raw/imgs/ \
     --output $MEDCFA_STORAGE/data/medcfa_pairs_pilot50.jsonl \
     --limit 50
   ```

3. 打开 JSONL，确认每张图有约 10 个 pair（每个 pathology 一个 yes/no question），bbox 数值在合理像素范围内。

---

## Hour 5–6：单模型加载 sanity

1. 选 **Qwen2.5-VL-7B-Instruct**（无 gating 风险最低）。第一次下载约 14GB，正常网络下 ~10 分钟。

2. Smoke load：
   ```bash
   $MEDCFA_PY -c "
   from transformers import AutoModelForVision2Seq, AutoProcessor
   import torch
   m = AutoModelForVision2Seq.from_pretrained(
       'Qwen/Qwen2.5-VL-7B-Instruct',
       torch_dtype=torch.bfloat16,
       device_map='auto',
       trust_remote_code=True,
   )
   p = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', trust_remote_code=True)
   print('model loaded, params=', sum(t.numel() for t in m.parameters())/1e9, 'B')
   "
   ```
   预期：~7B 参数；下载完成且 GPU 占用合理。

3. 用 1 张图跑 1 个 yes/no 问题（写一个最小 `_smoke_one_inference.py`，参考官方 Qwen-VL inference 示例 + `code/cfa_mask_operators.py` 里 `apply_bbox_mask` 用法）：
   ```bash
   $MEDCFA_PY $MEDCFA_STORAGE/work/scripts/_smoke_one_inference.py \
     --image $MEDCFA_STORAGE/data/chexlocalize/raw/imgs/<某张positive图>.jpg \
     --question "Is there any sign of Cardiomegaly in this image?"
   ```
   预期：输出一个 yes 或 no（或带短解释的 yes/no）。

---

## Day 1 结束 checkpoint

到 hour 6 结束时，下面这些勾应该都打上：

- [ ] env 已 source；GPU 可见；写入 `$MEDCFA_STORAGE` OK
- [ ] CheXlocalize 已下载，≥ 600 张图可定位
- [ ] `cfa_mask_operators.py` smoke 跑通；`/tmp/smoke_*.jpg` 看上去合理
- [ ] `medcfa_pairs_pilot50.jsonl` 生成，≥ 200 pair
- [ ] Qwen2.5-VL-7B 加载成功，答了 1 个问题

### 发给 Qizhen 的 EOD 5 行：

```
medcfa day 1 完成
[x] env / storage / GPU
[x] chexloc 下载 (N=XXX images)
[x] mask operator smoke (sanity inside/outside ssim 已 print)
[x] pair gen pilot (N=XXX pairs)
[x] qwen2.5-vl-7b 加载 + 1 个答案
blockers: <无 | 说明>
明天: D2 全量 sanity + cfa_build_pairs.py 全量跑
```

---

## Day 1 常见坑（这些都是真的遇到过的）

| 症状 | 原因 | 修复 |
|---|---|---|
| `OSError: ... HF_HUB_OFFLINE` | env 没 source | 重新 `source setup/env.sh` |
| HF 下载速度 < 1MB/s | 代理或网络 | `unset HTTPS_PROXY HTTP_PROXY` 或确认 `HF_HUB_ENABLE_HF_TRANSFER=1` |
| `CUDA out of memory` on Qwen2.5-VL-7B | `device_map='auto'` 选了一张卡 | 强制 `device_map={"": 0}` 或用 accelerate 平均分 |
| CheXlocalize HF repo 404 | mirror 不在了 | 走 fallback 链；30 分钟卡住就 ping Qizhen |
| `apply_operator` ValueError: matched_patch requires distractor | 忘记传 `distractor_image` | matched_patch 永远要传一张 fallback distractor |
| PIL `mode='F'` 或 `mode='1'` errors | CXR 有时是 16-bit grayscale | 所有 `Image.open(...)` 后立刻 `.convert('RGB')` |
| LLaVA-Med 加载报错 | 需要 `LlavaForConditionalGeneration` 而不是 `AutoModelForVision2Seq` | 查 LLaVA-Med GitHub README 的加载片段 |
| Qwen2.5-VL 报错 `qwen-vl-utils` 缺失 | 漏装 | `pip install qwen-vl-utils` |
| torchxrayvision 在 D2 sanity 阶段报错 | 模型 weight 第一次自动下载 | 先 import 一次让它下载完成 |

---

**提醒**：Day 1 是 21 天里的第 1 天。过度工程化的诱惑是真实的。**别**。Day 1 = smoke + 探路；真正的工程从 Day 2-3 开始。
