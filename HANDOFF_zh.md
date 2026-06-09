# Handoff: MedCFA — Medical Counterfactual Audit (Chest X-Ray)

| | |
|---|---|
| **PI** | Qizhen Lan (`lanqz7766@gmail.com`) |
| **Student (executor)** | Yuke He |
| **Sync cadence** | **每周至少一次** 例会；unblock 消息任何时候发，PI 优先响应 D7 / D14 决策门和 stop-condition 触发 |
| **硬件** | 2× A100 80GB（你自有节点；Qizhen 不直接访问你的 server） |
| **时间预算** | **3 周** — experiments + writing 全部完成、可投稿状态 |
| **目标会场** | **WACV 2027 Round 1**（deadline 约 2026-06 中下旬，Qizhen 会确认具体日期） |
| **包内位置** | 本目录 `handoff_pkg/` 自包含；不依赖任何远端 server |
| **日期** | 2026-05-28 |

## 配套文件清单（必读顺序）

1. **`README_zh.md`** — 包整体说明
2. **`setup/README_zh.md`** — 5 分钟配你自己的环境
3. **`DAY_1_zh.md`** — Day 1 hour-by-hour，含 traps 和 fallback。**第一天直接照着做**。
4. **`setup/env_template.sh`** — env vars 模板，你填自己路径
5. **`setup/download_chexlocalize.sh`** — 数据下载脚本（含 3 级 fallback 链）
6. **`code/README_zh.md`** — 参考代码用法（mask operator 已 ready-to-use）
7. **`literature/related_work_zh.md`** — 预先做好的文献地图（arxiv ID + takeaway，节省 1-2 天文献调研）
8. **`paper_template/README_zh.md`** — WACV 模板用法
9. 本文件 `HANDOFF_zh.md` — 完整 spec
10. 后续每天写 `experiments/EXPERIMENT_TRACKER.md`（D2 起建）—— 见下面 §15

---

## 0. 给你的话（Yuke，必读，30 秒）

你来执行一个 **benchmark + audit** 类 paper，目标 **WACV 2027 Round 1**。Qizhen 已经把方案降级到最快档（mask-based counterfactual，不用 diffusion；纯推理，不训练；单 modality；3 周可投稿）。

**重点不是创新一个 method，而是干净地证明一件事**：

> **现有 medical VLM 在 chest X-ray 上的 accuracy 高低，与"它真的看了病灶"几乎无关。**

如果这个结论站得住（pilot 第 3 天就能看到信号），论文就是 PASS。如果不站，立刻按"Stop conditions"切换备份故事（§13），不要硬推。

不要：自己加 method、加 modality、加新指标、改 mask 形式。这些都是博士论文级 scope，3 周做不完。

## 0.5 Position & Importance（为什么这个 idea 不是随便想的）

### Position（这篇 paper 在 portfolio 和 community 中的位置）

**在 Qizhen 的 WACV portfolio 中**（参考 `../STORAGE_MAP.md` + `../../CLAUDE.md`）：

```
GEER (active)         —— inference-time decision policy: 用 grounding sufficiency 决定何时停止推理
EvidenceKD (active)   —— training-time KD signal: 蒸馏时保留 evidence dependence
MedCFA (this paper)   —— evaluation-time audit protocol: 揭示现有 medical VLM 的 evidence shortcut
NVT (archived)        —— inference-time learned gate: 学一个 cell-level necessity selector（被 GEER 吸收）
```

四个 paper 共用一个 conceptual axis "**evidence-causal-dependence of multimodal models**"，但分别打在 **inference decision / training signal / evaluation / inference gate** 四个不同的 intervention point。MedCFA 是 portfolio 里**唯一一篇 evaluation-only**——这是它的护城河，**不和 GEER / EvidenceKD 抢工作**。

**在 medical VLM community 中**：

| 现有评测维度 | benchmark/工作 | 它们没测什么 |
|---|---|---|
| accuracy on closed QA | OmniMedVQA, SLAKE, VQA-RAD, PathVQA | 模型是否真的看了图 |
| hallucination on open generation | POPE, HallusionBench | 因果替换下答案是否变 |
| localization quality | HEAL-MedVQA, S-Chain | bbox 是否对答案因果必要 |
| robustness to shift | MedFair, MedShift | 视觉证据本身被破坏后的行为 |

MedCFA 占的是这张表的空格：**"causal dependence of answer on visual evidence"**。

### Importance（为什么 reviewer 会觉得重要）

1. **临床部署性**：medical VLM 已经在试用临床。如果一个模型 accuracy 80% 但其中 30% 来自非视觉 shortcut（如 report-style prior），部署时遇到 distribution shift 会 silent fail。Δ-Flip 是 deployment-time 该测的指标，但今天没人测。
2. **可复现的 audit protocol**：mask-based 不需要 diffusion / 不需要 radiologist，**任何 lab 拿一个 bbox-annotated dataset 都能跑**，可推广到 pathology / MRI / CT。
3. **暴露 medical-finetuned VLM 的失败模式**：如果 LLaVA-Med / HuatuoGPT-Vision / MedGemma 的 SCI 比 general Qwen2.5-VL 还高（finding A），意味着 **medical 数据 finetune 反而强化了 shortcut**——这是会被 cite 的 finding。
4. **方法学示范**：mask-based counterfactual 是因果 ML / explainable AI 大家庭里 ROAR 类思想的 inference-only 简化。第一次干净落在 medical VLM 上，会拿到 method-school 引用。

### Why now（时机判断）

- Medical VLM 2024-2026 喷井式涌出（LLaVA-Med, HuatuoGPT-Vision, MedGemma, RadFM, BiomedGPT...）；大家都报 accuracy，**几乎没人报 visual causal dependence**。这个空窗 6-12 个月后会被填上，**这窗口里抢出第一个 audit protocol 价值很高**。
- WACV 接受 benchmark / audit paper（不强制 method）。`benchmark-paper-template` skill 已经装好。
- 你（Yuke）这次执行的硬件 + 时间约束（2 A100 + 3 周）正好够 mask-based v1 + 第二 benchmark 横向验证，再大 scope 都会爆。

## 0.7 前置验证（这个 idea 不是空想——内部已经有强 signal）

下面是 Qizhen 之前的 NVT 论文（已 archive，未发表）在 V*Bench 上跑出来的**已被内部验证**的关键 finding，直接搬到 MedCFA 也成立的概率 > 70%。这些数字是 Qizhen 团队内部的实验结果，不是公开 paper 引用。

### 验证 1：mask-based counterfactual 在 V*Bench 上**显著**区分必要 vs 非必要视觉区域

来源：NVT R015 (E0 visual necessity audit)，Tracker §G7。

| Setting | Selected region 替换后 error | Random region 替换后 error | Gap |
|---|---|---|---|
| zero operator | **47.44%** | 6.41% | **+41.03** |
| blur operator | **50.64%** | 9.62% | **+41.02** |
| matched_patch | **48.72%** | 8.97% | **+39.75** |

n=156 (V*Bench full subset)。**+40 个百分点的因果替换 gap**——这是干净的因果 signal，证明三个 operator 在 general VQA 上都能识别必要视觉证据。

> **MedCFA 直接预期**：在 chest X-ray 上，CheXlocalize 病灶 bbox 区域对一个**真正看病灶**的模型应该有同等量级的 gap。如果 D5 dry-run 看不到 +20 以上 gap，要么 model 不真依赖 bbox（finding A 成立 — paper PASS），要么 mask 没擦掉病灶（sanity 失败 — fix mask params）。两种结果都有 actionable response。

### 验证 2：medical 域上 structured / long CoT 会**加重**而非减少 unsupported claims

来源：NVT 早期 R001-R006 在 OmniMedVQA hard80 + Qwen3-VL-8B：

| Rationale type | Unsupported clinical claims / sample |
|---|---|
| Med-SCoT-style structured CoT | **2.96** |
| Long CoT | 1.48 |
| Minimal-posthoc rationale | 1.11 |

> **MedCFA 用法**：这条 finding 在 §1 Introduction 第 2 段当 motivation 引用——"现有 medical VLM 的 verbosity 不等于 faithfulness"。配 introduction 的 hook。

### 验证 3：mask operator 不同形式不会全跑出一样的结论（有 ablation 空间）

NVT R015 / R017 / R018 / R019 系统对比过 zero / blur / matched_patch：

- `zero` 最 leaky（容易被模型识别为"被擦了"，artifact 当 shortcut）
- `blur` 中间
- `matched_patch` 最严格（视觉上仍是真实图像）

> **MedCFA 直接抄**：三个 operator 跑，主结果用 matched_patch，ablation 报另外两个的 Δ-Flip 差异——天然 1 个 figure。

### 验证 4：Qwen-VL 系列 + medical prompt 已经跑过

NVT R001-R002 已经验过 Qwen3-VL-8B 在 OmniMedVQA hard80 上**能跑通** structured / minimal / direct prompt 模板，verifier 也通过了。

> **MedCFA 直接收益**：Qwen2.5-VL-7B / Qwen3-VL-8B 这两个 model 的 loading + chat template + yes/no parsing 在 NVT `scripts/vstar_qwen_nvt_variants.py` 里现成，**Yuke 不用从零写 inference 适配**。

### 验证 5：自动 sanity 是可行的（不强求 radiologist）

NVT R015 用 SSIM + 对比 model accuracy drop 当自动 sanity（不用 radiologist），跑通了 156 样本。

> **MedCFA 用法**：CheXpert classifier (torchxrayvision 包) 当自动 verifier，确认 mask 后病灶 confidence 真的降下来。不需要 radiologist 就能交 v1。

---

**总结**：这个 idea 不是"PI 拍脑袋让你跑"。**5 个核心技术假设里 4 个已在 V*Bench 上前置验证过**，只剩"medical 域是否复现"这一个真正的 unknown。这是 **low-risk benchmark paper 该有的样子**——大部分管线已经有了 confidence，剩下 3 周的工作主要是把 V*Bench 的 audit 协议**忠实地** port 到 chest X-ray，并加一个第二 benchmark (MS-CXR) 横向验证证明 generalization。

---

## 1. 论文工作题目

**首选**：

```
Do Medical VLMs Look at the Lesion?
A Mask-Based Counterfactual Audit of Chest X-Ray Vision-Language Models
```

**备选**（如果 reviewer 觉得首选太 catchy）：

```
Evidence Sensitivity Is Not Accuracy:
A Counterfactual Benchmark for Chest X-Ray Vision-Language Models
```

---

## 2. 一句话 pitch

我们用 bbox-mask 反事实图像 audit 6 个 chest X-ray VLM，发现 accuracy 高的模型**不一定**在病灶被擦掉时改答案——意味着它的答案大量来自非病灶 shortcut（report style、人口学先验、解剖背景）。这是医学 VLM 部署的 silent failure mode。

---

## 3. Gap（一定要在 paper 里写清楚）

| 现有工作 | 他们做了什么 | 我们的差别 |
|---|---|---|
| **POPE / HallusionBench** | object-presence hallucination | 不针对 medical；不做 counterfactual replacement |
| **NaturalBench** | counterfactual pairs，general domain | 不是 medical |
| **HEAL-MedVQA / S-Chain** | medical bbox + grounding，**用作训练监督** | 我们用 bbox 当 **counterfactual operator** 不是训练信号 |
| **CheXlocalize / CheXpert** | classification + localization 标注 | 我们把它当 audit benchmark 用，不是直接训练 |
| **MedVCTP / Med-SCoT** | structured medical CoT，**inference 时改 prompt** | 我们不动 prompt，**动图像**测因果依赖 |
| **MedAdapt / MedFair benchmarks** | 测 distribution shift / fairness | 不测 evidence-causal-dependence |

**真正没人做的事**（这是 paper 的 punch line）：

> 把 **同一个 bbox** 同时作为(a) accuracy 评估的 ground truth、(b) counterfactual replacement 的 mask 区域，用因果替换检验现有 medical VLM 的**视觉证据真依赖**，并发现 accuracy 排序 ≠ evidence-sensitivity 排序。

---

## 4. 思路 / 核心逻辑（写 paper 时按这个顺序展开）

```
Step 1.  Medical VLM benchmark 普遍只测 accuracy
Step 2.  Accuracy 高可能来自非视觉证据 shortcut（report-style prior，
         demographic prior，anatomical background，频率 prior）
Step 3.  要测"模型真的看了吗"，需要因果干预：擦掉病灶看答案变不变
Step 4.  CheXlocalize 已经标了 bbox → 直接当 mask 区域
Step 5.  用三种 mask operator（zero / blur / matched-patch）替换 bbox 内像素，
         看每个模型答案翻转率（Δ-Flip）
Step 6.  Δ-Flip 高 = 真的依赖病灶；Δ-Flip 低 + accuracy 高 = shortcut
Step 7.  我们 audit 6 个模型，画 Accuracy × Δ-Flip 散点，
         show 两个 axis 几乎不相关（or specific patterns）
Step 8.  Per-pathology / per-model 拆解，找出 shortcut-prone pathologies
Step 9.  结论：medical VLM 部署前必须加 evidence-sensitivity audit；
         我们的 benchmark + protocol 是 plug-and-play
```

---

## 5. 数据

### 5.1 主数据：CheXlocalize（必须用，不要替换）

- **官方**：<https://stanfordmlgroup.github.io/competitions/chexlocalize/>
- **HF mirror**（如果 Stanford 直连不通）：搜 `chexlocalize` on HF
- **规模**：668 张 test images，10 类病灶（Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Lung Lesion, Lung Opacity, Pleural Effusion, Pneumonia, Pneumothorax），**每张图上每个 positive 病灶都有 bbox**
- **License**：研究用允许；要在 paper 写明引用 Saporta et al. 2022

### 5.2 备用 / 扩展（v1 不做，留 ablation 用）

- **MS-CXR**（MIMIC-CXR + radiologist bbox）：如果 CheXlocalize 不够 robust，扩到这里
- **VinDr-CXR**：越南 dataset，22 类 bbox

### 5.3 VQA 配对怎么造（重要）

CheXlocalize 是 classification + bbox，不是 VQA。你需要把它转 VQA：

**对每张图 × 每个 pathology**：
- 如果该 pathology 在该图上 positive 且有 bbox → 生成 yes-question
- 如果 negative → 生成 no-question（同样 prompt 模板）
- 如果 positive 但无 bbox → 跳过（不能 mask）

**Prompt 模板**（不要改，模板差异会污染跨 model 比较）：

```
You are a radiologist reviewing a chest X-ray. Answer with one word: yes or no.
Question: Is there any sign of {pathology} in this image?
```

预期规模：668 imgs × 10 pathologies ≈ 6680 questions，去掉无 bbox 的 positive 后**约 4000-5000** valid (image, pathology) pairs。每个 pair 跑 4 个 condition（original + zero + blur + matched-patch），总推理量 16k-20k forward。**6 个模型 × 20k 推理 = 120k forward，2× A100 上单模型 < 2 小时**（7-8B 模型，bs=1 单图）。

---

## 6. Counterfactual operator（直接抄 NVT archive 代码）

不要重新实现。代码已经在：

```
_archived/WACV27-NVT-Necessary-Visual-Evidence/scripts/vstar_visual_necessity_audit.py
```

里面有 `apply_zero_mask`, `apply_blur_mask`, `apply_matched_patch_mask` 三个函数。直接复用，**只改 mask region 来源**（NVT 用的是 grid cell，你用的是 CheXlocalize bbox）。

### 三个 operator 的含义（写 paper 用）

| Operator | 像素替换 | 因果含义 |
|---|---|---|
| `zero` | bbox 内置 0（黑） | 最简单，但引入"黑色 patch"这种 distribution-shift artifact |
| `blur` | bbox 内 Gaussian blur σ=15 | 保留 low-freq 解剖背景，破坏 high-freq 病灶纹理 |
| `matched-patch` | bbox 内填充同图随机健康区域 patch | 最严格 — 看上去仍然是真实 CXR，但病灶不在了 |

**三个都跑**，paper 主结果用 `matched-patch`（最干净因果），其他两个放 ablation。

### Sanity checks（必须做，写在 paper supplementary）

对每个 masked image：
- SSIM(masked, original) 在 bbox 内 < 0.5（mask 确实改了像素）
- SSIM(masked, original) 在 bbox 外 > 0.95（mask 没影响其他地方）
- 跑一个 CheXpert classifier（pretrained，HF 有），原图 positive、masked 图 prediction confidence 显著下降（否则 mask 没擦掉病灶 → 报告失败 + 调参）

---

## 7. 模型清单（audit list）

**必跑 6 个**（如果哪个 HF gating 暂时拿不到，立刻 ping 我，不要自己 silently 换掉）：

| Model | HF path | 大小 | 类型 | 备注 |
|---|---|---|---|---|
| LLaVA-Med v1.5 | `microsoft/llava-med-v1.5-mistral-7b` | 7B | medical | Microsoft 经典 |
| HuatuoGPT-Vision-7B | `FreedomIntelligence/HuatuoGPT-Vision-7B` | 7B | medical | 中文医疗 |
| MedGemma-4B | `google/medgemma-4b-it` | 4B | medical | Google 2025 |
| Qwen2.5-VL-7B-Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | general | 通用 baseline |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | 8B | general | 较新 |
| InternVL3-8B | `OpenGVLab/InternVL3-8B` | 8B | general | 二选一 LLaVA-OneVision-7B 也行 |

**禁选**：
- GPT-4V / Claude / Gemini API — 太贵 + 推理慢 + 不可复现；写 paper 时说"closed-source models excluded for reproducibility"
- 30B+ — 单卡装不下，不必要

### 推理设置（不要改）

- bf16
- batch_size = 1（单图，简化 OOM 处理）
- max_new_tokens = 8（只问 yes/no，长输出浪费时间）
- temperature = 0 / greedy
- vLLM 不用（KV cache 优化对 single-token 答案无意义；HF transformers 直跑）

---

## 8. Metrics（写 paper 用，公式写清楚）

设 $X$ 为原图，$X^{(\text{mask}, p)}$ 为 pathology $p$ 的 bbox 被 mask 后的图，$M_\theta(X, p)$ 为模型 $\theta$ 在问 "Is there $p$?" 时的二元答案。

### 8.1 Accuracy（已有）
$$\text{Acc}(\theta) = \mathbb{E}_{X, p}\, \mathbf{1}[M_\theta(X, p) = y(X, p)]$$
$y$ 是 CheXlocalize ground truth。

### 8.2 Counterfactual Flip Rate（主指标，命名为 **Δ-Flip**）

只在 positive 子集（$y = \text{yes}$ 且有 bbox）上算：

$$\Delta\text{-Flip}(\theta, \text{op}) = \mathbb{E}_{(X,p):\, y(X,p)=\text{yes}, M_\theta(X,p)=\text{yes}}\, \mathbf{1}[M_\theta(X^{(\text{op},p)}, p) = \text{no}]$$

**直观**：模型原本答对的 positive 病例里，擦掉病灶之后改答 "no" 的比例。高 = 真依赖病灶。

### 8.3 Spurious Confidence Index（命名为 **SCI**）

$$\text{SCI}(\theta) = \text{Acc}(\theta) - \Delta\text{-Flip}(\theta, \text{matched-patch})$$

**直观**：accuracy 减去 evidence-sensitivity。高 = 答对很多但靠 shortcut 答对。**这是 paper 的 punch metric**。

### 8.4 Per-pathology breakdown

每个 metric 拆 10 个病灶，看哪些病灶 shortcut-prone（SCI 高）。预测：cardiomegaly 因为心脏轮廓全图可见，SCI 应该最高；lung lesion / nodule 应该最低。

### 8.5 Sanity-only metric

$$\text{MaskValid}(X, p) = \begin{cases} 1 & \text{if classifier confidence drops} \geq 30\% \text{ after mask}\\ 0 & \text{otherwise} \end{cases}$$
只在 $\text{MaskValid}=1$ 的样本上计算 Δ-Flip（去掉 mask 失败案例）。

---

## 9. 实验 matrix（21 天，day-by-day）—— R1 节奏

R1 接收率高，但 reviewer 期望也更高。比 R2 兜底版本多两个动作：
1. **第二 benchmark (MS-CXR) 横向验证** —— 证明 finding 不是 CheXlocalize 单数据集 artifact
2. **更扎实的写作 polish + 多轮 audit** —— claim-audit + citation-audit + kill-argument 各跑一轮

### Week 1：数据 + pipeline + 第一波模型

| Day | 任务 | 通过条件 |
|---|---|---|
| D1 | 自配环境（`setup/README_zh.md`）+ HF cache + CheXlocalize 下载完成 | 数据 `ls` 看到 ≥ 600 imgs + bbox json |
| D2 | 移植 `code/cfa_mask_operators.py` + 写 sanity 脚本；100 张 pilot | 100 张 mask 通过 SSIM 双向阈值 + torchxrayvision classifier confidence drop |
| D3 | VQA 配对生成（4-5k pairs） + prompt 模板锁定 + model adapters 框架 | jsonl 文件 4-5k 行，schema 见 §11 |
| D4 | 跑 1 个模型 (Qwen2.5-VL-7B) 全流程 dry run | 全部 metric 算出来；Acc ≥ 50% |
| D5 | metric 脚本完整化；跑第 2 个模型 (Qwen3-VL-8B) | 两个模型主表 v0 |
| D6 | 跑第 3 个模型 (LLaVA-Med v1.5) | 3 模型主表 v1 |
| D7 | **第一决策门**：和 Qizhen review 3 模型主表，判断是否 Δ-Flip 有区分度 | PI 签字；继续 / pivot |

### Week 2：完整 audit + 第二 benchmark

| Day | 任务 | 通过条件 |
|---|---|---|
| D8 | 跑第 4-5 个模型 (HuatuoGPT-Vision, MedGemma) | 5 模型主表 |
| D9 | 跑第 6 个模型 (InternVL3 或 LLaVA-OneVision) | 6 模型完整主表 |
| D10 | per-pathology breakdown + operator ablation 分析 | Table 2 v1 + Figure 2 v1 |
| D11 | Figure 1 (Accuracy × Δ-Flip 散点) + Figure 3 qualitative cases | 3 个 figure v1 |
| D12 | **MS-CXR 第二 benchmark 数据 + sanity** | MS-CXR pair JSONL + sanity 通过 |
| D13 | 6 模型在 MS-CXR 上跑一遍 (operator 主要用 matched_patch) | MS-CXR 主表 |
| D14 | **第二决策门**：完整跨 benchmark review，定 finding A/B/C | PI 签字；故事锁定 |

### Week 3：写作 + 三层 audit + 提交

| Day | 任务 | 通过条件 |
|---|---|---|
| D15 | LaTeX skeleton（用 `paper_template/main_skeleton.tex`）+ abstract + intro draft | section 0-1 |
| D16 | Section 2 (related work, 用 `literature/related_work_zh.md`) + Section 3 (protocol) | section 2-3 |
| D17 | Section 4 (experiments, 含 CheXlocalize 主表 + MS-CXR generalization) + Section 5 (discussion) | section 4-5 |
| D18 | 跑 `paper-claim-audit` 和 `citation-audit` skill；按结果改 | 0 critical claim issue, 0 hallucinated citation |
| D19 | 跑 `kill-argument` 对抗 review；按 top-3 weakness 改 | rebuttal-ready defense 已加 |
| D20 | 终稿 polish + figure caption 优化 + anonymity 检查 + `\todo` grep 干净 | PDF 几乎成稿 |
| D21 | `paper-compile` 终编译；Qizhen 终审；submit | **投稿完成** |

**两个决策门是命门**：

- **D7 决策门**：3 个模型主表必须看到 Δ-Flip 有显著区分度（某些模型 < 50%，某些 > 70%）。否则切 finding C (operator taxonomy 路线)。
- **D14 决策门**：第二 benchmark (MS-CXR) 上 finding 必须**至少有方向性的复现**（不要求数字一致，但 SCI 排序要保持）。否则 paper 只能 claim "CheXlocalize-specific findings"，downgrade 一个等级。

---

## 10. 工作清单（按时间顺序，可直接当 todo）

每完成一项打勾。

### 数据 + 基础设施
- [ ] **W1**: 远端建 `$MEDCFA_STORAGE/{data, cache, results, logs}`；按 `setup/README_zh.md` 配好 env
- [ ] **W2**: 下载 CheXlocalize 到 `data/chexlocalize/`，dataset card 写一份本地 `data/README.md`
- [ ] **W3**: 直接用 `code/cfa_mask_operators.py`（已经 bbox-adapted ready-to-use）；smoke test 见 `code/README_zh.md`
- [ ] **W4**: 写 `scripts/cfa_build_pairs.py` 生成 jsonl，schema 严格按 §11
- [ ] **W5**: 写 `scripts/cfa_sanity_check.py` — SSIM + CheXpert classifier confidence drop
- [ ] **W6**: 写 `scripts/cfa_run_audit.py` — 单模型推理，支持 `--model qwen2_5_vl_7b`, `--operator matched_patch` 等 CLI flag

### 推理 + 评估
- [ ] **W7**: D4 dry run on Qwen2.5-VL-7B；输出格式锁定后，做成 `run_audit_{model}_{operator}.sh` 模板
- [ ] **W8**: 平行跑 6 模型 × 4 condition（original + 3 operator）= 24 个 run，结果 dump 到 `results/<model>/<operator>/preds.jsonl`
- [ ] **W9**: 写 `scripts/cfa_metrics.py` — 算 Acc / Δ-Flip / SCI / per-pathology / per-operator
- [ ] **W10**: 主表 + figure 1/2/3（用 `paper-figure` skill，不要手 plot）

### 写作 + 审计 + 提交
- [ ] **W11**: 复制 `paper_template/` 下所有文件到你的工作目录 `paper/`；按 `paper_template/README_zh.md` 操作
- [ ] **W12**: 用 `paper-write` skill 生成 section 1-7 草稿，按 §12 outline 顺序
- [ ] **W13**: 用 `paper-claim-audit` 跑零上下文 claim 核对（**必须！** 这个 skill 是 reject 杀手）
- [ ] **W14**: 用 `citation-audit` 跑零上下文引用核对
- [ ] **W15**: 用 `kill-argument` skill 跑对抗 review；列出 top-3 weakness，每条写一段 defense
- [ ] **W16**: 用 `paper-compile` 编译 PDF；human read once；改 typo
- [ ] **W17**: 把 paper folder 整体 ready，pi 终审

---

## 11. 数据 schema + 预期 figures

### VQA pair JSONL schema（不要改 field 名）

```json
{
  "qid": "chexloc_00001_pneumonia",
  "image_id": "00001",
  "image_path": "data/chexlocalize/imgs/00001.jpg",
  "pathology": "Pneumonia",
  "label": "yes",
  "bbox": [123, 45, 234, 178],
  "question": "Is there any sign of Pneumonia in this image?",
  "expected_answer_original": "yes",
  "expected_answer_masked": "no"
}
```

### Figure 1（主图）
**Accuracy × Δ-Flip 散点**：x = Accuracy，y = Δ-Flip(matched-patch)，6 个点（6 个模型），不同颜色区分 medical vs general。**预期**：medical VLM 在右下（高 acc 低 Δ-Flip），暴露 shortcut。

### Figure 2（operator ablation）
每个模型一组柱状图：zero / blur / matched-patch 三种 operator 下的 Δ-Flip。**预期**：matched-patch 最严格（Δ-Flip 最低），zero 最 leaky。

### Figure 3（qualitative）
2×3 网格：原图 + matched-patch masked 图 + 模型答案对比，覆盖 (high-SCI 模型, low-SCI 模型) × (cardiomegaly, pneumonia, lung lesion)。

### Table 1（主表）
| Model | Acc | Δ-Flip(zero) | Δ-Flip(blur) | Δ-Flip(MP) | **SCI** |
|---|---|---|---|---|---|

### Table 2（per-pathology）
10 个病灶 × 6 个模型的 SCI heatmap。

---

## 12. 预期 main finding + backup findings

### A. Main finding（80% 概率出现）
```
Medical-pretrained VLMs (LLaVA-Med, HuatuoGPT-Vision, MedGemma) show
higher accuracy but lower Δ-Flip than general VLMs (Qwen2.5/3-VL,
InternVL3), implying their accuracy gain partially comes from
report-style/demographic shortcuts rather than improved visual grounding.
```

### B. Backup 1（如果 A 不成立 — 大家 Δ-Flip 都很高）
```
All current VLMs exhibit high evidence sensitivity on chest X-ray,
but per-pathology breakdown reveals that {cardiomegaly, opacity} are
systematically less grounded than {nodule, pneumothorax} — pointing
to anatomical shortcuts where global structure leaks into the answer.
```

### C. Backup 2（如果 A 和 B 都不成立 — 一切都正常）
```
Even when accuracy and Δ-Flip correlate, the 3 mask operators reveal
divergent failure modes (zero-mask leaks shape, blur leaks texture,
matched-patch leaks none), and we propose Δ-Flip(matched-patch) as the
standard audit metric for medical VLM deployment.
```

**至少 A/B/C 之一一定成立**（C 是 method-protocol 类，托底）。你 D7 决定走哪条。

---

## 13. Stop conditions（D7 前必读）

| 症状 | 决定 |
|---|---|
| 6 个模型 Δ-Flip 全部 > 80% 且 SCI < 10% | A 不成立 → 切 B |
| sanity check 失败率 > 30%（mask 没擦掉病灶） | mask operator 调参；若 D3 还不行，扩到 MS-CXR / VinDr-CXR |
| 某个 HF model gating 拿不到 | **立刻 ping PI**；不要 silently 换；最坏情况降到 5 个模型 |
| CheXpert classifier confidence drop 不显著 | 换一个 chest X-ray classifier（DenseNet121 from torchxrayvision） |
| 跑了 3 天 model 推理还没出结果 | 一定有 bug；不要硬调；找 PI 一起 debug |
| D10 figure 看不出 story | 不要硬解释；C 托底永远成立 |

---

## 14. 不要做的事（明确禁止）

1. **不要训练任何模型**。这是 inference-only audit。
2. **不要自己加 operator**（saliency mask, attention mask, semantic mask）。三个就三个，多了 ablation 维度爆炸。
3. **不要扩 modality**（pathology, MRI, CT）。v1 chest X-ray only。
4. **不要换 prompt**。所有模型用同一个 prompt，否则跨模型对比污染。
5. **不要做 multi-turn / agent**。single-shot only。
6. **不要在 paper 里 claim "medical VLM 应该用 X 方法"**。这是 audit paper，不是 method paper。最强 claim 上限：*"Δ-Flip should be reported alongside accuracy for medical VLM deployment."*
7. **不要重写 NVT 的 mask code**。复用，加 bbox 适配层就够。
8. **不要写"未来可以...."类的虚 future work**。要写就写一条具体的（"OPD-based fix that uses Δ-Flip as training signal will be reported in follow-up work"）。

---

## 15. 包内复用资产清单

所有 ready-to-use 资产已经在这个 handoff 包内，**你不需要从 Qizhen 的服务器拿任何东西**。

| 用 | 包内位置 |
|---|---|
| Mask operator (zero / blur / matched-patch) | `code/cfa_mask_operators.py` (ready-to-import) |
| NVT 原始 reference 实现 | `code/_nvt_apply_operator.py` (只读参考) |
| Mask sanity 检查 (SSIM-based) | `code/cfa_mask_operators.py:sanity_check_mask()` |
| WACV LaTeX template + lineno.sty v5.5 修复 | `paper_template/{wacv.sty, lineno.sty, ieeenat_fullname.bst}` |
| LaTeX preamble (含 \\method, \\todo 宏) | `paper_template/preamble.tex` |
| LaTeX 主文件骨架 | `paper_template/main_skeleton.tex` |
| 文献地图 (arxiv ID + takeaway + 攻击预案) | `literature/related_work_zh.md` |

### Experiment Tracker 模板（D2 起自建）

参考下面这个最小结构：

```markdown
# MedCFA Experiment Tracker

## Claim Gates
| Gate | 要求 | 状态 | 证据 |
|---|---|---|---|
| G1 | CheXlocalize 数据下载 + ≥ 4000 VQA pairs | ? | ? |
| G2 | Mask sanity 通过率 ≥ 90% | ? | ? |
| G3 | Qwen2.5-VL-7B dry run Acc ≥ 50%, Δ-Flip > 0 | ? | ? |
| G4 | 6 模型全部 preds.jsonl 落盘 | ? | ? |
| G5 | MS-CXR 第二 benchmark Δ-Flip 排序一致 | ? | ? |
| G6 | paper-claim-audit + citation-audit clean | ? | ? |

## Runs
| Run ID | 任务 | Model | Operator | 状态 | 输出路径 |
|---|---|---|---|---|---|
| R001 | Qwen2.5-VL-7B dry run | qwen2_5_vl_7b | all | ? | $MEDCFA_RESULTS/r001/ |
| R002 | ... | ... | ... | ? | ... |
```

---

## 16. Agent / Skill 调用建议（按时序）

**强制必跑的 skill**（不跑 = 这个 paper 不要交）：

- D11 写作：`paper-write`（按 outline 生成 LaTeX）
- D13 审 claim：`paper-claim-audit`（零 context cross-model 审计每个数字）
- D13 审引用：`citation-audit`（零 context cross-model 审引用真实性）
- D14 对抗：`kill-argument`（找 reviewer-2 风险，**这一步是 reject 拦截器**）
- D14 编译：`paper-compile`

**可选 skill**：

- D8 figure：`paper-figure`（给我 4 个 figure 的初稿）
- D10 figure：`figure-designer`（如果 figure 3 case study layout 卡住）
- D11-12 outline 检查：`benchmark-paper-template` 看 audit/benchmark 类 paper 的 5-pillar 是否齐全
- 任何阶段 stuck：`codex:rescue` 让 codex 跑一遍诊断

**parallel agent 用法**（learn from PI 的 workflow）：

每跑一个模型推理可以放 background：

```bash
# 假设 cfa_run_audit.py 接受 --model 和 --operator flag
for model in llava_med huatuogpt medgemma qwen25vl qwen3vl internvl3; do
  for op in zero blur matched_patch; do
    tmux new -d -s "audit_${model}_${op}" \
      "bash scripts/run_audit_${model}_${op}.sh > logs/audit_${model}_${op}.log 2>&1"
  done
done
```

或者用 `experiment-queue` skill 一把梭。

---

## 17. 写作 outline（LaTeX section plan，按这个顺序写）

```
Section 1  Introduction (~1 page)
  para 1 — 医学 VLM accuracy 进展快，但 deployment 信心来自 accuracy
  para 2 — accuracy 可能来自非视觉证据 shortcut
  para 3 — 现有 benchmark 不测因果证据依赖
  para 4 — 我们引入 mask-based counterfactual audit 协议
  para 5 — 主要发现 (3 个 bullet)
  para 6 — contributions (3 个 bullet)

Section 2  Related Work (~0.75 page)
  - Medical VLM faithfulness (Med-SCoT, HEAL-MedVQA, MedVCTP)
  - Counterfactual evaluation (NaturalBench, CounterCurate, CF-VQA)
  - Hallucination benchmarks (POPE, HallusionBench)
  - Why none of them measures what we measure (1 sentence each)

Section 3  Audit Protocol (~1.5 page)
  3.1 problem definition
  3.2 dataset (CheXlocalize stats + how we build VQA pairs)
  3.3 counterfactual operators (zero / blur / matched-patch) + Figure shows examples
  3.4 metrics (Acc / Δ-Flip / SCI formal definitions)
  3.5 sanity checks (SSIM + classifier-confidence-drop)

Section 4  Experiments (~2 page)
  4.1 models & inference setup
  4.2 main results (Table 1 + Figure 1)
  4.3 per-pathology breakdown (Table 2)
  4.4 operator ablation (Figure 2)
  4.5 qualitative cases (Figure 3)

Section 5  Discussion (~0.75 page)
  5.1 medical vs general VLM 的 SCI gap 含义
  5.2 mask-based vs diffusion-based counterfactual trade-off
  5.3 何时该用 Δ-Flip 评估 medical VLM
  5.4 限制：single modality, single language, no radiologist verification

Section 6  Conclusion (~0.25 page)
```

总长度目标 8 页正文（WACV 限制）+ supplementary。

---

## 18. 路径约定（你的服务器你自己定）

所有路径都通过 `$MEDCFA_STORAGE`（你在 `setup/env.sh` 里填的根目录）派生。**包内文档里所有路径变量都已经用 `$MEDCFA_*` 引用**，不要硬编码绝对路径。

| 类型 | 路径模板 |
|---|---|
| 数据 | `$MEDCFA_DATA/chexlocalize/raw/` |
| 模型 cache | `$HF_HOME/`（自动派生自 MEDCFA_STORAGE） |
| 结果 | `$MEDCFA_RESULTS/<model>/<operator>/preds.jsonl` |
| Logs | `$MEDCFA_LOGS/audit_<model>_<operator>.log` |
| 本地 LaTeX | （你自己定，建议 `$MEDCFA_STORAGE/paper/` 或 git repo 里） |

---

## 19. Env vars

见 `setup/env_template.sh`。你**必须**先按 `setup/README_zh.md` 配好自己的 `env.sh`，每个新 shell `source` 一遍。所有脚本默认假设 env 已 source。

---

## 20. Sync protocol (Yuke ↔ Qizhen)

### 例会节奏

**至少每周一次**，期间 unblock 消息任何时候都可以发。

### 周会必带（Yuke 准备）

1. 当前 EXPERIMENT_TRACKER.md 最新行（不超过 5 个 Run）
2. 上周的关键 finding（无论正负）
3. 下周打算解的 3 个问题
4. **当前 D-day 在 §9 实验 matrix 的哪一格** —— 落后不可怕，没汇报落后不行

### 必须立即 ping 的触发（不等周会）

- 任何 stop condition（§13）触发
- 任何 HF model gating 拿不到
- mask sanity 失败率 > 30%
- D7 决策门到了（必须 PI 在线一起看主表）
- finding 走 A / B / C 哪条还拿不准
- 任何 reviewer-risk 的发现（比如撞到一个 2025-2026 的近似工作）

### 沟通渠道

- 同步：周会
- 异步 unblock：邮件 / 即时通讯（Qizhen 优先回 stop-condition 触发 和 D7 决策）
- 文件：本地 git folder（每天结束 push commits 到 `paper/medcfa` branch，msg 用 `<day>: <one-line summary>`）

### 我（Qizhen）的承诺

- **D7 第一决策门**：3 模型主表 push 后 24h 内 review，不让你卡在 PI 等待
- **D14 第二决策门**：完整 6 模型 + MS-CXR 跨 benchmark 结果 push 后 24h 内 review
- **D17 outline check**：intro + protocol + experiments 草稿 push 后 24h 内反馈
- **D21 终审**：PDF 编出来当天读完，决定 submit / hold

---

## 21. 最后一句话

这个 paper 有 **两个命门**：

- **D7 单 benchmark 主表的 SCI 数字** → finding A/B/C 走哪条
- **D14 跨 benchmark (MS-CXR) 复现** → 故事强度（强 finding 还是 protocol-only finding）

前 14 天工程跑完，那两张数表会告诉我们 paper 的 ceiling。无论走哪条，**v1 都是 R1 可投稿状态**——这正是为什么我们用 mask、用单 modality、用 audit-only。

不要追求宏大，追求 **干净 + 快 + 可解释**。3 周后我们看 PDF。

— Qizhen, 2026-05-28
