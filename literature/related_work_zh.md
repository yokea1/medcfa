# Related Work — Pre-Survey (don't re-survey from zero)

预先做好的文献地图。**每条都标了 arxiv ID / 出处和 takeaway**。Yuke 在 D2 之前读 5-7 条最相关的（标 ★ 的），其他作为引用候选。所有 arxiv ID 都建议在 D13 `citation-audit` 阶段交叉验证（少数我标了 `[verify]`）。

## A. 直接 prior art（必读、必引）

### A1. Counterfactual reasoning in VQA / VLM

★ **CF-VQA (Niu et al., 2021)** — `arXiv:2006.04315`
> 用 counterfactual inference 移除 question-only language bias。和我们的不同：他们干预 question 不动图，我们干预图不动 question。

★ **NaturalBench (Li et al., NeurIPS 2024)** — `arXiv:2410.14669`
> 用人工对照对 (original, counterfactual) 测 VLM；general domain。我们是 medical domain + automatic mask-based。

**CounterCurate (2024)** — `arXiv:2402.10632` [verify]
> 合成 counterfactual VQA pairs 用于 fine-tuning。和我们的差别：他们造数据训模型，我们造数据 audit 模型。

### A2. Hallucination / faithfulness benchmarks

★ **POPE (Li et al., EMNLP 2023)** — `arXiv:2305.10355`
> object-presence hallucination benchmark。我们是它的"causal 版"——POPE 测"模型说看到了 X 吗"，我们测"X 被擦掉模型还说看到吗"。

**HallusionBench (Liu et al., CVPR 2024)** — `arXiv:2310.14566`
> 大型多模态 hallucination 评测。包含 visual / linguistic / hybrid。我们 specifically 测 visual-only causal dependence。

### A3. Medical VLM benchmarks (背景，节选引)

**OmniMedVQA-V2** — `arXiv:2402.09181` [verify]
> 大规模 medical VQA benchmark。引用为 background；不当 baseline。

**SLAKE / VQA-RAD / PathVQA** — `arXiv:2102.09542` (SLAKE) / `arXiv:1810.09786` (VQA-RAD)
> 经典 medical VQA 数据集。supplementary 里提一句 trans-generalization 可能性。

### A4. Chest X-ray grounding data（我们的数据源）

★ **CheXlocalize (Saporta et al., Nature MI 2022)** — `arXiv:2207.04106`
> Stanford 出的 chest X-ray classification + bbox 标注。**我们直接用其 test set**。要引为 dataset 来源。

★ **CheXpert (Irvin et al., AAAI 2019)** — `arXiv:1901.07031`
> CheXlocalize 的图像源。简提。

**MS-CXR (Boecking et al., ECCV 2022)** — `arXiv:2204.09817`
> MIMIC-CXR + sentence-level bbox。我们 v1 不用，但 supplementary 列为可扩展数据源。

**VinDr-CXR (Nguyen et al., 2022)** — `arXiv:2012.15029`
> 越南 chest X-ray 多类 bbox。同上备选。

### A5. Medical VLM 模型（被审计的对象）

★ **LLaVA-Med (Li et al., NeurIPS 2023 Datasets)** — `arXiv:2306.00890`
> Microsoft 出，第一代 instruction-tuned medical VLM。v1.5 是 mistral-based。

★ **HuatuoGPT-Vision (Chen et al., 2024)** — `arXiv:2406.19280`
> 中国出的医疗 VLM；大规模中文医疗数据训练。

★ **MedGemma (Google, 2025)** — Google blog + tech report [verify arxiv]
> Google 2025 出的医疗 multimodal 模型。4B / 27B 变体。

**InternVL3 (Chen et al., 2024)** — `arXiv:2412.05271` [verify]
> 通用 VLM；我们当 general baseline。

**Qwen2.5-VL (Bai et al., 2025)** — `arXiv:2502.13923` [verify]
> 通用 VLM；强 baseline。

**Qwen3-VL (2025)** — official tech report [verify arxiv]
> 我们用 8B-Instruct 变体。

### A6. Grounded / structured medical CoT（不和我们撞，要标 distinguish）

**HEAL-MedVQA / LobA** — `arXiv:2505.00744` [verify]
> localize-before-answer for medical VQA。我们不是 localize，我们是 counterfactual audit。**这条要在 §2 Related Work 显式 distinguish**。

**S-Chain (2026)** — project page `s-chain.github.io` [verify arxiv]
> bbox + structured visual medical CoT。**Related Work 显式 distinguish**：他们用 bbox 当推理监督，我们用 bbox 当因果干预 mask。

**Med-SCoT / Step-CoT / MedVCTP** — 你不用读 paper，借 NVT archive `experiments/EXPERIMENT_TRACKER.md` G1 行直接引：medical structured CoT family，引一下，说我们不是 structured CoT 而是 audit。

## B. 远端 prior art（仅简提，supplementary 引）

- **Counterfactual Visual Explanations (Hendricks et al., 2018)** — `arXiv:1802.07451` — pre-VLM 时代的 saliency-counterfactual。
- **CLIP-style faithfulness probes** — POPE 之外的视觉 grounding 评测，作为方法 lineage 简述。

## C. 反向 prior art（提防 reviewer 引来 attack 我们）

reviewer 可能用这些 paper 说"已经有人做过"——你要准备好 1-2 句 differentiator：

| 候选攻击 paper | reviewer 可能问 | 你的回答 |
|---|---|---|
| ROAR (Hooker et al., 2019) `arXiv:1806.10758` | "你 mask 后 retrain 看影响——这是已有方法" | ROAR 是 retrain saliency benchmarking；我们 inference-only 且**不重训**，目标是 audit 而非 attribution |
| Smoothgrad / Grad-CAM 之类 saliency methods | "你这只是 saliency 评估" | saliency 是用模型自己产 explanation；我们用 dataset bbox 产 ground-truth necessity，不需要模型 explainer |
| CounterCurate / NaturalBench-Medical [verify exists] | "你这就是 medical 版 NaturalBench" | NaturalBench 是 human-curated text+image pairs；我们是 mask-based automatic operator + 因果隔离 + medical bbox-aligned |

## D. 文献检索建议

D2 之前用以下 query 在 `semantic-scholar` 和 `arxiv` skill 各搜一次：

1. `"counterfactual" AND ("VQA" OR "vision-language") AND "medical"` → 过去 18 个月的接近工作
2. `"chest x-ray" AND ("hallucination" OR "faithfulness" OR "grounding") AND (VLM OR "vision-language")` → 同 domain 同问题
3. `"CheXlocalize" OR "MS-CXR"` AND "audit" → 检查这两个 dataset 是否被用过做类似 audit（我们要保证没人抢先）

每个 query 看 top-20，记录"如果有撞车风险就 ping PI"的 paper 到 `literature/literature_log.md`。
