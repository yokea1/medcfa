# MedCFA 论文交接包 —— 给 Yuke

收件人：**Yuke He**
发件人：**Qizhen Lan** (`lanqz7766@gmail.com`)
日期：2026-05-28
目标：**WACV 2027 Round 1 投稿**（deadline 约 2026-06-中下旬，3 周后；具体日期 Qizhen 会确认）
**实际工期：3 周**（实验 + 写作 + 提交准备全部完成、可投稿状态）

> **注意 R1 vs R2 的差别**：R1 接收率更高、reviewer 时间更宽松、reviewer 期望更高质量。我们走 R1 意味着 v1 paper 的故事必须**比 R2 兜底版本更扎实**——具体的策略调整见 `HANDOFF_zh.md` §0.5 Position 部分。

---

## 这个包是什么

一份**自包含**的论文交接包。所有你需要的文件都在这个目录里——文档、参考代码、WACV 模板、文献地图——不依赖任何远端 server。你按自己的环境配好后即可上手。

## 先看哪个文件（**严格按这个顺序**）

1. **`README_zh.md`**（你现在在读）
2. **`setup/README_zh.md`** —— 5 分钟配好你自己的环境
3. **`DAY_1_zh.md`** —— 第一天 6 个小时逐小时计划，含真实 traps
4. **`HANDOFF_zh.md`** —— 完整规格（§0、§0.5、§0.7、§9、§14、§20 是必读，其他细节按需翻）
5. **`literature/related_work_zh.md`** —— 文献地图（D2 之前选 3-5 篇 ★ 读）
6. **`code/README_zh.md`** —— 参考代码用法
7. **`paper_template/README_zh.md`** —— WACV 模板上手

## 目录结构

```
handoff_pkg/
├── README_zh.md                ← 你现在读的
├── HANDOFF_zh.md               ← 完整论文规格（632 行）
├── DAY_1_zh.md                 ← 第一天逐小时计划
│
├── setup/
│   ├── README_zh.md            ← 如何配你自己的环境
│   ├── env_template.sh         ← 环境变量模板（你填路径）
│   ├── download_chexlocalize.sh  ← 数据下载脚本（含 fallback）
│   └── requirements.txt        ← Python 依赖
│
├── code/
│   ├── README_zh.md            ← 如何使用参考代码
│   ├── cfa_mask_operators.py   ← ready-to-use mask 操作（zero/blur/matched_patch）
│   └── _nvt_apply_operator.py  ← NVT 原始参考实现
│
├── literature/
│   └── related_work_zh.md      ← 文献地图（arxiv ID + takeaway + reviewer 攻击预案）
│
└── paper_template/
    ├── README_zh.md            ← 如何使用 WACV 模板
    ├── preamble.tex            ← LaTeX 宏定义
    ├── main_skeleton.tex       ← 你写论文的起点
    ├── wacv.sty                ← WACV 2027 官方模板（别动）
    ├── lineno.sty              ← v5.5 修复版（解决 WACV gutter 行号问题，别动）
    └── ieeenat_fullname.bst    ← 引用风格
```

## 时间表（高层，3 周节奏）

```
Week 1: 数据 + pipeline + 单模型 dry run
  Day 1     ← setup + smoke test
  Day 2-3   ← mask code + 数据配对生成 + sanity
  Day 4-5   ← Qwen2.5-VL-7B dry run（验证 pipeline）+ metric 脚本
  Day 6-7   ← 第一波模型推理（3 个）+ Day 7 **第一决策门**
              （和 Qizhen review 单模型结果，决定后续模型是否调整）

Week 2: 完整 audit + 第二 benchmark + 主要 finding
  Day 8-10  ← 平行跑剩余 3 个模型 × 4 condition；主表 v1
  Day 11    ← per-pathology + operator ablation 分析；Figure 1-2 v1
  Day 12-13 ← **MS-CXR 第二 benchmark**（R1 需要这一步证明 generalization）
  Day 14    ← **第二决策门**：和 Qizhen 一起 review 完整结果，定 finding A/B/C

Week 3: 写作 + 三层 audit + 提交准备
  Day 15-17 ← LaTeX 写作（intro + protocol + experiments + discussion）
  Day 18    ← `paper-claim-audit` + `citation-audit`
  Day 19    ← `kill-argument` 对抗 review；按结果改 3-5 处
  Day 20    ← 终稿 polish + figure caption 优化 + anonymity 检查
  Day 21    ← `paper-compile` 终编译；Qizhen 终审；submit
```

## 沟通

- **同步**：每周至少一次例会
- **异步 unblock**：邮件 / 即时通讯，Qizhen 优先回 stop-condition 和 D7 决策门
- **必须立即 ping Qizhen**（不等周会）的情况：见 `HANDOFF_zh.md` §20

## 你需要的硬件

- 2× A100 80GB（或等效）
- ~100 GB 磁盘空间（模型 cache + 数据 + 中间结果）
- 稳定网络（HuggingFace 模型下载约 50-80 GB 总量）

## 起跑第一句话

> 你不需要从头设计这个 paper。Qizhen 已经把方案降级到最快档（mask-based，不用 diffusion；纯推理，不训练；单 modality）。**你的任务是干净地把方案执行下来，2 周后我们看 PDF。**

任何阻塞或疑问，直接发消息。不要在不确定的细节上自己 spin。

— Qizhen，2026-05-28
