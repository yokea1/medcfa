# paper_template/ —— WACV 2027 LaTeX 模板

## 文件清单

| 文件 | 来源 | 你要不要改 |
|---|---|---|
| `wacv.sty` | WACV 2027 官方 author kit | **绝对不要改** |
| `lineno.sty` | 本地 v5.5 修复版 | **不要改**。这是修复 WACV gutter 行号问题的专用版本 |
| `ieeenat_fullname.bst` | WACV 2027 官方引用风格 | **不要改** |
| `preamble.tex` | MedCFA 适配的宏定义 | 可以加新 macro；不要删现有 |
| `main_skeleton.tex` | 主文件骨架 | **复制为 `main.tex` 之后再改** |

## 上手步骤

```bash
# 在你的 paper 工作目录下
mkdir paper/
cp paper_template/* paper/
cd paper/
cp main_skeleton.tex main.tex
mkdir sections/

# 创建空 section 文件占位
for s in 00_abstract 01_intro 02_related_work 03_protocol 04_experiments 05_discussion 06_conclusion; do
  echo "% TODO: $s" > sections/${s}.tex
done

# 创建空 bib
touch main.bib

# 尝试编译（应该报 "no bibliography" 警告但 PDF 能出）
latexmk -pdf main.tex
```

## ⚠️ 关于 lineno.sty（必读，否则会卡）

WACV 2027 默认编译会出现**行号和正文叠在一起**的 bug。本目录里这份 `lineno.sty` 是 v5.5 修复版。

**不要做的事**：
- 不要替换为系统 lineno.sty（很多 TeX 发行版自带的是旧版本，会复发 bug）
- 不要在 `preamble.tex` 里加 lineno 相关 patch
- 不要动 `wacv.sty` 来"修" 这个问题

**正确做法**：保持本目录的 `lineno.sty` 不动，LaTeX 编译时它会优先于系统版本被找到（因为同目录优先级高）。

## 投稿 vs Camera-ready

`main_skeleton.tex` 顶部：

```latex
\usepackage[review,algorithms]{wacv}
```

- `review` 模式：双盲，有行号，作者匿名 → **投稿时用**
- `final` 模式（提交 camera-ready 时改）：显示作者，无行号

提交前 grep 一遍 anonymity：

```bash
grep -niE "qizhen|yuke|lan|\\bhe\\b|institution|university" sections/*.tex
```

发现任何识别信息**必须改成 anonymous reference**。

## 编译失败的常见原因

| 报错 | 原因 | 修复 |
|---|---|---|
| `! LaTeX Error: File 'wacv.sty' not found.` | 工作目录不对 | `cd paper/`，确认所有 `.sty` 都在当前目录 |
| `! Package keyval Error: review undefined.` | 不是用官方 `wacv.sty` | 重新 `cp paper_template/wacv.sty paper/` |
| 行号和正文叠在一起 | lineno.sty 没生效 | 检查 `paper/lineno.sty` 存在且是 154K 那个版本 |
| 引用全是 `[?]` | bibtex 没跑 | `pdflatex main && bibtex main && pdflatex main && pdflatex main` |
| `! Undefined control sequence \method` | preamble.tex 没 input | `main.tex` 里有 `\input{preamble}` 这一行吗 |

## 提交前的 LaTeX checklist

- [ ] `main.pdf` 编译成功，无 error
- [ ] 没有 `\todo{}` / `\fix{}` / `\check{}` 残留（preamble 里定义的 debug 宏）
- [ ] 没有作者 / 机构识别信息泄漏（grep 上面那条）
- [ ] 引用全部解析（无 `[?]`）
- [ ] 8 页正文 + 任意页 references
- [ ] line number 显示正常（不和正文重叠）
- [ ] 跑过 `paper-claim-audit` skill
- [ ] 跑过 `citation-audit` skill
- [ ] 跑过 `kill-argument` skill
