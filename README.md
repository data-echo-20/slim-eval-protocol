# slim-eval-protocol

复现材料：**《符号的轴不是规模：多模态 RAG 知识冲突中评测协议对规模效应的支配》**

The axis of the sign is not scale: evaluation protocol determines the sign of the
scale effect in multimodal RAG.

---

## 结论一句话

在同一批受控冲突样本、同一判定口径、同一组模型上，只改变 system prompt 对参数知识的
**授权强度**（七档预注册的有序立场），模型跟随参数知识的比例变化量 $\Delta$ **跨零**：

| 授权强度 | 立场 | Qwen3-VL 2B→8B | Qwen2.5-VL 3B→7B |
|---|---|---|---|
| 弱 | `ctx_only` | $-27.3$ pp | $-8.4$ pp |
| 弱 | `orig` | $-23.7$ pp | $-13.8$ pp |
| — | `ctx_hedge` | $+29.2$ pp | $+11.6$ pp |
| 中 | `neutral` | $+1.5$ pp | $+15.9$ pp |
| 强 | `own_only` | $+12.1$ pp | $+21.7$ pp |
| 强 | `len_ctrl` | $+21.4$ pp | $+12.3$ pp |
| 强 | `prior` | $+28.0$ pp | $+33.7$ pp |

规模效应的**符号**不是模型尺寸的函数，而是评测协议的函数；该结构在两个独立模型族上
分别复现。论文另报告三处方向明确的测量偏倚（显式驳回被记为不可判定、分母塌陷被误读为
最强效应、位置偏好冒充内容判断），每一处都曾把我们自己的某条结论推向相反方向。

---

## 目录

| 路径 | 内容 |
|---|---|
| `paper_zh/` | 中文版论文：`paper_cjc_v11.pdf` 终稿、LaTeX 工程、绘图脚本 |
| `paper_en/` | 英文版论文：`paper.pdf` 终稿、LaTeX 工程、分块 Markdown 源 |
| `supplement/` | 补充材料：支撑表格、预注册文件、从正文移出的表 |
| `src/` | 全部分析脚本（判据回归、分层分析、预测检验、绘图） |
| `results/` | 论文引用的分析产物 JSON（每张表、每个统计量的直接来源） |
| `results_adjudicate/` | 第 8 档立场 `adjudicate` 的原始采集输出 |
| `data/` | 核心输入：927 条冲突样本、649 条锚定集、锚定协议中间产物 |
| `figures/` | 论文全部图的 PDF 矢量源（英文一套在 `figures/`，中文一套在 `paper_zh/figures/`） |

---

## 未随仓库提供

以下体积大且可重新获得，故未纳入：

| 未包含 | 体积 | 获取方式 |
|---|---|---|
| 模型权重 Qwen2.5-VL-3B/7B、Qwen3-VL 2B/4B/8B | ~10 GB | HuggingFace / ModelScope |
| 原始数据集（E-VQA、iNaturalist 等）与图像 | ~18 GB | 各数据集官方地址 |
| 第三方论文 PDF | 66 MB | 按 arXiv 编号自行下载 |
| 远端 vLLM 服务日志 | 86 MB | 仅含推理时序，非任何数字来源 |

绘图字体 Noto Sans SC 亦未纳入，用系统已装字体即可。

---

## 引用

若使用本材料，请引用论文（投稿中，2026）。
