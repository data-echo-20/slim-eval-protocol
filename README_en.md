# slim-eval-protocol

Reproduction material for **"The Axis of the Sign Is Not Scale: Evaluation Protocol
Determines the Sign of the Scale Effect in Multimodal RAG."**

---

## Result in one sentence

On the same batch of controlled conflict samples, under the same decision criterion, with the
same models, changing only the system prompt's **authorisation strength for parametric
knowledge** (seven pre-registered ordered stances), the change in the rate at which the model
follows parametric knowledge, $\Delta$, **crosses zero**:

| Authorisation | Stance | Qwen3-VL 2B→8B | Qwen2.5-VL 3B→7B |
|---|---|---|---|
| weak | `ctx_only` | $-27.3$ pp | $-8.4$ pp |
| weak | `orig` | $-23.7$ pp | $-13.8$ pp |
| — | `ctx_hedge` | $+29.2$ pp | $+11.6$ pp |
| mid | `neutral` | $+1.5$ pp | $+15.9$ pp |
| strong | `own_only` | $+12.1$ pp | $+21.7$ pp |
| strong | `len_ctrl` | $+21.4$ pp | $+12.3$ pp |
| strong | `prior` | $+28.0$ pp | $+33.7$ pp |

The **sign** of the scale effect is not a function of model size but of the evaluation
protocol, and the structure reproduces independently across two model families. The paper also
reports three directionally clear measurement biases — explicit rejection misread as
undecidable, denominator collapse misread as the strongest effect, and position preference
masquerading as content judgement — each of which at some point reversed one of our own
conclusions.

---

## Contents

| Path | Contents |
|---|---|
| `paper_zh/` | Chinese version: final `paper_cjc_v11.pdf`, LaTeX project, figure scripts |
| `paper_en/` | English version: final `paper.pdf`, LaTeX project, chunked Markdown source |
| `supplement/` | Supporting tables, the pre-registration file, tables moved out of the body |
| `src/` | All analysis scripts (criterion regression, stratified analysis, prediction test, figures) |
| `results/` | Analysis-product JSON cited by the paper (direct source of every table and statistic) |
| `results_adjudicate/` | Raw collection output for the eighth stance `adjudicate` |
| `data/` | Core inputs: the 927 conflict samples, the 649-item anchored set, anchoring intermediates |
| `figures/` | Vector PDF sources for every figure (English set) |

---

## Not included

The following are large and re-obtainable, so they are not shipped:

| Excluded | Size | How to obtain |
|---|---|---|
| Model weights: Qwen2.5-VL-3B/7B, Qwen3-VL 2B/4B/8B | ~10 GB | HuggingFace / ModelScope |
| Raw datasets (E-VQA, iNaturalist, …) and images | ~18 GB | official dataset pages |
| Third-party paper PDFs | 66 MB | download by arXiv ID |
| Remote vLLM service logs | 86 MB | inference timing only, source of no number |

The drawing font Noto Sans SC is also excluded; any installed system font will do.

---

## Citation

If you use this material, please cite the paper (under submission, 2026).
