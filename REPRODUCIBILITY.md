# Reproducibility checklist

Reference implementation for **"The Axis of the Sign Is Not Scale: Evaluation Protocol
Determines the Sign of the Scale Effect in Multimodal RAG."**

Everything cited in the paper traces to a JSON file in this repository and, from there, to a
script in `src/`. This document records the artefacts, the runtime, the mapping from scripts to
numbers, and the reproduction pitfalls we hit ourselves.

---

## 1. Artefacts

### Datasets

| File | Rows | What it is |
|---|---|---|
| `data/syc6_927.jsonl` | 927 | The controlled conflict samples: counterfactually edited question, parametric answer, conflicting context, alias sets |
| `data/dstar_649.json` | 649 | The anchored set $D^* = \bigcap_s \{i : K_s(i)=1\}$ — samples every scale point knows |
| `data/medusa_anchors.jsonl` | 236 | Anchoring-protocol intermediates (image resolution, knowledge screen) |
| `data/medusa_combined.jsonl` | 1146 | Combined conflict sample set before anchoring |

### Raw decoding output

`results_adjudicate/` holds the raw output for the eighth stance `adjudicate` on all five
models, the direct input to the pre-registered prediction test.

The seven-stance ladder's raw decoding output is produced by `src/eval_sycophancy.py` (one file
per model and arm; the NO-CTX knowledge arm comes from `src/eval_pkd.py`). It is decoded against
a served model, so it is regenerated rather than shipped; `src/analyze_scale_ladder.py` reads it
from `--probe` and emits the analysed products below.

### Analysis products

`results/*.json` are the analysis products the paper cites directly: `scale_ladder.json`
(the seven-stance $\Delta$ table), `scale_ladder_negation.json` (the negation-aware criterion),
`negation_rescue_counts.json` and `negation_rescue_rates.json` (the rescue counts behind the
criterion correction), `stance_dose_response.json` / `..._alias.json` (the dose-response fits),
`length_confound.json` (the length-confound exclusion), `stratified.json` (the per-slot
stratification), `mde.json` (minimum detectable effect), `interference_*.json` (the two-arm
vision/text comparison), `cross_env_*.json` (cross-environment replication),
`adjudicate_test*.json` (the pre-registered prediction test).

---

## 2. Runtime

Analysis and figure generation run on CPU.

```
Python      3.9.13
numpy       2.0.2
matplotlib  3.9.4
scipy       1.13.1
```

Model inference is served by a vLLM OpenAI-compatible endpoint; the scripts talk to it over
HTTP and never load weights themselves. Decoding: `temperature=0`, `seed=0`,
`max_tokens=2048`, `--no-vision` for the text arm, `--question-field=named`,
`--ctx-field=conflict`.

Models used: Qwen3-VL 2B / 4B / 8B, Qwen2.5-VL 3B / 7B (Instruct). Weights are not shipped
(see `README.md`); fetch them from HuggingFace or ModelScope and serve with:

```
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-2B-Instruct --port 8000 --limit-mm-per-prompt image=1
```

---

## 3. Scripts to numbers

| Script | Emits | Paper location |
|---|---|---|
| `analyze_scale_ladder.py` | `scale_ladder.json` | the seven-stance $\Delta$ table (main result) |
| `regrade_negation.py` | `scale_ladder_negation.json`, `negation_rescue_counts.json` | the negation-aware criterion and rescue counts |
| `regrade_aliases.py` | `stance_dose_response_alias.json` | length-invariant alias criterion |
| `analyze_dose_response.py` | `stance_dose_response.json` | the dose-response fit over the stance ladder |
| `analyze_length_confound.py` | `length_confound.json` | the length-confound exclusion |
| `analyze_stratified.py` | `stratified.json` | the per-slot stratification |
| `analyze_mde.py` | `mde.json` | minimum detectable effect |
| `analyze_interference.py` | `interference_*.json` | the vision/text arm comparison |
| `cross_env_compare.py` | `cross_env_*.json` | cross-environment replication |
| `analyze_adjudicate.py` | `adjudicate_test*.json` | the pre-registered prediction test |
| `analyze_rescue_counts.py` | `negation_rescue_counts.json` | rescue counts per cell |
| `anchor_ladder.py` | `ladder_anchors.json` | the anchoring protocol $D^*$ |
| `eval_harness.py` + `rescore_harness.py` | `results/probe/_harness_summary.json` | the framework-replication table |
| `make_figures.py`, `make_tables.py` | figures, LaTeX tables | all figures and tables |

The raw collection scripts are `eval_sycophancy.py` (stance ladder), `eval_pkd.py` (knowledge
arm), `eval_harness.py` (framework replication); the data-construction chain is
`build_data.py` → `map_images.py` → `screen_knowledge.py` → `verify_anchors.py`.

Recompute statistics, then redraw:

```
cd src
python analyze_scale_ladder.py --probe ../results/probe --out ../results/scale_ladder.json
python regrade_negation.py --probe ../results/probe --out ../results/scale_ladder_negation.json
python make_figures.py
```

---

## 4. Reproduction pitfalls

Each of these is a mistake we made and fixed. They are listed because each one silently
produces plausible but wrong numbers.

1. **Anchoring is not optional.** Restricting to the intersection of sample *keys* across the
   stance files is a no-op — all files carry the same 927 keys. The anchor must come from the
   NO-CTX knowledge arm (`pkd_*_text.json`). Skipping it leaves samples a small model cannot
   know in the denominator, which systematically lowers $\text{PKD}_{\text{small}}$ and
   *inflates* $\Delta$.

2. **Explicit rejection is not "undecidable."** A plain alias criterion files answers that
   reject the context — which name the wrong value in order to reject it — under BOTH and drops
   them from the denominator. The more clearly a model rejects, the more likely it is dropped,
   and the drop rate rises with scale. The negation-aware criterion repairs this direction, and
   both criteria must be reported.

3. **A length-invariant criterion is required.** The evidence-grounded criterion (`rescore_grounded.py`)
   correlates with stance answer length at $r = 0.983$ across the seven stance aggregates,
   manufacturing a dose-response curve. The alias criterion is length-invariant; use it.

4. **Quantitative counterfactuals need a single-pass substitution.** Replacing digits on an
   already-rewritten string lets the new number be matched by a later pattern, producing
   `7.2.35 to 4 eggs`. Use one `re.sub` pass over the original.

5. **iNaturalist category ids are zero-padded.** `6513` must become `06513` before matching
   `val.json`, or the intersection is 0% and the mapping looks infeasible.

6. **Count-shaped quantities must stay integers and stay distinct.** A 45–65% scale factor
   rounds small integers back to their original value, yielding a "counterfactual" identical to
   the original; and `1 eggs` / `3 meter` need noun-number agreement.

7. **Substring matching breaks on short tokens.** Putting `in` inside a word-boundary pattern
   matches the preposition; `edible` is a substring of `inedible`. Anchor on word boundaries
   and mask the entity name.

---

## 5. Manual verification

The negation-aware criterion is a string rule, not a semantic judge. To check that its pattern
table is not tailored to one stance, we hand-annotated the cell with the largest rescue volume
(Qwen3-VL 8B, `ctx_hedge`): with seed 20260924, 19 rescued items were sampled and read
individually. All 19 state the parametric truth and explicitly reject the context.

The reverse check sampled 19 items the rule still keeps as BOTH, across stances and scale
points: 10 restate the context value, 7 restate a polarity slot, 2 hedge between both values.
No true rejection was found among them.

---

## 6. Licences and provenance

The conflict samples derive from E-VQA and iNaturalist annotations, redistributed under their
original licences; images are not redistributed. Third-party paper PDFs are not redistributed.
