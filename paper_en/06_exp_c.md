# 4.7 Stratified test

One explanation to rule out is that the scale effect is carried by a single editing type: if the
effect holds within only one slot, it may reflect a property of that slot rather than a general
interaction between scale and stance. This section recomputes the scale effect within each slot.
The risk of stratification is that a subset effect and the full-sample effect can have the same
direction but different significance, and reporting only one is misleading [30,43].

## 4.7.1 Slot distribution and results inside the two large slots

The slots are the semantic class of the word replaced by the counterfactual edit; within
$D^*$ only `place` and `other` have sufficient sample size, the other three comprising 28 items
in total. We compute $\Delta$ within each slot under the paired protocol (negation-aware
criterion, as in §4.2). The Qwen2.5 family's `place` slot is fully consistent with the overall
result, all seven stances significant, span 54.1 pp — the most complete single-slot replication
here — while its `other` slot is not, the `ctx_only` cell being $+1.1$ ($p = 0.832$) and by
§4.5.4 given no directional interpretation. The Qwen3 family is the opposite: all seven `other`
cells are significant and consistent (span 62.1 pp), whereas the two middle `place` cells are
not ($p = 0.440$, $0.111$). The only cross-family inconsistency is `other`/`ctx_only` (Qwen2.5
$+1.1$, not significant; Qwen3 $-7.6$, $p = 0.014$); that cell uses the same batch with
identical composition in both families, so slot composition cannot explain it, and we record it
faithfully as a limitation. The decidable samples for `ctx_hedge` within the two slots are ample
(374 and 193) against 12 and 8 under the pure alias criterion — the latter from denominator
collapse, not from the stance's true decidable volume.

## 4.7.2 Small slots: explicitly no conclusions drawn

For the three classes `diet`, `time`, and `quantity` the decidable samples within the anchoring
set drop to the teens or single digits; they are listed only to show that they are unusable
(Table S9, online material). The $\Delta$ values there can reach $+78.6$ pp numerically
(`prior` of `diet`), but rest on one or two flip pairs, with confidence intervals covering the
entire $[0,1]$. Most cells in the `time` slot are $0.0$ — not "no effect", but neither scale
flipping, so the denominator supplies no information. All three small slots are far below the
$n<50$ threshold of §4.6.2, so this section only reports numbers and draws no conclusions.
Reporting proportions without denominators gives the mistaken impression that "all slots point
the same way and some have stronger effects"; we therefore require the number of flipped cells
in all stratified tables.

## 4.7.3 Relation to differences in slot composition

The slot-composition difference between the two generations (§4.1.5) does not affect the scale conclusions: the comparisons are paired, so slot composition is identical on both sides and cancels in the paired difference. It does affect cross-generation comparisons, so we report no cross-generation numerical comparison.

# 4.8 Limitations

This section lists the boundaries of this paper. The items listed first directly weaken the strength of the claims; the reason they are written in the main text rather than left to be discovered later
is that several of them are themselves part of the methodological conclusions of this paper.

## 4.8.1 Limitations of samples and scale

Only two model families: the scale ladder is built on Qwen3-VL 2B/4B/8B and Qwen2.5-VL 3B/7B,
both from the Qwen series, so we cannot rule out that the conclusions are tied to that series'
training recipe and we make no cross-family generalization. Evaluation under multiple prompts
and conditions has been recommended as the default protocol [39,38], and our two-family,
two-criterion design is the minimal implementation of that recommendation here. The third scale
point, 32B, was empirically falsified: its public weights are AWQ quantized, incomparable with
the bf16 of the other scales, which would reintroduce the confound of §4.3.4 and break the
anchoring protocol's precondition. The effect saturates early (§4.2.3.2), so the scale
conclusions cover only 2B to 8B and cannot be extrapolated to "the effect keeps growing with
scale". Single data source: all experiments rest on one counterfactual-editing dataset; the
preconditions for a second (an OK-VQA subset) were empirically shown not to hold — the 3B
parameter hit rate on it is only 25.5% and the questions are all perceptual with no evidence
field, making conflict conditions impossible to construct. Hence "the conclusions are not
specific to this dataset's format" is something we cannot prove, and this is the most
substantial limitation of this paper.

## 4.8.2 Limitations of criteria and measurement

The main criterion is a string rule, not a semantic judgement; complex or non-English outputs
may be missed, which creates no false positives but underestimates the effect. The absolute
magnitude for `ctx_hedge` is untrustworthy, and on the Qwen2.5 family the two criteria give
opposite signs (alias $-4.2$ pp, $n=24$; negation-aware $+11.6$ pp, $n=584$). The former misses
the $n\geq50$ threshold and by rule takes part in no sign comparison, but the disagreement must
be disclosed: it is a counterexample to "criterion choice can move the sign" and it occurs on
one of our own main-result cells — one that carries none of the paper's claims, since removing
it leaves the sign sequence unchanged and the span drops only from 56.5 to 55.3 pp (§4.2.3.4).
Position preference makes some forced-choice conditions unusable (§4.4.3), and the same risk may
exist in other forced-choice results without order control. The eighth stance's sign (§4.9)
holds only under the negation-aware criterion and its magnitude is sensitive to prompt length;
we claim only the sign.

## 4.8.3 Limitations of the mechanistic explanation and of priority of discovery

The mechanistic explanation is a behavioural-level inference, not a representational probe: we
explain what behaviour arises under what conditions, but not how the model internally represents
the conflict, which would require activation access and representational probes we have not
done. That "instruction stance explains the disagreement between two published works" is an
external inference; we lack their original internal records and cannot verify it. Nor do we
claim priority for either sign direction or for originating "the effect sign can cross zero"
itself (§2.2); what we claim is placing them on the same pre-registered authorisation ladder and
obtaining all signs on the same batch of samples.

# 4.9 The eighth stance `adjudicate`: a pre-registered prediction test

## 4.9.1 Why a prediction test is needed

The preceding sections give descriptive findings: the seven stances' signs occupy the same
positions in both families, the first two negative and the last five positive. A descriptive
finding cannot be distinguished from "these signs were these numbers all along" — as long as
the stances are fixed after seeing the results, any sign structure can be narrated as a
regularity. We therefore performed a falsifiable test before collecting the data: treat "signs
are ordered with authorization strength" as the hypothesis, use the first six stances to fix a
sign rule on the training family, and predict the sign of the eighth stance `adjudicate`, never
measured in this work, with the prediction, decision rule, and numeric interval all fixed in
writing before seeing any output of this stance (`prereg_stance8.md`). Predicting unseen data
guards against selective reporting [43,30]. The eighth stance sits deliberately between
`own_only` and `prior`: it assigns no side priority but asks the model to judge the reliability
of the context and of its own knowledge, testing the authorization order itself rather than the
wording of any one stance.

## 4.9.2 The pre-registered prediction and decision rule

The sign rule is taken from the first six stances: those preceding `neutral` in the
authorization order are predicted negative, the rest positive. The order (weak to strong) is
`ctx_only` < `orig` < `neutral` < `adjudicate` < `own_only` < `len_ctrl` < `prior`;
`ctx_hedge` does not enter it, its sign being unclaimable in the Qwen2.5 family (the two
criteria give opposite signs and the alias denominator misses the $n\geq50$ threshold), so it
is retained as an outlier diagnostic arm. The decision rule was fixed before seeing the data:
both families positive is a successful prediction, both negative a failure requiring the core
claim to be downgraded, and one of each a partial failure; when the sign is positive but the
magnitude falls outside the predicted interval, the two are reported separately.

## 4.9.3 Measured results

The anchoring set follows $D^*$ of the main experiment, $|D^*|=649$, and is not recomputed
(the eighth stance does not change the anchoring protocol); the criterion is the negation-aware
one, as elsewhere. Table~\ref{tab:adj} gives the four arms side by side under the two criteria;
the decision and claimability vary with the criterion, so the two must be read together
(§4.9.5 explains the degradation under the alias criterion).

///[adj] Measured results of the prediction test for the eighth stance under the two criteria
| Family | Scale pair | Arm | Criterion | $\Delta$ (pp) | $p$ | $n$ | Claim |
|---|---|---|---|---|---|---|---|
| Qwen3-VL | 2B → 8B | `adjudicate` | neg.-aware | $+46.1$ | $2.1\times10^{-50}$ | 360 | Yes |
| Qwen3-VL | 2B → 8B | `adjudicate_len` | neg.-aware | $+45.6$ | $1.1\times10^{-47}$ | 344 | Yes |
| Qwen2.5-VL | 3B → 7B | `adjudicate` | neg.-aware | $+15.7$ | $9.2\times10^{-15}$ | 485 | Yes |
| Qwen2.5-VL | 3B → 7B | `adjudicate_len` | neg.-aware | $+29.0$ | $4.1\times10^{-33}$ | 465 | Yes |
| Qwen3-VL | 2B → 8B | `adjudicate` | pure alias | $+2.4$ | $1.00$ | 41 | No, $n<50$ |
| Qwen3-VL | 2B → 8B | `adjudicate_len` | pure alias | $+0.0$ | $1.00$ | 29 | No, $n<50$ |
| Qwen2.5-VL | 3B → 7B | `adjudicate` | pure alias | $+2.3$ | $0.54$ | 175 | No, $p>0.05$ |
| Qwen2.5-VL | 3B → 7B | `adjudicate_len` | pure alias | $+8.0$ | $2.3\times10^{-2}$ | 150 | Weak |

The sign prediction holds: in both families $\Delta$ is positive and highly significant, so by
the pre-registered rule the core claim for this stance is upgraded from descriptive to a
predictable regularity — the authorization order predicts the sign of a previously unmeasured
stance. The magnitude prediction fails: the pre-registered intervals are $[+12,+28]$ for the
Qwen3 family and $[+22,+34]$ for the Qwen2.5 family, and both measured values fall outside, in
opposite directions ($+46.1$ above the upper bound, $+15.7$ below the lower). The hypothesis
that `adjudicate` should fall between `own_only` and `prior` does not hold: the Qwen3 value
exceeds that family's highest existing stance, `prior` ($+28.0$).

The two families fall at the two ends of the ladder rather than in the same segment, the most
substantial result of this section. It sharpens the cross-family heterogeneity of §4.7.1: the
same stance differs between families not merely in magnitude but in its position relative to
the existing stances. The authorization order can therefore predict the sign, but not the
relative position of the sign across families.

No saturation artifact is observed: across the five scale points the parameter-knowledge
proportion for `adjudicate` is 0.382, 0.512, 0.550 (Qwen3-VL 2B/4B/8B) and 0.493, 0.683
(Qwen2.5-VL 3B/7B), rising monotonically with scale, so $+46.1$ is an ordered effect rather
than a ceiling artifact. The paired flips in the Qwen3 family are one-directional
($\uparrow$ 0, $\downarrow$ 166).

## 4.9.4 The length-control arm

`adjudicate_len` appends "Take your time to think." after the same prompt, growing the
instruction from 37 to 42 words; its pre-registered purpose is to rule out length as an
alternative explanation. The two families have the same sign, so length does not drive the
sign — supporting the main conclusion — but it clearly drives the magnitude, and in opposite
ways: the Qwen2.5 family moves from $+15.7$ to $+29.0$ ($+13.4$ pp, inside the predicted
interval) while the Qwen3 family barely moves ($-0.5$ pp). Differing by one sentence of prompt,
the two families yield results nearly 14 pp apart, showing that the magnitude for this stance
is sensitive to prompt length — itself another instance of a measurement protocol affecting
conclusions.

## 4.9.5 The same test under the alias criterion

Under the rule of §4.6 the same test is recomputed under the alias criterion, in the lower half
of Table~\ref{tab:adj} (the last four rows); this reading is clearly weaker, and we list it
faithfully rather than hiding it. The two Qwen3 cells' decidable sample counts are 41 and 29,
both below the $n\ge50$ threshold, so by §4.6 they take part in no sign decision; the Qwen2.5
family's main arm has $n=175$, passing the threshold, but $p=0.54$, likewise constituting no
claim. The collapse has the same cause as in §4.2.1.1: the alias criterion judges an explicit
rejection naming both values at once as undecidable, and `adjudicate` precisely induces such
rejections (the alias decision rate of Qwen3 8B is only 0.073 against 0.894 under
negation-aware). The sign claim therefore holds only under the negation-aware criterion; we take
that as the reporting protocol because the criterion choice was fixed under §4.6 before seeing
this stance's data.

# 4.10 The common structure of the three measurement biases

The three biases are instances of one class of failure. In the criterion defect (§4.2.1), alias
matching judges an explicit rejection that names two values at once as undecidable and removes
it, yet such answers are the most explicit class of following parametric knowledge, and the
removed amount rises with scale. In the denominator collapse (§4.6.2), when paired decidable
samples fall to single digits $\Delta$ can only take a degenerate value, yet in the table it
looks like the strongest cell. In the position-preference self-refutation (§4.4.3), rerunning
with the option contents swapped leaves the output almost unchanged while the letter preference
flips, showing the quantity measured is not a content judgement. What the three share is that
the quantity measured and the quantity reported are not the same, with the bias direction
systematic in each; the two corrections (the negation-aware criterion and position control) both
point the same way — first confirm what is being measured, then report the number measured.
Most numbers in knowledge-conflict evaluation come from a default protocol, and several of its
internal decisions — how explicit rejections are counted, how far the denominator collapses,
how the options are ordered — can each move the sign.
