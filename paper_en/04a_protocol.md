# 4.1 Dataset and filtering

## 4.1.1 From E-VQA to two generations of anchoring sets

Our evaluation set is built on E-VQA, whose questions refer to their entity through a
referring phrase ("this plant", "this animal"), so the entity's identity can be determined only
from the image. That is an advantage in the original task but an obstacle for our quantity:
feed the referring expression verbatim and what the model answers is the linguistic prior of
that question type, not knowledge about the species. The symptom surfaced in the first round of
filtering: the 6 samples flagged "strongly anchored" all fall on one template — `In which season
does this plant give flowers?` is always answered `Spring` across 6 different species. What the
model gets right is the common-sense distribution of flowering time, independent of whether it
recognises the species. The remedy was to replace the referring expression with the species
name and re-run the filtering, which lowers the hit rate but buys a reliable measurement of
"the model holds this fact" rather than "the model is familiar with this sentence pattern".
Recent multimodal knowledge-conflict benchmarks likewise centre the question on the entity [33],
the conflict lying in the inconsistency between the model's parametric memory of that entity
and the context. The two generations of anchoring sets, whose sizes and filtering criteria are
in Table S11 (online material), barely overlap: of the 927, only 147 belong to the 236.
The first generation got the editing pipeline working and the second is the object of study
here; keeping them apart matters because the first was filtered by a single model and carries a
selection bias — facts known by the 3B model skew towards high-frequency, templated species
attributes, exactly the ones a large model can answer without retrieval, so comparing scales on
such a set absorbs the difference into the filtering criterion.

## 4.1.2 Construction of the second-generation set

The second-generation set is built in three steps: take the entries with evidence sentences
from the E-VQA candidate pool, re-ask them with the entity-name phrasing, have Qwen2.5-VL 3B and
7B each produce one context-free answer, and keep the samples for which both produce the
parametric ground truth. Two scales rather than one are used because our dependent variable is
the behavioural difference across scales: filtering with a single model would select samples
necessarily "known" to it while leaving untested whether the other scale knows them too, and
only when both are confirmed to know them is any later difference guaranteed not to be the
trivial effect of "a larger model knows a few more facts". Each sample then undergoes
counterfactual editing; 1146 enter the valid pool, of which 927 pass quality control and
constitute the main set. The other 219 fail rewriting — the edited text is semantically
self-contradictory, the number format is malformed, or the edit does not actually change the
value — and have an empty `context_conflict` field; at evaluation time they are fed a context
consistent with parametric knowledge, so they can never exhibit conflict, and leaving them in
the denominator dilutes PKD with zeros.

## 4.1.3 Counterfactual editing pipeline

The carrier of the edit is a single slot in the evidence sentence; after editing, the context
conflicts with the parametric ground truth while everything else stays verbatim. Single-slot
substitution is necessary: it makes the conflict location precisely identifiable, so that "did
the model notice the conflict" and "on which semantic dimension did the model concede" can be
examined separately. Rewriting a whole passage cannot do this. Conflict injection is cognate
with existing knowledge-conflict benchmarks [19,21]; the only difference is granularity —
baseline work usually injects a whole conflicting evidence passage, whereas we touch a single
slot.

The pipeline handled four classes of failure. The disposition of the four classes is listed in
Table S1 (online material).

The final quality-control result is malformed numbers 0, duplicated punctuation 0, edits that
failed to take effect 0.

## 4.1.4 Image mapping

All 236 anchoring samples are matched with iNaturalist 2021 images; the mapping chain has three
segments:

```
wikipedia_title --(scientific name match)--> val.json.categories.name
    -> categories.image_dir_name -> images.file_name
```

The scientific-name match rate of the first segment is measured at 100%. Two error-prone
points are recorded below. First, `dataset_category_id` must be zero-padded to 5 digits, or the
directory-name match fails. Second, `dataset_image_ids` stores the original iNat photo numbers,
which are not among the 100,000 published for the competition and cannot be used to locate
files; one must go through the `image_dir_name` and `file_name` fields.

## 4.1.5 Slot composition

The coarse-grained slot distribution of the 927-sample main set is place 512, other 335,
quantity 33, diet 28, time 19; place accounts for 55.2%, so the conclusions rest mainly on
geographical facts. At the fine-grained level the top `edit_slot` categories are place 507,
city 281, quantity 44. One reading caution: the quantity slot has only 33 entries here against
66 in the first-generation 236 set, its share dropping from 28.0% to 3.6%, so the two
generations' slot compositions differ considerably and we draw no per-slot scale-effect
conclusion, reporting it only alongside the domain-wise tests of §4.7 with each group's sample
size. The scale comparison is performed not on these 927 but on their subset $D^*$ (§4.2.5).

# 4.2 Main result: the sign of the scale effect is decided by the instruction stance

## 4.2.1 Choice of criterion

Before reporting any scale difference we must fix what counts as "the model followed
parametric knowledge". Our dependent variable is a binary choice behaviour requiring a
criterion that decides between PARAM and CTX; two candidates exist, and they do not affect the
conclusion identically, so the choice needs justification.

The first criterion, `distinctive_ground`, requires all content words of the answer to fall
within a single evidence source. Its appeal is strictness: only a model that can actually
produce that proper name counts as following the parameter. But the longer the answer, the
harder this is to satisfy, so the probability of being judged "undecidable" rises with length.
The fragility of string criteria to paraphrase is documented in the evaluation literature and
has been shown to be enough to reverse the entire ordering of methods [36,38]; the length
covariation here is a concrete form of the same problem.

The mean answer lengths of the seven stances differ by nearly 20-fold (`orig` 1.5 words,
`ctx_hedge` 30.6 words), and the undecidable rate under the original criterion tracks the mean
length almost exactly (Table~\ref{tab:len-undecidable}):

///[len-undecidable] Mean answer length and undecidable rate under the original criterion, by instruction stance
| Stance | 3B mean length (words) | Undecidable rate under the original criterion |
|---|---|---|
| `orig` | 1.5 | 20.7% |
| `ctx_only` | 2.2 | 22.5% |
| `neutral` | 3.7 | 30.0% |
| `own_only` | 6.0 | 33.9% |
| `prior` | 17.9 | 54.3% |
| `len_ctrl` | 25.8 | 68.3% |
| `ctx_hedge` | 30.6 | 94.4% |

The Pearson correlation across the seven points is $r = 0.983$, the Spearman rank correlation
$\rho = 1.000$. Table~\ref{tab:len-undecidable} is taken from Qwen2.5-VL 3B on the common
coverage set. Recomputing on the anchoring set $D^*$ gives $r = 0.986$, $\rho = 1.000$, and
the conclusion is unchanged; switching to character units also gives $r = 0.983$,
$\rho = 1.000$. This covariation is insensitive to the measurement convention. The correlation is between
seven stance-level aggregate points, which is the level at which we compare: the difference
between stances is precisely a length difference.

This criterion therefore largely measures "how long the answer is" rather than "whose side
the model takes": the undecidable rate of `ctx_hedge` under it reaches 94.4%, while most of
those are decidable under the alias criterion.

The second criterion matches against two alias tables, `parametric_aliases` and
`context_answer_aliases` (`common.answers_match`: bidirectional containment plus content-word
coverage). It is length-insensitive, and the alias tables are manually verified, independent of
the models evaluated. All main results use it; the comparison with `distinctive_ground` is in
§4.6.

### 4.2.1.1 Defects of the alias criterion itself, and the second criterion

Bidirectional containment introduces a second, subtler problem. Consider a typical answer:

> The context is incorrect. Porrentruy Castle is not located in Peru.
> It is actually situated in Switzerland.

This answer hits both the context alias Peru and the parametric alias Switzerland and is
judged BOTH — "undecidable" — then removed entirely from the PKD denominator. Yet its stance
is unambiguous: it rejects the context and gives the parametric ground truth. It should be the
strongest evidence for PARAM; the very reason it is removed is that to reject the context it
had to name the wrong value.

The defect has a definite direction: the stronger the model, the more it rejects explicitly
rather than silently taking a side, so the amount removed rises with scale. On $D^*$ for the
`prior` stance the share of "explicit rejections among BOTH" is Qwen3-VL 2B 19.0%, 4B 88.8%,
8B 75.7%, and Qwen2.5-VL 3B 53.1%, 7B 85.8%; the total number of rescues across the seven
stances is Qwen3 166 → 956 → 1042 and Qwen2.5 600 → 1257, both rising monotonically with
scale. The removal is thus biased against large models and systematically underestimates
$\text{PKD}_{s_{\max}}$, pushing down our core positive effect.

We therefore introduce a second criterion: for answers judged BOTH, detect whether they
explicitly reject the context and, if so, count them as PARAM. The detection uses a
pre-registered table of string patterns in two classes, with the fence-sitting cues
(`could be either`, `uncertain`, `both are possible`, `ambiguous`) never rescued once they
appear, so as to prefer missing rescues. The pattern table, its disposition (Table S12), and
the two-sided spot-check record (19 rescued answers judged PARAM; 0 of 19 retained falsely
rescued) are in the online material.

We report both criteria without selective presentation: Table~\ref{tab:claim-criteria}
places all claims of the two side by side, marking the one that is inconsistent.

///[claim-criteria] Claims under the alias-only criterion versus the negation-aware criterion
| Claim | Alias-only criterion | Negation-aware criterion |
|---|---|---|
| Number of significant conditions, qwen3 family | 6/7 | 6/7 |
| Number of significant conditions, qwen25 family | 6/7 | 7/7 |
| Span, qwen3 family | 45.9 pp | 56.5 pp |
| Span, qwen25 family | 37.5 pp | 47.5 pp |
| Sign ordered (all negatives before positives) | Holds for both families | Holds for both families |
| $\Delta$ crosses zero | Holds | Holds |
| Magnitudes not strictly monotone | Holds | Holds |
| Cell-wise consistent sign ($n\geq50$ convention) | Holds | Holds |

The correction
makes the effect sizes larger, widening the span by 10.6 pp and 10.0 pp in the two families,
with the sign structure unaffected. The two criteria disagree in sign in exactly one cell,
`ctx_hedge` of the Qwen2.5 family ($-4.2$ pp, $n=24$, alias; $+11.6$ pp, $n=584$,
negation-aware); it misses the $n\geq50$ threshold of §4.6 and by rule does not enter the sign
comparison, and in all cells that pass, the two criteria agree cell by cell. The denominator gap
(24 versus 584) is itself an instance of our argument: the alias criterion removes answers that
must name the wrong value in order to reject the context, and `ctx_hedge` most strongly induces
rejection, so it suffers most (§4.6.2, §4.8.2). The criterion is a string rule, not a semantic
decision; complex or ironic phrasing may be missed, but a miss only falls back to the alias-only
criterion and cannot manufacture false positives, because a rescue requires an explicit pattern
match. The spot-check records for both blind directions are in the online material.

## 4.2.2 Stance ladder

The sign of the scale effect depends on where the instruction places parametric knowledge. We
set seven stances and pre-register their ordering from weak to strong authorisation —
pre-registered meaning the ordering is fixed before the data are seen, not derived from a post
hoc fit. Conclusions under a single prompt often cannot represent behaviour over the prompt
space [39,38], and authorisation strength is a prompt dimension not previously scanned on its
own. The wording and authorisation strength are in Table S13 (online material; verbatim
prompts in Table~\ref{tab:stance-ladder}). The position of `len_ctrl` deserves separate
comment: it separates "authorise parametric knowledge" from "require a long answer", and is our
main design for countering the length confound. Were the scale effect driven purely by answer
length, `len_ctrl` should give results close to `prior`; empirically it does not (§4.2.4).

## 4.2.3 Cross-scale results

For each stance ι and scale s, the PKD rate is defined as

$$\text{PKD}_s(\iota) = \frac{\left|\{i \in D^* : \text{grade}(y_s(i,\iota)) = \text{PARAM}\}\right|}{\left|\{i \in D^* : \text{grade}(y_s(i,\iota)) \in \{\text{PARAM}, \text{CTX}\}\}\right|}$$

The denominator counts only decidable samples; BOTH and NONE are removed. The scale effect is
defined as the difference between the largest and smallest scale within a family:

$$\Delta_s(\iota) = \text{PKD}_{s_{\max}}(\iota) - \text{PKD}_{s_{\min}}(\iota)$$

### 4.2.3.1 Measured results (5 scale points, two model families)
The number of scale points expanded from 2 to 5: Qwen3-VL forms the 2B→4B→8B doubling
ladder ($|D^*| = 649$) and Qwen2.5-VL a second family, 3B→7B. Table~\ref{tab:pkd-both} gives
the endpoint comparison for the two families under the negation-aware criterion.

///[pkd-both] PKD endpoint comparison by stance for the two model families (negation-aware criterion)
| Stance | Qwen3 2B | Qwen3 8B | $\Delta_{\text{q3}}$ (pp) | $p$ |
|---|---|---|---|---|
| `ctx_only` | 0.324 | 0.050 | $-27.3$ | $2.9\times10^{-38}$ |
| `orig` | 0.382 | 0.145 | $-23.7$ | $2.6\times10^{-29}$ |
| `ctx_hedge` | 0.687 | 0.979 | $+29.2$ | $4.5\times10^{-49}$ |
| `neutral` | 0.549 | 0.564 | $+1.5$ | $0.491$ |
| `own_only` | 0.621 | 0.742 | $+12.1$ | $4.9\times10^{-10}$ |
| `len_ctrl` | 0.548 | 0.762 | $+21.4$ | $1.9\times10^{-26}$ |
| `prior` | 0.631 | 0.911 | $+28.0$ | $8.3\times10^{-45}$ |

///[pkd-both-q25] PKD endpoint comparison by stance, Qwen2.5-VL (negation-aware criterion)
| Stance | Qwen2.5 3B | Qwen2.5 7B | $\Delta_{\text{q25}}$ (pp) | $p$ |
|---|---|---|---|---|
| `ctx_only` | 0.244 | 0.160 | $-8.4$ | $2.5\times10^{-6}$ |
| `orig` | 0.347 | 0.209 | $-13.8$ | $4.5\times10^{-13}$ |
| `ctx_hedge` | 0.846 | 0.962 | $+11.6$ | $1.1\times10^{-18}$ |
| `neutral` | 0.457 | 0.615 | $+15.9$ | $8.7\times10^{-15}$ |
| `own_only` | 0.449 | 0.666 | $+21.7$ | $4.4\times10^{-25}$ |
| `len_ctrl` | 0.572 | 0.694 | $+12.3$ | $2.5\times10^{-10}$ |
| `prior` | 0.552 | 0.890 | $+33.7$ | $3.5\times10^{-50}$ |

The two families independently give an isomorphic conclusion: the two weakest-authorisation
stances have $\Delta < 0$, the two strongest have $\Delta > 0$, and $\Delta$ crosses zero —
two different families, different parameter ranges, run independently, with the same signs.

### 4.2.3.2 The effect does not grow smoothly with scale but saturates early

The decomposition into adjacent pairs yields an important qualification. For the Qwen3
family, on the three strongest-authorisation stances the 2B→4B segment already achieves almost
the entire effect and the 4B→8B segment is essentially flat (`ctx_hedge` $+2.5$,
`len_ctrl` $-0.6$, `prior` $-1.0$, the latter two not significant): the scale effect saturates
early within our range and does not grow smoothly with scale. The shape of the curve depends
on the measure — the same quantity can appear as a continuous climb or a discontinuous jump
depending on difficulty and scoring convention [28,29] — and we measure the sign of a binary
decision, so the saturation point may precede that of the capability itself. The correct
statement is: the sign is decided by the stance, the magnitude saturates early. The two
families differ in shape: the negative side of the Qwen3 family does not saturate (`ctx_only`
$-12.9$ then $-14.7$), whereas the Qwen2.5 family is near saturation at 7B. The adjacent-pair
values are in Table S2 (online material).

### 4.2.3.3 "Sign ordering" and "monotone magnitudes" are two different things and must not be conflated

The two claims have different evidential strength and must be split. Weak claim (supported):
the sign of $\Delta$ is ordered with authorisation strength, all negative signs preceding all
positive signs with no interleaving. Strong claim (not supported): the magnitude rises
monotonically with authorisation strength. Qwen2.5-VL 3B→7B has two adjacent inversions
(`ctx_only` $-8.4 \ge$ `orig` $-13.8$, `own_only` $+21.7 \ge$ `len_ctrl` $+12.3$),
$\rho = 0.857$ ($p = 0.012$); Qwen3-VL 2B→8B has one (`ctx_hedge` $+29.2 \ge$ `neutral`
$+1.5$), $\rho = 0.643$ ($p = 0.070$). The trend is significant but far from monotone, so we
claim only the weak claim, which holds under both criteria (paired convention, spans
$45.9$/$37.5$ pp).

### 4.2.3.4 Sensitivity check excluding `ctx_hedge`

`ctx_hedge` is the only one of the seven cells whose magnitude is highly unstable across
criteria (the two criteria differ by 60.1 pp here), so it is worth testing what remains when it
is removed entirely. The weak claim not only holds but becomes stronger. The rank correlations
and spans before and after exclusion are in Table S3 (online material). The sign
sequences of the two families are identical after exclusion — still the first two negative and
the last four positive, still crossing zero — and the span is nearly unchanged (Qwen3 drops only
from $56.5$ to $55.3$ pp); the Qwen3 family becomes strictly monotone ($\rho = 1.000$), so its
low rank correlation with seven cells is caused entirely by `ctx_hedge`. Two implications
follow: the least credible cell is not load-bearing — remove it and the conclusion is
strengthened, not weakened — and `ctx_hedge`'s absolute magnitude overlaps heavily with the
negation criterion's pattern table, dragging the Qwen3 family's rank correlation from $1.000$
down to $0.643$. We therefore keep `ctx_hedge` in the table for the complete seven-level scan
but rely on it to support no claim. The Qwen2.5 family's $\rho$ drops from $0.857$ to $0.771$
after exclusion, its monotonicity still limited by the adjacent inversion `own_only` $(+21.7) \ge$
`len_ctrl` $(+12.3)$, which excluding `ctx_hedge` cannot fix; the two families' rank
correlations go one up and one down, so our weak claim remains limited to sign ordering and we
do not claim monotone magnitudes.

### 4.2.3.5 `len_ctrl`: authorisation decides the sign, length decides the magnitude
Among the three inversions above, `own_only ≥ len_ctrl` is the most informative, being a
built-in control: both stances' authorisation wording requires parametric knowledge to take
precedence, and only `len_ctrl` additionally constrains answer length. On Qwen2.5-VL 3B→7B it
lowers the length by 35 characters while pushing $\Delta$ from $+21.7$ down to $+12.3$. The
determinants of sign and of magnitude are in Table S14 (online material). This is
complementary to §4.2.4's refutation of $H_{\text{len}}$, not contradictory: length cannot
decide the sign, but it can decide the magnitude — hence the paper limits its title-level
assertion to the sign and attributes the magnitude explicitly to length, a known confound.

## 4.2.4 Ruling out the length confound

A natural alternative explanation is that large models give longer answers, and longer answers
have more opportunities to hit a parametric entity word, so that "following parametric
knowledge more" is really length at work. Denote this $H_{\text{len}}$. It has a directly
decidable corollary: if it holds, the sign of $\Delta_s$ should follow the sign of the
large-minus-small answer-length difference,

$$\text{sgn}\left(\overline{|y_{s_{\max}}|} - \overline{|y_{s_{\min}}|}\right) = \text{sgn}\,\Delta_s(\iota), \quad \forall \iota$$

Any cell with "length grows while $\Delta < 0$" or "length shrinks while $\Delta > 0$" is a
counterexample. Table~\ref{tab:hlen-both} gives the decision for each of the fourteen cells of
the two families, with $\Delta$ from the negation-aware criterion (cognate with §4.2.3): the
length difference, $\Delta$, the decision rate, and the predicted versus measured sign of
$H_{\text{len}}$; cells where the two disagree are counterexamples.

///[hlen-both] Length-confound decision table for the two families (negation-aware criterion)
| Family | Stance | Length diff (chars) | $\Delta$ (pp) | Decision rate | $H_{\text{len}}$ pred. | emp. |
|---|---|---|---|---|---|---|
| Qwen3 | `ctx_only` | $-7.7$ | $-27.3$ | 91.8% | negative | negative |
| Qwen3 | `orig` | $-2.3$ | $-23.7$ | 93.7% | negative | negative |
| Qwen3 | `ctx_hedge` | $+68.9$ | $+29.2$ | 88.1% | positive | positive |
| Qwen3 | `neutral` | $-14.8$ | $+1.5$ | 92.9% | negative | **positive** |
| Qwen3 | `own_only` | $-4.7$ | $+12.1$ | 90.8% | negative | **positive** |
| Qwen3 | `len_ctrl` | $+83.1$ | $+21.4$ | 81.5% | positive | positive |
| Qwen3 | `prior` | $+48.2$ | $+28.0$ | 88.6% | positive | positive |
| Qwen2.5 | `ctx_only` | $+15.0$ | $-8.4$ | 86.4% | positive | **negative** |
| Qwen2.5 | `orig` | $+2.3$ | $-13.8$ | 91.4% | positive | **negative** |
| Qwen2.5 | `ctx_hedge` | $-35.5$ | $+11.6$ | 90.0% | negative | **positive** |
| Qwen2.5 | `neutral` | $+37.6$ | $+15.9$ | 92.1% | positive | positive |
| Qwen2.5 | `own_only` | $+32.4$ | $+21.7$ | 90.0% | positive | positive |
| Qwen2.5 | `len_ctrl` | $-22.6$ | $+12.3$ | 81.7% | negative | **positive** |
| Qwen2.5 | `prior` | $+42.7$ | $+33.7$ | 83.7% | positive | positive |


Six of the fourteen cells have opposite signs, refuting $H_{\text{len}}$. The sharpest are
`ctx_only` and `orig` of the Qwen2.5 family: the large model gives longer answers yet follows
parametric knowledge less. And `len_ctrl`, designed specifically to separate authorisation
from length, does lower length by about 23 characters while $\Delta$ remains positive. Under
the alias-only criterion there are more opposite-sign cells (eight of fourteen), because it
flips the length differences of the `ctx_hedge`, `own_only`, and `len_ctrl` cells to negative —
all three with denominator collapse (decision rate $13\%$–$40\%$), which records "the length of
the removed part" as a negative difference and is not comparable, for the same reason as in
§4.2.3.3. $H_{\text{len}}$ is refuted under both criteria, in the same direction. Stance does
change length (Kruskal-Wallis $p < 10^{-160}$ on every scale point,
$H = 1764.5 / 1915.3 / 1495.4 / 2344.7 / 2461.0$, $df = 6$), so length is not a constant and $H_{\text{len}}$ is not
untestable; it is simply not sufficient to decide the sign.

### 4.2.4.1 A corollary: the decision rate must be reported alongside $\Delta$

The defects of our criterion first surfaced in the decision rate. `ctx_hedge` at 3B has a mean
length of 188 characters, 20 times that of the adjacent stance `orig`, because that prompt
makes the model give the parametric and the context answer at once; under the alias-only
criterion such answers fall into BOTH and are removed, so its decision rate is as low as
6.0%–66.3% while other stances are mostly above 80%. The decision rate must therefore be given
alongside $\Delta$: reporting only $\Delta$ and $p$ places a weakly evidenced condition beside
a strongly evidenced one, and every table here that reports $\Delta$ includes a decision-rate
column. The negation-aware criterion lifts this column across the board (`ctx_hedge` from 6.0%
to above 90%, the whole table stabilising at 85%–97%), direct evidence that it repairs the
denominator collapse — the low decision rate is not "the model cannot answer" but "the model
rejected and was removed" — without removing the reporting standard.

## 4.2.5 Shrinkage of the anchoring protocol

The 927-sample main set was itself filtered by the intersection of the knowledge of Qwen2.5-VL
3B and 7B, so for these two $K_s(i)=1$ holds almost by construction; the newly added Qwen3-VL
family needs independent testing. The signal is unambiguous: Qwen3-VL 2B produces the
parametric ground truth under NO-CTX only 0.785 of the time, against 0.992/0.997 for Qwen2.5-VL
3B/7B. This must be handled, or the comparison is contaminated by a trivial effect: if the
small model does not know a fact, "not following parametric knowledge" is not a choice but
having nothing to follow, and leaving it in the denominator would systematically favour the
small model. Note this is opposite in direction to the criterion defect of §4.2.1.1 — that
pushes down large models, this pushes down small ones, and the presence of both does not mean
they cancel.

The anchoring protocol takes the samples that all scale points answer correctly:

$$D^* = \bigcap_{s} \{ i : K_s(i) = 1 \}$$

where $K_s(i)$ is measured on the NO-CTX arm alone, independent of the conflict condition.
$D^*$ shrinks monotonically as scale points are added: 720 for the earliest three scale points
(Qwen3-VL 2B, Qwen2.5-VL 3B/7B), 672 after adding Qwen3-VL 4B, and 649 when all five scale
points are present, 70.0% of the 927 common-coverage samples. The amount removed for each model
relative to these 927 is in Table S4 (online material).

The amount removed falls monotonically with scale, consistent with the knowledge rate rising
with scale (relative to the common coverage of 927, not to each model's own knowledge set).
$D^*$ shrinks from 927 to 649, a loss of 30%, but 649 still supports every test here (the
per-cell decidable samples are above 550), so the scale conclusion remains confirmatory. The
main cost falls on Qwen3-VL 2B, 21.7% of whose samples are removed: its PKD is measured on the
subset it genuinely knows, so the remaining difference can be attributed to scale and stance
rather than to how much is known.

# 4.3 Measurement protocol and implementation details

Every conclusion in this chapter rests on one kind of measurement: give the model a context
that conflicts with its own knowledge and see whom it follows. Such measurements are unusually
sensitive to implementation details, and our pilot experiments found three choices that make
the same batch of samples yield completely different results. We state them before the results
because every number that follows depends on them, and because these errors are widespread in
published work: a reader who does not reproduce our protocol choices cannot obtain comparable
numbers.

## 4.3.1 Three measurement conventions

### Convention one: the context-free arm must use a neutral system prompt

Measuring "how much the model knows on its own" requires withholding the context, but the
system prompt still affects the measurement: reusing the context arm's prompt tells the model,
under a condition where it should rely on its own knowledge, "not to rely too much on yourself",
and on 236 samples the parametric knowledge rate is 0.496 with that prompt against 0.9746 with a
neutral one — nearly 48 pp, enough to reverse the basic judgement of "whether the model knows".
The context-free arm therefore uses a separate neutral prompt,
`You are a knowledge assistant. Answer with a short phrase.`; the seven stances of the context
arm each have their own wording (§4.3.2).

### Convention two: the knowledge rate is read out only on the text-only arm

The image arm measures "how much parametric knowledge remains after the image intervenes",
which is not an unbiased estimate of knowledge held; filtering samples with it would mix visual
interference into the selection. Our anchoring set $D^*$ is therefore constructed entirely from
the text-only arm; the visual arm is used only for the comparison in §4.6.

### Convention three: the effective denominator of the conflict condition must exclude samples whose rewriting failed

Convention three is as in §4.1.2: a sample whose rewriting failed has a context consistent
with parametric knowledge and can never exhibit conflict, so counting it would dilute PKD with
zeros; the effective denominator is 927 after the 219 failed rewrites are removed.

## 4.3.2 The seven instruction stances

The seven stances change only the wording of the system and user templates; the samples, the
context, the image, and the decoding parameters are completely unchanged.
Table~\ref{tab:stance-ladder} gives the verbatim definitions and the authorisation strength.
Authorisation strength is the degree of authorisation of parametric knowledge, pre-registered
from weak to strong; the ordering is fixed before the data are seen.

///[stance-ladder] Verbatim definitions of the seven instruction stances and their authorisation strength
| Stance | system prompt | user template |
|---|---|---|
| `ctx_only` | `...that relies solely on the provided context.` | `Answer briefly using only the context:` |
| `orig` | `...using the provided context when relevant.` | `Answer briefly based on the context and the image:` |
| `ctx_hedge` | `...if you believe the context is inaccurate, answer from your own knowledge.` | `Answer briefly, correcting the context if it is wrong:` |
| `neutral` | `You are a question answering system.` | `Answer briefly:` |
| `own_only` | `...that relies solely on your own knowledge.` | `Answer briefly using only your own knowledge:` |
| `len_ctrl` | `Use the provided context and your own knowledge together, and answer as appropriate.` | `Answer briefly using the context and your own knowledge as appropriate:` |
| `prior` | `Use your own knowledge when it is more reliable than the provided context.` | `Answer briefly, preferring your own knowledge if it is more reliable:` |

Three stances need their rationale explained. `orig` copies the reproduced work's prompt
verbatim, including `and the image`; the wording looks superfluous on the text-only arm, but
changing it would mean it is no longer a reproduction, and keeping it in the same run rather
than taking numbers from another run eliminates batch differences — the seven rows of the main
table come from one decoding run, with the same server-side state and batch effects.
`ctx_hedge` covers the intermediate state real retrieval systems use: the first four stances
sit between "instruct to use the context" and "permit use of one's own knowledge", and a
natural objection is that the gradient is manufactured by pushing to the extremes, so
`ctx_hedge` tests the intermediate state directly and provides the gradient's zero-crossing
point. `len_ctrl` is the length-control placebo: the first four stances differ in word count
(`ctx_only` 13, `neutral` 7, `prior` 17, `own_only` 14), so the gradient covaries with length,
whereas `len_ctrl` equals `prior` in length and is isomorphic in structure (17 words, `Use X
when Y`) but semantically neutral, with `as appropriate` granting precedence to neither side.
Had it fallen near `neutral` rather than `prior`, the gradient would come from semantics rather
than length; measured empirically it falls between the two (§4.2.3.5).

## 4.3.3 Decoding and evaluation parameters

The decoding and evaluation parameters are in Table S5 (online material).

Temperature is fixed at zero for reproducibility, at the cost of being unable to report the
variance across runs. We substitute paired testing: the same batch of samples receives the seven
stances within the same run, the comparisons between stances are a paired design, and the
model's own sampling noise cancels in the paired differences.
Zero temperature does not guarantee deterministic output; the non-determinism introduced by
inference frameworks and batching has been documented empirically [41,42]; the paired design is
our response to this, but it cancels noise within a run, not drift between runs. The concurrency
settings and the service-readiness probe are in the reproduction material.

## 4.3.4 Quantisation: one confound that must be disclosed

All main experiments use bf16 precision. The first-generation results did not: 3B used fp16 and
7B AWQ quantisation, a mixed-precision comparison whose "parametric knowledge following rate
decreases with scale" was a quantisation artefact rather than a scale effect. Quantisation
damages precisely the long-tail fact recall this research measures: the same Qwen2.5-VL-3B has
a parametric knowledge rate of 0.9746 under bf16 but 0.6864 under AWQ (a loss of 28.8 pp), and
the unquantised 3B knows far more than the quantised 7B (0.9746 versus 0.6017, 37.3 pp). Our
second methodological conclusion is therefore that cross-precision scale comparisons are
unusable here. This is also why we abandoned Qwen2.5-VL-32B: its public weights are AWQ
quantised, not comparable with the bf16 of 3B/7B, so we base the scale ladder on the Qwen3-VL
family, whose 2B/4B/8B points all have bf16 weights.

## 4.3.5 Judge

The annotations of the reproduced dataset are long descriptive phrases (median 4 words,
maximum 116). Deciding by substring containment systematically errs — when the model answers
`Spring` and the annotation is `late winter to spring`, the model is right but recorded as
wrong — so we use bidirectional containment plus content-word coverage tiers, which passes all
22 regression cases. The judge's three-tier definitions are in Table S6 (online material); in the main experiment the `letter` tier accounts for 1.000 at 3B and 0.984 at 7B,
the rest `text`, with `none` below 1%. The free-generation arm has no option letters and relies
entirely on alias matching; its criterion defect and correction are in §4.2.1.

## 4.3.6 A note on baselines

This paper is an analytical work, and the meaning of baseline differs from that in a
conventional methods paper. Table~\ref{tab:baseline-protocols} lists comparisons of measurement protocols of the
same kind, used to show that our conclusions are not the product of one particular
measurement convention rather than as "compared methods".

///[baseline-protocols] Comparison of measurement protocols of the same kind and their role in this paper
| Protocol | Source | Role in this paper |
|---|---|---|
| Free generation with instruction-style prompts | Reproduced work | Reproduces the known negative sign |
| `ctx_only` | Our stance scan | Negative end |
| `neutral` | Our stance scan | Near the zero crossing |
| `prior` / `own_only` | Our stance scan | Positive end |
| `len_ctrl` | Added in this paper | Rules out the length cause |
| `ctx_hedge` | Added in this paper | Covers the deployment intermediate state |
| ConflictBank template | Existing work | Cross-framework comparison |
| Explicit and implicit conflict template | Existing work | Cross-framework comparison |
| Forced-choice format | This paper | A third value of the sign |


One point stated firmly: the scale-gradient claim holds only within a single model family, and
we make no cross-family comparison. The knowledge-rate difference between the two families
belongs to training data and recipe, is unrelated to scale, and would produce meaningless
conclusions if mixed in.

