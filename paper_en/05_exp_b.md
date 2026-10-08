# 4.4 Framework replication and self-refutation

All the conclusions of §4.2 rest on a question format of our own design. One obvious objection is: could the conclusions simply be an artifact of our question format? Prior work has reported this phenomenon using its own different templates; if we reproduced them verbatim, would we obtain the opposite sign, thereby showing that our gradient does not generalize?

This section tests that question directly. The conclusion is that our core finding is robust to the choice of framework, but the replication process exposed one failure that must be reported: the forced-choice template of one framework is completely dominated by position preference, and the numbers under that condition cannot be used for any content-level inference.

## 4.4.1 The three replicated external frameworks

The replication targets are the templates of two published works, whose wording and option construction we copied verbatim. The sources and construction of the three external framework conditions are given in Table S7 (online material).

In addition, that work's default condition `cb_default` serves as a control. The four conditions share the same set of 927 samples, the same pair of models (Qwen2.5-VL 3B/7B), and the same decoding parameters.
The motivation for this section comes from a repeatedly documented observation: option order and option format both shift multiple-choice results substantially [35,36], so "framework invariance" itself needs to be tested rather than assumed.

## 4.4.2 Main result: the sign does not flip with the framework

Under the replication run with a fixed option order (the parameter's true value is always A), the size effect $\Delta P(A)$ is negative for all four conditions:

/// Size effects and McNemar tests for the four replication conditions
| Condition | 3B | 7B | $\Delta P(A)$ | McNemar $p$ |
|---|---|---|---|---|
| `cb_default` | 0.981 | 0.941 | $-0.040$ | $3.0\times10^{-9}$ |
| `cb_conflict` | 0.494 | 0.475 | $-0.019$ | $0.23$ |
| `xie_implicit` | 0.479 | 0.476 | $-0.003$ | $0.89$ |
| `xie_explicit` | 0.475 | 0.436 | $-0.039$ | $0.018$ |


This agrees in sign with the two weakly authorized stances `ctx_only` and `orig` of our §4.2. In other words: under formatted forced-choice questions, the sign of the size effect is stably negative and does not change with the framework. The disagreement between the two published works cannot be explained by their prompt frameworks alone, which is consistent with the stance conclusion of our §4.2 — what determines the sign is the locus of authorization over parameter knowledge, not the template wording.

The magnitude of the effect should be noted: under forced choice it is at most only 4 pp, whereas under free generation the stance sweep spans 47.5 to 56.5 pp (negation-aware criterion, Qwen2.5 family and Qwen3 family).
The format itself compresses the effect, and this is itself the "third value of the sign" to be reported in §4.5.

## 4.4.3 One replication failure: position preference can completely override content

The reliability of the table above depends on one premise: the option letter carries no
information. We test it by swapping option positions, rerunning the same samples with the A/B
contents exchanged. (Sequence position also changes how much the model exploits the same
content [47]; our swap conflates order and position effects and cannot fully separate them, a
limitation we list.) Position preference is independently reported in multiple-choice evaluation
and model judging [35,36,40]; our contribution is its magnitude under the conflict condition —
0.961 is close to complete domination. The premise fails completely for `cb_default`, whose
position preference is close to 1 (Table S8, online material): its output is almost
entirely determined by which letter is the answer, independent of content, appearing to nearly
always choose correctly under one order (0.981) and incorrectly under the other (0.019).
Neither number measures "model judgment", and `cb_default` cannot be used for any content-level
inference. The position preferences of `cb_conflict` and both Xie frameworks are all below
0.11, so content dominates there and the sign conclusions of §4.4.2 hold; the `cb_default` row,
though of the same sign, must not be cited as evidence.

## 4.4.4 The self-refutation role of this section

We keep this failure in the main text rather than an appendix because it shows our robustness
checks are effective rather than formal: without the order swap, `cb_default`'s 0.981 would be
read as "the model makes almost no errors in this format" and, added to our negative-sign
conclusion, would reinforce a false impression. It also yields a transferable warning: the
accuracy of a forced-choice format is uninterpretable before order is controlled, yet such
formats are widely used in the knowledge-conflict literature and position preference is
invisible without a control. We therefore report order-control results for every forced-choice
condition, using their numbers only where position preference is negligible.

# 4.5 Statistical tests and power

All of our comparisons are paired binary comparisons: the same samples receive different
conditions within the same model and run, and what is compared is the direction of the flip.
This determines the choice of test and the power calculation.

## 4.5.1 Primary test: McNemar exact test

For each condition pair we construct a 2×2 contingency table; only the discordant cells (the
two directions of flip) enter the test. We use the exact binomial version of McNemar rather
than the chi-square approximation, because the flip counts are small in some cells
(`cb_default` only 43) and the approximation is unreliable there: for discordant counts $b$ and
$c$, $p = 2\sum_{i=0}^{\min(b,c)} \binom{b+c}{i} 2^{-(b+c)}$, truncated at 1, with $p=1$ when
the flip count is zero. We give the effect size alongside the sample size rather than
significance alone [43]. A paired test is chosen over a two-group independent-proportion test
for power: the same samples' answers under the two conditions are highly correlated, so the
paired design removes the large variance component of sample difficulty. For example, `prior`
on Qwen2.5-VL 3B→7B has flips of 15 versus 234, giving $p = 1.0\times10^{-51}$ from only 249
discordant samples.

## 4.5.2 Twofold testing of monotonicity

§4.2 needs to test the ordered alternative "$\Delta$ rises with authorization strength", which
cannot use a chi-square-type test, so we use two complementary quantities: the Spearman rank
correlation $\rho$, measuring the strength of the trend, and a permutation test with
$2\times10^5$ reshufflings of the authorization-order labels, giving the null distribution of
$\rho$ and the one-tailed $p$. Permutation rather than a table lookup is used because with only
7 stances the asymptotic distribution is unreliable; the test makes no distributional
assumption, and $2\times10^5$ reshufflings suffice to estimate $p$ stably to three significant
figures. Both are reported because they answer different questions — $\rho$ "how strong is the
trend", $p$ "how likely is a trend this strong by chance" — and we have cases where $\rho$ is
significantly positive yet the ordering still contains inversions (§4.2.3.3), where reporting
only "significant" would cause overinterpretation.

## 4.5.3 Between-group comparison: Kruskal-Wallis

Testing whether "stance significantly changes answer length" compares the length distribution
of the seven stance groups. Such data violate normality — the answers are text, the distribution
is right-skewed, and the median often falls in single digits while the P90 exceeds 300 — so we
use the Kruskal-Wallis rank-sum test, which requires only similar distribution shapes. With
$R_j$ the rank sum of group $j$,

$$H = \frac{12}{N(N+1)}\sum_{j=1}^{k} \frac{R_j^2}{n_j} - 3(N+1), \qquad
H_{\text{corr}} = H \Big/ \left(1 - \frac{\sum (t^3 - t)}{N^3 - N}\right)$$

the second being the tie correction, which cannot be omitted on integer length data. The tail
probability is given by the chi-square distribution; the implementation and its validation are
described in the reproduction material.

## 4.5.4 Power accounting

The anchor set $D^*$ has 649 items and all primary comparisons are paired on the same samples,
so the detection limit must be given for a paired design. With $\pi_d$ the observed proportion
of sample pairs on which the two models disagree, the minimum detectable effect is

$$\text{MDE} \approx (z_{1-\alpha/2} + z_{1-\beta})\sqrt{\frac{\pi_d}{n}}$$

Because $\pi_d$ is observed rather than designed, this accounting is post hoc: it answers "given
the disagreement rate actually observed for each stance, how large a $\Delta$ can be detected",
not "how many samples should have been recruited". Post hoc detection limits carry a known risk
of being over-optimistic [30], since the disagreement rate is itself affected by the condition,
so these numbers serve only as a sieve for "which cells are adequately powered", not as a power
claim.

Taking $\alpha = 0.05$ and power $1-\beta = 0.80$, and substituting the observed $\pi_d$ for each stance
(range 0.123 to 0.351), we obtain the per-stance detection limits: 4.1 to 7.1 pp for the Qwen2.5-VL family, median 5.5 pp; and 5.4 to 6.4 pp for the Qwen3-VL family, median 6.3 pp.
All values were computed by `src/analyze_mde.py` and saved to `results/mde.json`.

Read alongside the per-cell results of §4.2: the smallest significant $\Delta$ across the two
families is $-8.4$ pp for `ctx_only` in the Qwen2.5-VL family, exceeding that stance's 5.0 pp
detection limit, and the other significant cells are larger, up to 56.5 pp (between $+29.2$ for
Qwen3-family `ctx_hedge` and $-27.3$ for `ctx_only`). "Significant" on these cells is therefore
not an artifact of sample size.

Only one cell falls below its detection limit: $\Delta = +1.5$ pp with $p = 0.491$ for the Qwen3-family `neutral` stance, below that stance's 5.4 pp detection limit. This non-significance does not mean the effect is zero; it is below the detection limit. Throughout the main text we use "not detectable at our sample size" for non-significant cells rather than "no effect," and we neither use a sub-limit effect as evidence nor interpret it as "no difference in fact."

## 4.5.5 Two common tests we do not use

All primary comparisons are paired on the same samples, so we do not use the
independent-sample proportion test (it would underestimate power and overestimate $p$). The
Stuart-Maxwell test applies to paired comparisons among A/B/C; our abstention-pivot claim was
falsified and retracted, so the test lost its use and we report no abstention conclusion based
on three-category pairing.

# 4.6 Criterion comparison

Our main results depend on one set of alias-matching criteria. This section reports all
alternative criteria side by side, so that readers can judge how much of the conclusion comes
from the choice of criterion. The three agree in direction but differ greatly in magnitude.
Criterion dependence cannot be adjudicated by how good the results are: the reasons must be
written down before the results are seen [39,38], otherwise any choice can be defended after
the fact.

## 4.6.1 The three criteria

The defect of `distinctive_ground` (its undecidable rate covaries with answer length, $r = 0.983$) was described in §4.2.1; we retain it only as a control. Table~\ref{tab:criteria-rule} gives the decision rule and length sensitivity of the three criteria.

///[criteria-rule] Decision rules and length sensitivity of the three criteria
| Criterion | Decision rule | Length-sensitive |
|---|---|---|
| `distinctive_ground` | All content words of the answer must fall within a single evidence source | Highly sensitive |
| Alias matching | Bidirectional containment with the parameter alias table or the context alias table | Not sensitive |
| Negation-aware alias | Alias matching + those explicitly rejecting the context count as following the parameter | Not sensitive |


## 4.6.2 Comparison results for the three criteria

We compare the $\Delta$ given by the three criteria on the same family (Qwen2.5-VL 3B → 7B).
The numbers in parentheses are the paired decidable sample count for that cell — both sides must be decidable.

/// Size effects given by the three criteria on the same model family
| Stance | `distinctive_ground` | Alias matching | Negation-aware |
|---|---|---|---|
| `ctx_only` | $-7.3$ ($n=451$) | $-9.2$ ($n=556$) | $-8.4$ ($n=561$) |
| `orig` | $-11.7$ ($n=496$) | $-14.0$ ($n=591$) | $-13.8$ ($n=593$) |
| `ctx_hedge` | $-100.0$ ($n=1$, insufficient denominator) | $-4.2$ ($n=24$, insufficient denominator) | $+11.6$ ($n=584$) |
| `neutral` | $+8.7$ ($n=345$) | $+11.7$ ($n=520$) | $+15.9$ ($n=598$) |
| `own_only` | $+15.0$ ($n=327$) | $+15.9$ ($n=492$) | $+21.7$ ($n=584$) |
| `len_ctrl` | $+16.4$ ($n=122$) | $+9.0$ ($n=366$) | $+12.3$ ($n=530$) |
| `prior` | $+28.0$ ($n=93$) | $+23.5$ ($n=230$) | $+33.7$ ($n=543$) |
| Span (cells with $n\geq50$ only) | $39.7$ | $37.5$ | $47.5$ |


On the six cells with sufficient paired decidable samples the three criteria agree completely
in sign, with no cell reversed: the sign conclusion does not depend on the criterion. The only
sign-reversed cell is `ctx_hedge`, precisely where the denominator collapses most severely
The denominator collapses under the two length-sensitive criteria and is restored under the negation-aware one (Table~\ref{tab:denominator}):

///[denominator] Paired decidable sample counts and discarded amounts for the `ctx_hedge` stance under each criterion
| Criterion | `ctx_hedge` paired decidable $n$ | Discarded BOTH count (both sides combined) |
|---|---|---|
| `distinctive_ground` | 1 | 3 |
| Alias matching | 24 | 1108 |
| Negation-aware | 584 | 62 |


The three rows say the same thing: the denominator collapses from 584 to 1, not because the
criteria "disagree" but because the same answer is judged undecidable; after the collapse
$\Delta$ can only take degenerate values of $\pm100$ (the $-100.0$ above is determined by a
single sample), yet in the table it looks like "the cell with the strongest effect". We
therefore stipulate that cells with $n<50$ are not counted in the span and take part in no sign
comparison (as in §4.7.2: whenever a proportion is reported, its denominator must accompany
it). Without this the table would create the illusion that "the three criteria disagree on
`ctx_hedge`", whereas two of them have no usable denominator there at all; "six cells
consistent" is the robustness claim we actually use. In magnitude the negation-aware criterion
gives the largest span (47.5 pp) and alias matching the smallest (37.5 pp), consistent with
the criteria's length sensitivity; the larger span of `distinctive_ground` is not "a larger
effect" but "the hard-to-measure end was deleted" — it leaves only 93 decidable samples on
`prior` and 1 on `ctx_hedge`.

## 4.6.3 The `len_ctrl` cell: where the three criteria diverge most

The three criteria differ most on `len_ctrl` ($+16.4$, $+9.0$, $+12.3$; the $n$ of `distinctive_ground` has dropped to 122, so its magnitude has limited reference value), because it was designed precisely for length control and its answer lengths are of the same order as its authorization stance. Under that stance the model often mentions the parameter value and the context value at the same time, which the pure alias criterion judges BOTH and discards (the judged rate falls to 42.2%–67.7%); the negation-aware criterion rescues the explicitly rejecting portion, raising it to 82.8%~89.4%. All three agree in sign, positive; the divergence is only in magnitude, so we claim only the sign of that cell (written into §4.2.3.5).

## 4.6.4 Criterion choice cannot be decided by how good the results are

Changing the criterion makes the results stronger, which is itself a danger signal, so our
reasons must be structural defects diagnosed in advance and independent of the result's
direction. Two reasons (see §4.2.1): the construction itself fixes the outcome — bidirectional
containment forces "naming the wrong value in order to reject it" into BOTH — and the correction
rules (the pattern table of `regrade_negation.py`) were fixed before the results were seen,
split into five linguistic categories, with no tuning to results. Both criterion sets' numbers
are reported in full in §4.2.1.1, with no selective presentation. The criterion is a string
rule, not a semantic judgment; complex phrasings may be missed, which only falls back to the
pure alias criterion and cannot create false positives.
