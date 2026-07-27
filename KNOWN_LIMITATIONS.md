# Known Limitations

This document records limitations of the modeling pipeline and its output
that were **actually discovered while building and rigor-testing this
project** (the F1–F9 verification passes), not a generic disclaimer
checklist. Every number below comes from a real run against the live
Olist dataset (`data/processed/olist_flattened_cleaned_features.csv`,
target `is_late_delivery`, `random_state=42`, grouped split on
`customer_unique_id`) — reproduce them yourself with:

```
python -m src.agents.ml_agent data/processed/olist_flattened_cleaned_features.csv is_late_delivery
```

(pass `group_col="customer_unique_id"` when calling `MLAgent.run()`
directly — the module's CLI entrypoint does not expose it; see
[Split-strategy sensitivity](#split-strategy-sensitivity-a-footgun-we-hit-ourselves) below).

---

## 1. Raw predicted probabilities are not calibrated

**Finding.** The production model's `predict_proba` output does not mean
what it looks like it means. Measured Expected Calibration Error (ECE) on
the held-out test set is **0.3061** — over 6x this project's own
0.05 "acceptable" threshold — with a Brier score of **0.1758**. A
diagnostic isotonic-calibrated comparison model achieves ECE **0.0070**,
but **that calibrated model is not what's shipped**: the serialized
production artifact (`models/best_production_model.pkl`) is always the
raw, uncalibrated `HistGradientBoostingClassifier`.

**Why.** Every classifier here trains with `class_weight="balanced"` to
correct for the ~8% positive rate. That correction systematically shifts
`predict_proba` output away from true probabilities — it's the same
mechanism that improves minority-class recall, and it's not a bug, but it
means the model's probability outputs are not literal percentage chances.

**Practical consequence.** Do not treat a prediction like "probability of
late delivery = 0.62" as a real 62% chance. Only the three swept decision
thresholds (0.3 / 0.4 / 0.5) have been validated as precision/recall
operating points — treat them as business cutoffs, not calibrated
probabilities. If a downstream system needs genuine probability estimates
(e.g. for expected-cost calculations), it must use the isotonic-calibrated
comparison model, not the production one, and that comparison model has
never been serialized or evaluated for calibration drift over time.

---

## 2. Overall accuracy is misleading on this target — and is currently *worse* than doing nothing

**Finding.** On the held-out test set: OVERALL ACCURACY = **74.9%**
(confusion matrix `[[13397, 4291], [538, 1043]]`). A trivial classifier
that always predicts "on-time delivery" — zero modeling effort, zero
signal — would score **91.8%** on the same test set. The shipped model's
raw accuracy is **17 points below** that trivial baseline.

**Why.** This is the direct, expected cost of `class_weight="balanced"`:
the model trades away raw accuracy in exchange for catching more actual
late deliveries (recall on the minority class). At the default 0.5
threshold it catches 66.0% of actual late deliveries (recall) at the cost
of a lot of false alarms (precision 19.6%). This is not a flaw to be fixed
by better tuning — it's the tradeoff class-imbalance correction makes,
and it means **accuracy is the wrong headline metric for this model
entirely**. ROC-AUC (0.781) and the precision/recall pair at the chosen
threshold are what actually describe performance here.

**Practical consequence.** Never report "74.9% accurate" to a business
stakeholder without the baseline comparison — it reads as a strong result
when it is, by the accuracy metric alone, worse than a model with no
skill. (The Business Insights Agent prompt now enforces this comparison
automatically — see F8 — but any other consumer of this model's output
must apply the same caveat manually.)

---

## 3. Feature-importance rankings are seed-sensitive for geographic features

**Finding.** Re-running the full pipeline (split → model selection →
refit → permutation importance) across 5 different random seeds
(`42, 7, 123, 2024, 99`) shows:

- **Stable across every seed** (always in the top 5): `customer_zip_code_prefix`,
  `order_estimated_delivery_date`, `purchase_to_estimated_days`.
- **Seed-dependent** (in the top 5 for *some* seeds, not others):
  `customer_state`, `customer_city`, `primary_seller_state`.

Held-out metrics themselves are stable (f1_macro = 0.569 ± 0.005 across
the 5 seeds, roc_auc = 0.775 ± 0.004, min/max within ~1 point of the mean
in both cases), and the winning model (`HistGradientBoostingClassifier`)
was unanimous across all 5 seeds. Only the *ranking of correlated
geographic features* moves.

**Why.** `customer_state`, `customer_city`, `primary_seller_state`, and
`customer_zip_code_prefix` are all encoding roughly the same underlying
geographic signal (a customer's location). When features are this
collinear, which one a specific train/test split and CV fold happens to
lean on for its top-5 slot is partly arbitrary — the *model* is stable,
but attributing importance to one specific one of these four features
over the others is not.

**Practical consequence.** The single-seed report (the one shown in ML
Studio / Insights Panel by default) states `customer_state` as the single
most important feature. That is true for seed 42 specifically, and is a
defensible statement about that run — but it should not be read as "state
matters more than zip code" as a general, seed-independent claim. If a
business decision hinges on *which specific geographic granularity*
matters most (state vs. city vs. seller location vs. zip prefix), rerun
`MLAgent.run_robustness_check(...)` first and use `always_in_top_k_features`
/ `seed_dependent_features`, not a single run's ranking.

---

## 4. Error rates are unevenly and asymmetrically distributed across `customer_state`, in a pattern we can observe but not explain

**Finding.** Overall on the held-out set: false-negative rate = 34.0%,
false-positive rate = 24.3%. Broken down by `customer_state` (auto-detected
segment column, 24 distinct encoded values in the current run):

- **14 of 24 segments** — concentrated among the smaller/rarer states —
  show a **false-positive rate flagged as elevated** relative to the
  overall 24.3% (individual segment FP rates as high as 69%). These are
  *not* uniformly "all" segments: 2 of 24 are flagged elevated
  false-negative instead (see below), and 8 of 24 show no elevated flag
  at all — this should not be overstated (see the LLM-fidelity caveat
  below).
- The **single largest segment** (encoded frequency 0.42, n=8,035 — by
  far the highest-order-volume state, more than 3x the next largest) shows
  the *opposite* pattern: an **elevated false-negative rate** (52.8% vs.
  34.0% overall) rather than elevated false-positive. A second, mid-sized
  segment (n=993, the 5th-largest by row count) shows the same
  FN-elevated pattern (54.1%).
- `primary_seller_state` and `n_distinct_products` show one elevated
  segment each; `max_installments` and `n_items` show none.

**Why this happened is unknown.** Per this project's own causal-language
rule (see the Business Insights Agent prompt, F8), this is **an
association observed in held-out predictions, not a causal claim**. It
could reflect real logistics differences (distribution hub coverage,
carrier density, transit distance) that this dataset doesn't capture as
features — or it could partly reflect that smaller-`customer_state`
segments simply have fewer test rows (many near the 20-row minimum
segment-size floor), making their rates noisier even where individual
counts clear that floor. This project has not (and, with the current
feature set, cannot) distinguish between those explanations.

**A related, self-discovered limitation: the LLM narrative can overstate
this pattern.** When first generating `workspace/business_insights.md`
against this exact data, the LLM's own summary claimed the false-positive
rate was elevated "across every customer_state segment tested, ranging
from 0.534 to 0.692" — neither part of that claim is accurate. It's a
majority (14 of 24 segments), not all of them; the true range across
*all* 24 segments is 0.111–0.692, and even restricted to just the 14
segments actually flagged as elevated, the range is 0.382–0.692, not
0.534–0.692. The underlying `error_analysis` data passed to the LLM is
exact per-segment; the generated narrative compressed that precision into
a stronger, narrower-sounding claim than the data supports. **Any number
quoted in the generated markdown narrative should be checked against the
source JSON report before being repeated externally** — grounding the
prompt with real data measurably reduces hallucination
(see F8, F6) but does not eliminate the risk of the LLM overstating a
real pattern found in that data.

---

## 5. Split-strategy sensitivity — a footgun we hit ourselves

**Finding.** `MLAgent.run()`'s `group_col` parameter (which enables the
leakage-safe `GroupShuffleSplit` on `customer_unique_id` instead of a
plain row-random split) is **not exposed by the module's CLI entrypoint**
(`python -m src.agents.ml_agent <csv> <target> [id_col]`). Every metric in
this document uses the grouped split. Running the exact same data and
seed through the CLI instead — which several of this project's own
verification runs did by accident before this was caught — silently
produces a *different, leakier* result:

| | split_strategy | confusion matrix | accuracy | ROC-AUC |
|---|---|---|---|---|
| `run(group_col="customer_unique_id")` (correct) | grouped | `[[13397,4291],[538,1043]]` | 74.9% | 0.781 |
| CLI / `run()` with no `group_col` | row_random | `[[13262,4469],[541,1024]]` | 74.0% | 0.769 |

Both are deterministic and reproducible given their respective split
strategy and seed — this is not run-to-run noise, it is a real,
~1-point swing purely from which split code path executes.

**Practical consequence.** Any script, notebook, or ad-hoc invocation of
this pipeline must explicitly pass `group_col="customer_unique_id"` (or
call `orchestrator.py`, which always does). The bare CLI form is a trap
for exactly the customer-leakage failure mode `group_col` exists to
prevent.

---

## 6. Practical utility at any threshold is limited by low precision

**Finding.** Across the full threshold sweep, precision on the minority
class never exceeds ~20%:

| threshold | recall (catch rate) | precision (hit rate on flags) |
|---|---|---|
| 0.3 | 90.8% | 12.2% |
| 0.4 | 79.6% | 15.4% |
| 0.5 | 66.0% | 19.6% |

**Practical consequence.** Even at the most conservative threshold tested,
roughly 4 out of 5 orders flagged as "will be late" will not actually be
late. Any operational process built on these flags (e.g. proactive
customer outreach, expedited handling) must be designed around a high
false-alarm rate — this is a genericity/materiality limit of the
available features, not a tuning problem within reach of this pipeline's
current candidate models or feature set.

---

## 7. Interpretability is constrained by feature-engineering's encoding

**Finding.** Every categorical column (`customer_state`, `customer_city`,
`primary_seller_state`, `product_category`, timestamps) is frequency- or
label-encoded to an opaque float before reaching the model, and neither
the ML report nor the error-analysis segment breakdown retains a mapping
back to the human-readable original value. The segment-analysis table
literally reports rows like `customer_state=0.419742`, not `customer_state=SP`.

**Practical consequence.** We ourselves could only describe the
largest `customer_state` segment above as "almost certainly the
highest-volume state" — we could not name it without re-deriving the
mapping by hand from the pre-encoding data. Any consumer of the error
analysis or feature importance output who needs to act on a *specific*
segment (e.g. "target additional carrier capacity in state X") must
manually cross-reference the encoded value against
`data/processed/olist_flattened_cleaned.csv` (pre-feature-engineering)
or the `encoding_map`/`scaler_stats` in the features report — this
pipeline does not do that translation automatically.

---

## 8. Robustness testing itself has a limited scope

**Finding.** `run_robustness_check` (F7) varies only the random seed
(train/test split draw, CV fold assignment, model-internal randomness) —
5 seeds by default. It does **not** vary: the underlying data (e.g.
a different time window, or newly added regions/sellers not present in
training), the train/test size ratio, or the feature set itself. A
result reported as "robust" (winning model unanimous, metrics within a
tight band across 5 seeds) is only robust *to resampling noise on this
fixed dataset* — it says nothing about robustness to distribution shift
(e.g. a new shipping carrier, a change in Olist's estimated-delivery-date
policy, or expansion into a state not well-represented in this training
data).

---

## Summary table

| # | Limitation | Key number | Source |
|---|---|---|---|
| 1 | Predicted probabilities uncalibrated | ECE 0.306 (raw) vs 0.007 (isotonic, not shipped) | F3 |
| 2 | Accuracy misleading / below trivial baseline | 74.9% vs. 91.8% baseline | F8 |
| 3 | Geographic feature ranking is seed-dependent | 3/6 top features seed-stable, 3/6 not | F7 |
| 4 | Error rates uneven across `customer_state`, cause unknown | 14/24 segments elevated FP; 2/24 elevated FN instead (incl. the largest) | F6 |
| 5 | CLI omits `group_col`, silently changes results | 74.9%/0.781 (grouped) vs 74.0%/0.769 (row-random) | F1, verified F7–F9 |
| 6 | Low precision at every threshold | 12–20% precision across the sweep | (baseline ML report) |
| 7 | Encoded categorical values aren't human-readable | e.g. `customer_state=0.419742` | F5, F6 |
| 8 | Robustness check doesn't cover distribution shift | 5 seeds, same fixed dataset | F7 |
