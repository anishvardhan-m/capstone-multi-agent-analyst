# System Technical Specification

Capstone handbook Section 15.1 deliverable: architectural detail on the
data layer schemas, agent design decisions, and session-management loops
behind the Multi-Agent AI Data Analyst. This document assumes the reader
has read `README.md` for the high-level pitch and quick start; everything
here is implementation detail pulled directly from the current codebase.

---

## 1. Architecture overview

The capstone handbook's reference architecture specifies six layers:
User, Application, Agent, ML, Data, and Reporting. This is how each maps
onto the actual repository:

| Layer | Handbook responsibility | Concrete implementation |
|---|---|---|
| **User** | End-user interaction surface | `app/home.py` + `app/pages/2`–`9` — 8-page Streamlit dashboard |
| **Application** | Coordination, control flow, session state | `src/agents/orchestrator.py` (`OrchestratorAgent`) drives pipeline control flow; `app/dashboard_helpers.py` drives dashboard session state and cross-page path resolution |
| **Agent** | Task-specific autonomous units | `src/agents/cleaner.py`, `eda.py`, `feature_engineer.py`, `visualizer.py`, `insights.py`, `report_generator.py` — 6 of the 8 agents (ML is broken out below; Orchestrator is the Application layer) |
| **ML** | Model training, evaluation, validation rigor | `src/agents/ml_agent.py` (`MLAgent`) + `src/tools/ml_tools.py` — task-type detection, model selection, calibration, permutation importance, error analysis, seed robustness |
| **Data** | Persistence, transformation primitives, audit trail | `src/tools/data_tools.py`, `src/tools/feature_tools.py` (sklearn transformers), `src/tools/audit_db.py` (SQLite), `data/`, `workspace/metadata/` |
| **Reporting** | Human-facing output artifacts | `src/agents/visualizer.py` (charts) + `src/agents/report_generator.py` (PDF) → `workspace/visualizations/`, `workspace/executive_report.pdf` |

Data flows strictly left-to-right through the Agent/ML layers, with each
agent's output path becoming the next agent's input path (see Section 2's
per-agent I/O). The Orchestrator is the only component that calls every
other agent; no agent calls another agent directly. The Data layer
(SQLite audit trail) observes every agent call from the outside via a
decorator (`audit_logged`, Section 6) rather than agents writing to it
themselves — no agent module imports `sqlite3` directly except
`audit_db.py`.

---

## 2. Agent design

Eight agent classes exist: `DataCleaningAgent`, `EDAAgent`,
`FeatureEngineeringAgent`, `MLAgent`, `VisualizationAgent`,
`BusinessInsightsAgent`, `ReportGenerationAgent`, and `OrchestratorAgent`.

### 2.1 DataCleaningAgent (`src/agents/cleaner.py`)

- **Responsibility**: missing-data imputation, duplicate removal, type
  correction, low-variance column dropping.
- **I/O**: `run(data_path, output_path=None) -> (success, output_csv_path)`.
  Writes `<stem>_cleaned.csv` and `<stem>_cleaned_report.json`.
- **Implementation**: a `sklearn.pipeline.Pipeline` of four custom
  transformers from `src/tools/data_tools.py`, in order:
  `DuplicateRemover` → `DataTypeCorrector` → `MissingValueImputer` →
  `LowVarianceColumnDropper`.
- **Key design decision**: fully deterministic, no LLM call. See
  Section 7 for the reasoning shared across all five deterministic
  agents.

### 2.2 EDAAgent (`src/agents/eda.py`)

- **Responsibility**: descriptive statistics, Pearson correlation matrix,
  Fisher skewness, IQR-based outlier detection — per numeric column.
- **I/O**: `run(data_path) -> (success, report_json_path)`. Writes
  `<stem>_eda_report.json`.
- **Implementation**: pure pandas/numpy. Outlier fences use
  `iqr_multiplier` (default 1.5, the standard Tukey fence).
- **Key design decision**: deterministic; no model or LLM involved.

### 2.3 FeatureEngineeringAgent (`src/agents/feature_engineer.py`)

- **Responsibility**: drop redundant (highly correlated) columns, reduce
  skew via log1p, scale numeric features, encode categoricals.
- **I/O**: `run(data_path, output_path=None) -> (success, output_csv_path)`.
  Writes `<stem>_features.csv` and `<stem>_features_report.json`.
- **Implementation**: a 4-step `Pipeline` from `src/tools/feature_tools.py`,
  order deliberately fixed: `RedundantFeatureDropper` (`corr_threshold`,
  default 0.95) → `SkewnessReducer` (`skew_threshold`, default 1.0) →
  `NumericScaler` → `CategoricalEncoder` (`ohe_threshold`, default 20;
  one-hot below the threshold, frequency encoding above). Each step
  respects a `protected_cols` set (target + ID + any caller-supplied
  `extra_protected_cols`, e.g. a grouping ID) that passes through
  unmodified — this is what stops a grouping identifier from being
  frequency-encoded into a leaky count/proportion feature.
- **Key design decision**: deterministic pipeline order chosen so each
  step's assumptions hold for the next (e.g. scaling happens before
  one-hot encoding so binary dummy columns are never re-standardized).

### 2.4 MLAgent (`src/agents/ml_agent.py`)

- **Responsibility**: task-type detection, model selection via CV,
  final refit + held-out evaluation, calibration diagnostics,
  permutation-importance feature ranking, segment-level error analysis,
  optional multi-seed robustness check.
- **I/O**: `run(data_path, target_col, id_col=None, group_col=None) ->
  (success, report_json_path)`. Writes `<stem>_ml_report.json` and
  `models/best_production_model.pkl`; also appends one row to the
  `ml_experiments` SQLite table (Section 6). A second entry point,
  `run_robustness_check(data_path, target_col, id_col=None, group_col=None,
  seeds=(42, 7, 123, 2024, 99))`, writes `<stem>_robustness_report.json`
  and never touches the production model file (`save_model=False`
  internally).
- **Implementation**: candidate models are `LogisticRegression`,
  `RandomForestClassifier`, `HistGradientBoostingClassifier` for
  classification; `LinearRegression`, `RandomForestRegressor`,
  `HistGradientBoostingRegressor` for regression — swept via
  `GridSearchCV` (5-fold) except `LinearRegression`, which has no grid.
  See Section 5 for the rigor mechanisms (calibration, grouped
  splitting, robustness, segment detection) in detail.
- **Key design decision**: deterministic sklearn only — the model
  *training* is never LLM-driven. This is the clearest instance of the
  Section 7 tradeoff: an LLM could plausibly "write" a model-selection
  script per dataset, but the actual `GridSearchCV`/permutation-importance/
  calibration code must run identically and reproducibly every time.

### 2.5 VisualizationAgent (`src/agents/visualizer.py`)

- **Responsibility**: render every dashboard/PDF chart from the EDA
  report, ML report, and cleaned data.
- **I/O**: `run(eda_report_path, ml_report_path, cleaned_data_path,
  target_col="is_late_delivery", output_dir="workspace/visualizations")
  -> (success, visualization_report_json_path)`. Writes numbered PNGs
  (`01_distributions.png` … `08_error_by_segment.png`, task-type
  dependent — see below) plus `visualization_report.json`.
- **Implementation**: matplotlib/seaborn, `seaborn-v0_8-whitegrid` style,
  10×6" figures at 150 DPI. Charts 1–3 (distributions, correlation
  heatmap, feature importance) are task-agnostic. Charts 4–5 branch on
  `ml_report["task_type"]`: classification gets a confusion matrix +
  threshold-tradeoff line chart; regression gets an actual-vs-predicted
  scatter + residual plot. Chart 6 (top feature vs. target) is a box
  plot for classification, a scatter for regression. Chart 7
  (calibration curve) is classification-only. Chart 8 (error rate by
  segment) reads `error_analysis` and adapts its metric/label vocabulary
  to whichever task type produced it.
- **Key design decision**: every chart method independently checks for
  the upstream data it needs and calls `report.skip(name, reason)`
  instead of raising if that data is absent — matches the Orchestrator's
  "skip" recovery semantics (Section 4) by construction, not by
  coincidence.

### 2.6 BusinessInsightsAgent (`src/agents/insights.py`)

- **Responsibility**: the first (of two) genuinely LLM-calling agents —
  turns the EDA and ML reports into a markdown narrative for
  non-technical stakeholders.
- **I/O**: `run(eda_report_path, ml_report_path, output_dir="workspace")
  -> (success, markdown_path)`. Writes `business_insights.md`.
- **Implementation and design decisions**: see Section 4 (LLM
  integration) for the full prompt-engineering treatment.

### 2.7 ReportGenerationAgent (`src/agents/report_generator.py`)

- **Responsibility**: compile the cleaning report, ML report,
  `business_insights.md`, and chart PNGs into one executive PDF.
- **I/O**: `run(cleaning_report_path, ml_report_path, insights_md_path,
  chart_paths, output_path="workspace/executive_report.pdf",
  project_identifier=None) -> (success, pdf_path)`.
- **Implementation**: builds a self-contained HTML string (inline CSS,
  base64-embedded chart images) and renders it via
  `weasyprint.HTML(string=html).write_pdf(target=output_path)`. Exactly
  five sections, always in this order: Executive Summary → Data
  Diagnostics Profile → Model Performance Leaderboard → Automated
  Business Insights (narrative + chart gallery) → Strategic
  Recommendations. Section 3 branches on `ml["task_type"]`
  (`_REGRESSION_METRIC_LABELS` vs. `_CLASSIFICATION_METRIC_LABELS`) and
  never mixes the two metric vocabularies in one render.
- **Key design decision**: deterministic templating, explicitly *not* an
  LLM stage (see the module docstring) — reproducibility matters here:
  the same four inputs must always produce byte-for-byte the same PDF
  content. Every upstream input degrades to a placeholder string on
  missing/unreadable data rather than raising, recorded in
  `ReportGenerationReport.warnings` / `charts_skipped`.

### 2.8 OrchestratorAgent (`src/agents/orchestrator.py`)

- **Responsibility**: run all 7 other agents in sequence, threading each
  agent's output path into the next agent's input, and apply
  LLM-classified retry/skip/abort recovery on any step failure.
- **I/O**: `run(data_path, target_col, id_col=None, group_col=None,
  positive_label=None, negative_label=None, unit_label=None) ->
  (success, final_pdf_path_or_abort_reason)`.
- **Key design decision**: this is the second LLM-calling agent, but for
  a structurally different purpose than BusinessInsightsAgent — it
  never touches the data itself; it classifies *failures*. See Section 4.

### 2.9 The CrewAI/LangChain deviation

The handbook's "multi-agent" framing invites building this system on an
existing agent-orchestration framework (CrewAI, LangChain agents, etc.),
where each agent's task logic is itself LLM-generated/LLM-driven at
runtime. This project deliberately does not do that for six of the eight
agents. Every deterministic agent's module docstring makes the same
argument independently, in its own words (`cleaner.py`, `eda.py`,
`feature_engineer.py`, `visualizer.py`, `report_generator.py`,
`ml_agent.py` each state it separately rather than sharing one
canonical explanation) — `cleaner.py`'s version is representative:

> "this agent's cleaning LOGIC is fully deterministic (a Scikit-Learn
> Pipeline of the transformers in src/tools/data_tools.py). No LLM call
> happens inside this agent. This is intentional: cleaning rules should
> be reproducible and auditable, not subject to LLM non-determinism."

Concretely, an LLM-orchestrated framework would make each agent's
transformation logic itself a function of an LLM call — meaning the same
input CSV could plausibly produce different cleaning decisions,
different chart selections, or different model hyperparameter grids
between two runs, and unit-testing an agent's *behavior* (not just that
it "ran successfully") would require mocking an LLM at every layer. The
"agent" framing is preserved anyway — each class still inspects its
input, makes decisions (which columns to flag, which model wins CV),
and produces a structured, auditable report exactly like a domain
specialist would — but the decision *logic* is ordinary, testable Python
against a fixed rule set (thresholds, pipeline steps), not a prompt.

LLM calls are deliberately isolated to exactly two integration points
where non-determinism is either the point (natural-language narrative
generation) or bounded and safety-netted (pipeline failure
classification with a hard-coded `abort` fallback) — see Section 4. This
tradeoff is revisited explicitly in Section 7.

---

## 3. Data schemas

The JSON structures below are the actual `to_dict()` output of each
agent's report dataclass, current as of this document. Field names are
copied directly from source, not paraphrased.

### 3.1 `CleaningReport` (`src/agents/cleaner.py`)

```json
{
  "input_shape": [0, 0],
  "output_shape": [0, 0],
  "n_duplicates_removed": 0,
  "high_missing_columns_flagged": [],
  "low_variance_columns_dropped": [],
  "columns_type_corrected": [],
  "numeric_columns_imputed": [],
  "categorical_columns_imputed": []
}
```

### 3.2 `EDAReport` (`src/agents/eda.py`)

```json
{
  "input_shape": [0, 0],
  "numeric_columns": [],
  "categorical_columns": [],
  "descriptive_stats": { "col": { "mean": 0.0, "std": 0.0, "...": "pandas .describe() stats" } },
  "correlation_matrix": { "col_a": { "col_b": 0.0 } },
  "skewness": { "col": 0.0 },
  "outlier_summary": {
    "col": {
      "n_outliers": 0,
      "pct_outliers": 0.0,
      "lower_fence": 0.0,
      "upper_fence": 0.0
    }
  }
}
```

### 3.3 `FeatureEngineeringReport` (`src/agents/feature_engineer.py`)

```json
{
  "input_shape": [0, 0],
  "output_shape": [0, 0],
  "protected_columns": [],
  "redundant_columns_dropped": [],
  "log_transformed_columns": [],
  "scaled_columns": [],
  "zero_variance_columns_skipped": [],
  "encoding_map": {},
  "scaler_stats": {}
}
```

### 3.4 `MLReport` (`src/agents/ml_agent.py`)

The most complex schema in the system. All fields are always present
(defaulted to `null`/`{}`/`[]`); which ones are populated depends on
`task_type`.

```json
{
  "task_type": "binary_classification | multiclass_classification | regression",
  "best_model_name": "",
  "target_col": "",
  "cv_scores": { "ModelName": 0.0 },
  "best_hyperparameters": {},
  "test_metrics": {},
  "confusion_matrix": null,
  "threshold_metrics": null,
  "feature_importances": {
    "feature_name": {
      "importance_mean": 0.0,
      "importance_std": 0.0,
      "distinguishable_from_zero": true
    }
  },
  "error_analysis": {
    "task_type": "",
    "segment_columns": [],
    "detection_note": "",
    "overall": {},
    "segments": { "segment_col": [ { "segment_value": 0.0, "n": 0 } ] },
    "note": ""
  },
  "test_predictions": null,
  "split_strategy": "row_random | grouped",
  "group_col": null,
  "cv_fold_scores": {},
  "cv_std": {},
  "model_selection_note": null,
  "calibration": null,
  "calibrated_comparison": null,
  "calibration_note": "",
  "nested_cv_score": null,
  "nested_cv_std": null,
  "nested_cv_note": ""
}
```

Task-type-dependent fields:
- `confusion_matrix`, `threshold_metrics`, `calibration`,
  `calibrated_comparison`, `calibration_note` — classification only
  (calibration is binary-classification only specifically).
- `test_predictions` — regression only:
  `{"actual": [...], "predicted": [...]}`, downsampled for chart use.
- `error_analysis.segments[col][i]` shape depends on `task_type`:
  binary classification carries `false_negative_rate` /
  `false_positive_rate` (+ `elevated_false_negative_rate` /
  `elevated_false_positive_rate` flags); multiclass carries
  `misclassification_rate` (+ `elevated_misclassification_rate`);
  regression carries `mae` / `mean_error` (+ `elevated_mae` /
  `elevated_bias`).

`run_robustness_check` produces a separate, not-dataclass-backed dict
(`_aggregate_robustness`'s return value), written to
`<stem>_robustness_report.json`:

```json
{
  "task_type": "",
  "n_seeds": 0,
  "seeds": [42, 7, 123, 2024, 99],
  "best_model_by_seed": { "42": "ModelName" },
  "model_agreement": { "ModelName": 0 },
  "winning_model": "",
  "winning_model_seed_agreement": 0,
  "winning_model_is_robust": true,
  "test_metrics_by_seed": { "42": {} },
  "test_metrics_summary": { "metric": { "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0 } },
  "top_k_features": 5,
  "top_features_by_seed": { "42": [] },
  "feature_appearance_counts": {},
  "always_in_top_k_features": [],
  "seed_dependent_features": [],
  "note": "",
  "failed_seeds": [],
  "n_seeds_attempted": 0
}
```

### 3.5 `VisualizationReport` (`src/agents/visualizer.py`)

```json
{
  "charts": [ { "name": "", "path": "", "description": "" } ],
  "skipped": [ { "name": "", "reason": "" } ]
}
```

### 3.6 `InsightsReport` (`src/agents/insights.py`)

Written to `agent.report_` in-process (not itself serialized to JSON —
only its `narrative` field is persisted, as `business_insights.md`).

```json
{
  "narrative": "markdown text with ## What We Found / ## What Matters Most / ## Recommendations",
  "model_used": "inclusionai/ling-3.0-flash:free | openrouter/free",
  "output_path": "workspace/business_insights.md"
}
```

### 3.7 `ReportGenerationReport` (`src/agents/report_generator.py`)

```json
{
  "output_path": "",
  "sections_included": ["executive_summary", "data_diagnostics", "model_performance", "business_insights", "recommendations"],
  "charts_embedded": [],
  "charts_skipped": [ { "path": "", "reason": "" } ],
  "warnings": []
}
```

### 3.8 `OrchestratorReport` (`src/agents/orchestrator.py`)

```json
{
  "steps": [
    {
      "name": "DataCleaningAgent",
      "status": "success | success_after_retry | skipped | failed",
      "attempts": 1,
      "message": "",
      "llm_action": "retry | skip | abort | null",
      "llm_reasoning": "one sentence, or null"
    }
  ],
  "total_duration_seconds": 0.0,
  "model_path": null,
  "final_report_path": null,
  "aborted": false,
  "abort_reason": null
}
```

`steps` always has one entry per `STEP_NAMES` value that was reached:
`DataCleaningAgent, EDAAgent, FeatureEngineeringAgent, MLAgent,
VisualizationAgent, BusinessInsightsAgent, ReportGenerationAgent`.

---

## 4. LLM integration details

Exactly two agents make real LLM calls: `BusinessInsightsAgent` and
`OrchestratorAgent` (for failure recovery only, not for pipeline data).
Both connect through the OpenAI Python SDK pointed at an OpenAI-compatible
base URL — in practice OpenRouter — configured via two environment
variables (`.env`, loaded with `python-dotenv`):

```
OPENAI_API_KEY="..."          # an OpenRouter API key
OPENAI_API_BASE_URL="..."     # OpenRouter's OpenAI-compatible endpoint
```

### 4.1 Two-model fallback pattern

Both agents share the identical pattern, defined independently in each
module (`insights.py`, `orchestrator.py`):

```python
_PRIMARY_MODEL = "inclusionai/ling-3.0-flash:free"
_FALLBACK_MODEL = "openrouter/free"
```

`BusinessInsightsAgent.run()` tries `_PRIMARY_MODEL`, and on any
exception (including rate-limiting or an empty response) retries once
against `_FALLBACK_MODEL` before returning `(False, error_message)`.
`OrchestratorAgent._get_recovery_action()` follows the same two-model
loop, but with a hard-coded safe default if *both* fail or return an
unparseable response: `action="abort"`. This asymmetry is deliberate —
a failed narrative generation degrades the PDF's insight section to a
placeholder (Section 2.7); a failed recovery classification must never
leave the pipeline's next action undefined, so it fails closed rather
than open.

The LLM client is always constructor-injectable
(`BusinessInsightsAgent(client=...)`, `OrchestratorAgent(client=...)`),
which is what lets the test suite exercise prompt construction and
fallback logic against a mock with zero real network calls.

### 4.2 BusinessInsightsAgent prompt engineering

The prompt is built entirely in Python string templates
(`_build_classification_prompt` / `_build_regression_prompt` in
`insights.py`), never raw JSON — the EDA and ML reports are first
reduced to readable bullet-point summaries so the model reasons over
the numbers that matter rather than parsing structure. Four decisions
are worth calling out specifically:

- **Accuracy/ROC-AUC framing.** The prompt computes `OVERALL ACCURACY`
  and `MAJORITY-CLASS BASELINE ACCURACY` itself, in Python
  (`_accuracy_from_confusion_matrix`, `_majority_baseline_accuracy`) —
  never left for the LLM to derive — because the ML report's
  `test_metrics` only contains ROC-AUC and macro precision/recall/F1,
  not plain accuracy, and the two are easy for a model to conflate
  since both are single numbers in the same 0–1 range. The prompt then
  *requires* the narrative to state both figures together and say
  explicitly whether the model beat the baseline, with an explicit
  instruction never to present accuracy alone as good news on an
  imbalanced target.
- **Causal-language hedging.** `_CAUSAL_LANGUAGE_GUARDRAIL` is injected
  into every prompt (classification and regression alike) as a
  standalone instruction block, on top of section-specific hedges
  already present in the error-analysis instructions
  (`_ERROR_CONCENTRATION_HEDGE`, "an association observed in this
  held-out test set, not a causal claim..."). The guardrail explicitly
  bans causal words ("causes", "leads to", "because", "drives", etc.)
  everywhere in the narrative, not just in the error-analysis section —
  the module docstring notes this repetition exists because "an LLM
  given permission to sound authoritative in one section tends to drift
  into causal phrasing in the next one otherwise."
- **Materiality threshold.** `_IMPORTANCE_MATERIALITY_THRESHOLD = 0.01`.
  Features with `|importance_mean| <= 0.01` are filtered out of the
  prompt entirely (`_format_feature_importance_lines`), not just
  down-weighted — the reasoning documented in-line is that this is what
  stops the LLM from inventing a business story for a feature that
  isn't actually predictive. Features that clear the threshold but are
  flagged `distinguishable_from_zero: false` are still shown, but
  explicitly labeled `NOT DISTINGUISHABLE FROM ZERO`, and the prompt
  separately forbids building any claim around one.
- **Domain-label genericity.** `positive_label`, `negative_label`,
  `unit_label` are constructor parameters, not hardcoded strings — see
  Section 5.

### 4.3 OrchestratorAgent recovery prompt

A much narrower prompt than BusinessInsightsAgent's — a fixed system
prompt (`_RECOVERY_SYSTEM_PROMPT`) asks for exactly two lines of output
(`ACTION: retry|skip|abort` / `REASON: <one sentence>`), parsed by
`_parse_recovery_response` via prefix matching, not JSON parsing (chosen
for robustness against a free-tier model's tendency to wrap JSON in
markdown fences or add commentary). The user prompt gives the failed
step's name, its error message, which steps already completed, and
which remain — enough context for the LLM to judge whether a later step
can plausibly proceed without this one's output, without exposing any
of the underlying data.

---

## 5. Genericity mechanisms

The system is designed to run unmodified against any classification or
regression CSV, not just the Olist demo. Three mechanisms make that
true structurally, not just by convention:

1. **Task-type auto-detection** (`src/tools/ml_tools.py::detect_task_type`).
   Pure function, three ordered rules: exactly 2 unique non-null target
   values → `binary_classification`; integer/object dtype with
   `nunique <= multiclass_unique_threshold` (default 20) →
   `multiclass_classification`; otherwise → `regression`. Every
   downstream agent (`VisualizationAgent`, `ReportGenerationAgent`,
   `BusinessInsightsAgent`) branches its output vocabulary on this one
   value (`ml_report["task_type"]`) rather than re-deriving or assuming
   it.
2. **Configurable domain labels.** `positive_label` / `negative_label` /
   `unit_label` are constructor parameters on `VisualizationAgent` and
   `BusinessInsightsAgent`, threaded through from `OrchestratorAgent.run()`
   (also constructor params there), with generic defaults
   (`"Positive"`/`"Negative"`/`"record"` for charts;
   `"positive case"`/`"negative case"`/`"record"` for insights) used only
   when a caller doesn't supply its own domain vocabulary. This project's
   own `__main__` blocks pass concrete Olist values (`"late delivery"` /
   `"on-time delivery"` / `"order"`) explicitly rather than relying on
   the defaults — the agents themselves stay domain-agnostic.
3. **Threshold-based rules instead of hardcoded values.** Every
   structural decision in the pipeline is a configurable numeric
   threshold, not a column-name check: `high_missing_threshold` (0.5),
   `low_variance_threshold` (0.99), `corr_threshold` (0.95),
   `skew_threshold` (1.0), `ohe_threshold` (20 categories),
   `iqr_multiplier` (1.5), `_IMPORTANCE_MATERIALITY_THRESHOLD` (0.01),
   `_ECE_MISCALIBRATION_THRESHOLD` (0.05), and the segment-detection
   cardinality band (`_ERROR_ANALYSIS_MIN_CATEGORIES=2`,
   `_ERROR_ANALYSIS_MAX_CATEGORIES=50`). The segment-detection mechanism
   in particular (`MLAgent._detect_segment_columns`) never references a
   column name at all — it selects candidate segment axes purely by
   post-encoding cardinality (a one-hot flag has 2 distinct values, a
   frequency/label-encoded categorical has roughly one per original
   category, a truly continuous feature has close to as many distinct
   values as rows), so the same logic finds "the categorical-shaped
   columns" on a feature matrix with an entirely different schema.

---

## 6. Session/state management

### 6.1 SQLite audit trail (`src/tools/audit_db.py`)

Two tables, created idempotently via `CREATE TABLE IF NOT EXISTS` in
`init_db()`, at `workspace/metadata/audit_telemetry.db` by default
(`DEFAULT_DB_PATH`, resolved fresh on every call so tests can
monkeypatch it):

**`agent_runs`** — one row per agent `.run()` call, success or failure:

```sql
CREATE TABLE agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    status TEXT NOT NULL,
    input_path TEXT,
    output_path TEXT,
    error_message TEXT,
    duration_seconds REAL NOT NULL
)
```

**`ml_experiments`** — one row per successful `MLAgent.run()` (not
`run_robustness_check`, whose seeds are exploratory rather than
canonical runs), so the dashboard's Run History page can compare metrics
across past runs even though `MLAgent.run()` overwrites the same
`<data>_ml_report.json` file every time it's called:

```sql
CREATE TABLE ml_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    data_path TEXT NOT NULL,
    target_col TEXT NOT NULL,
    task_type TEXT NOT NULL,
    best_model_name TEXT NOT NULL,
    split_strategy TEXT,
    group_col TEXT,
    random_state INTEGER,
    n_features INTEGER,
    best_hyperparameters TEXT,   -- JSON
    cv_scores TEXT,              -- JSON
    cv_std TEXT,                 -- JSON
    test_metrics TEXT,           -- JSON
    model_selection_note TEXT,
    nested_cv_score REAL,
    nested_cv_std REAL,
    report_path TEXT
)
```

dict-valued fields are stored as JSON text (`json.dumps`) and decoded
back on read (`get_recent_experiments`); the schema itself has no
hardcoded metric names, so a classification run's `{"f1_macro",
"roc_auc", ...}` and a regression run's `{"rmse", "mae"}` store
identically in the same `test_metrics` column.

A failed `MLAgent.run()` still gets its usual row in `agent_runs` (every
`run()` call is `@audit_logged`, success or failure), but writes nothing
to `ml_experiments` at all — not even a partial row with nulls.
`log_ml_experiment()` is only called after the success path's early
return, so fields like `best_model_name` and `task_type`, which don't
exist yet on a failed run, are never in a position to be missing from an
inserted row.

Every agent wires into `agent_runs` via the `@audit_logged(agent_name,
input_arg=...)` decorator wrapping its `run()` method — it observes
timing and outcome without changing behavior; the decorated function's
`(success, message)` return value (or a re-raised exception) passes
through unchanged. `input_arg` can be a tuple for agents that read
multiple input files (e.g. `VisualizationAgent`'s
`("eda_report_path", "ml_report_path", "cleaned_data_path")`), joined
with `"; "` in the logged row.

### 6.2 Dashboard session_state + disk-fallback pattern

`app/dashboard_helpers.py` defines `DEFAULT_SESSION_STATE` — a flat dict
of keys (`raw_data_path`, `target_col`, `id_col`, `group_col`,
`pipeline_ran`, `pipeline_success`, `orchestrator_message`,
`orchestrator_report`, plus every intermediate artifact path) —
populated onto `st.session_state` by `ensure_session_state()` on every
page load. This is the dashboard's only in-memory state; nothing
persists across a full process restart except what's already on disk
(reports, charts, the SQLite DB).

Every page that displays pipeline output resolves its data through
`resolve_report_path(session_path, default_relative_path, project_root)`:
prefer this session's own output if `session_path` is set and the file
still exists; otherwise fall back to the checked-in Olist demo output at
a fixed `data/processed/...` path; otherwise show nothing. The function
returns `(path_or_None, source)` where `source` is `"session"` or
`"fallback"`, so the UI can tell the user when they're looking at prior
demo output rather than a run from the current session. This is the
same graceful-degradation contract every agent already follows, applied
to the dashboard layer: a fresh install with no pipeline run yet still
renders something real instead of an empty page.

---

## 7. Known architectural tradeoffs

**Deterministic pipeline code vs. LLM-generated/LLM-orchestrated code.**
The handbook's multi-agent framing is compatible with building this
system on CrewAI, LangChain agents, or a similar framework where each
agent's task-level logic is itself produced or driven by an LLM at
runtime. This project deliberately chose not to do that for 6 of the 8
agents (Section 2.9), trading a more literal reading of "multi-agent AI
system" for:

- **Reliability.** The same input CSV always produces the same cleaning
  decisions, the same chart selection, the same model comparison —
  there is no LLM-call failure mode (rate limit, hallucinated output,
  model drift between provider updates) anywhere in the data-processing
  path itself.
- **Testability.** 372 unit/integration tests exercise agent *behavior*
  directly — pipeline branching, threshold edge cases, graceful
  degradation on malformed input — without needing to mock an LLM at
  every layer or accept flaky, non-deterministic test assertions.
- **Auditability.** Every transformation an agent applies (which column
  was flagged, which model won, which chart was skipped and why) is
  traceable to an explicit, readable rule in Python, not to an opaque
  model decision that would itself need explaining.

The cost is a system less literally "agentic" in the framework sense —
there is no LLM in the loop deciding *how* to clean a dataset or *which*
chart to draw. The two points where an LLM genuinely is in the loop
(Section 4) were chosen because they are exactly the two places in this
pipeline where the tradeoff reverses: natural-language narrative
generation has no deterministic equivalent worth building, and pipeline
failure triage benefits from judgment across heterogeneous,
unpredictable error messages more than from a fixed if/else ladder — and
both are bounded by a hard-coded safe fallback (a placeholder narrative
section; `abort`) so an LLM failure degrades the *output*, never leaves
the *system* in an undefined state.

This tradeoff, and the specific numeric consequences of the ML layer's
own rigor work (calibration, seed sensitivity, segment error
concentration), are documented in full in
[`KNOWN_LIMITATIONS.md`](../KNOWN_LIMITATIONS.md) — this document covers
architecture; that one covers what the architecture's own verification
passes (F1–F9) found when pointed at the real Olist model.
