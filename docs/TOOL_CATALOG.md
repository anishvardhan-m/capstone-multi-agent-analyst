# API & Custom Tool Catalog

Capstone handbook Section 15.1 deliverable: reference documentation for
every function/class in `src/tools/` — the deterministic, unit-testable
building blocks the agents (documented separately in
[`TECHNICAL_SPEC.md`](TECHNICAL_SPEC.md)) compose into pipelines. This is
a reference document, organized by file in pipeline execution order
(`data_tools.py` → `feature_tools.py` → `ml_tools.py` → `audit_db.py` →
`logging_config.py`); search it rather than reading it end to end.

All signatures are copied verbatim from source, current as of this
document.

---

## `src/tools/data_tools.py`

Cleaning transformers used by `DataCleaningAgent`. Every class follows
the sklearn `BaseEstimator, TransformerMixin` interface (`fit`/
`transform`), so each composes directly into a `sklearn.pipeline.Pipeline`
and is individually unit-testable.

**Leakage-prevention contract (module-wide)**: `fit` learns any needed
statistics (medians, modes, thresholds) only from the data passed to
it. `transform` applies those already-learned values without
recomputing them. Callers are responsible for calling `fit` only on
training data and `transform` (never `fit_transform`) on
validation/test data — this is what stops statistics from a held-out
split leaking into training.

### `class DuplicateRemover(BaseEstimator, TransformerMixin)`

```python
def __init__(self, subset: Optional[list[str]] = None)
def fit(self, X: pd.DataFrame, y=None) -> "DuplicateRemover"
def transform(self, X: pd.DataFrame) -> pd.DataFrame
```

- **Does**: drops exact duplicate rows via `X.drop_duplicates(subset=self.subset)`.
- **Params**: `subset: list[str] | None = None` — columns to consider
  when identifying duplicates; `None` means all columns.
- **Output**: a `pd.DataFrame` with duplicate rows removed. Fitted
  attribute `n_duplicates_removed_: int` records how many rows were
  dropped.
- **Leakage note**: `fit` is a no-op (`return self`) — duplicate removal
  has no statistic to learn, so this transformer is leakage-safe by
  construction; `transform` alone does the work.
- **Tested in**: `tests/test_cleaner.py::test_duplicate_remover_removes_exact_duplicates`,
  `::test_duplicate_remover_handles_no_duplicates`,
  `::test_duplicate_remover_handles_empty_dataframe`.

### `class PlaceholderNullNormalizer(BaseEstimator, TransformerMixin)`

```python
def __init__(self, placeholder_values: Optional[list[str]] = None)
def fit(self, X: pd.DataFrame, y=None) -> "PlaceholderNullNormalizer"
def transform(self, X: pd.DataFrame) -> pd.DataFrame
```

- **Does**: converts placeholder "fake null" strings (e.g. `"UNKNOWN"`,
  `"ERROR"`, `"N/A"`) into real `NaN` on object/string-dtype columns, so
  `MissingValueImputer` — which only recognizes true `NaN`/null — picks
  them up correctly instead of treating them as legitimate categories.
  Runs before `MissingValueImputer` in `DataCleaningAgent`'s pipeline.
  Matching is case-insensitive and whitespace-stripped, but requires an
  EXACT match — a value is never swapped for containing a placeholder as
  a substring, so a genuine short value (e.g. a real category that
  happens to be `"-"`) is only affected when it equals a listed
  placeholder exactly. Numeric columns are never touched.
- **Params**: `placeholder_values: list[str] | None = None` — defaults to
  `["unknown", "error", "n/a", "na", "none", "null", "-", "--", ""]`.
- **Output**: a `pd.DataFrame` with matching placeholder values replaced
  by `NaN`. Fitted attribute `placeholder_counts_: dict` (col → count of
  placeholder values converted in that column).
- **Leakage note**: `fit` is a no-op — which strings count as
  placeholders is a fixed, caller-supplied list, not a statistic learned
  from data, so this transformer is leakage-safe by construction.
- **Tested in**: `tests/test_cleaner.py::test_placeholder_normalizer_converts_known_placeholders_to_nan`,
  `::test_placeholder_normalizer_then_imputer_fills_placeholders_with_mode`,
  `::test_placeholder_normalizer_is_case_insensitive`,
  `::test_placeholder_normalizer_leaves_non_matching_values_alone`,
  `::test_placeholder_normalizer_leaves_numeric_columns_untouched`,
  `::test_placeholder_normalizer_supports_custom_placeholder_list`.

### `class MissingValueImputer(BaseEstimator, TransformerMixin)`

```python
def __init__(self, high_missing_threshold: float = 0.5)
def fit(self, X: pd.DataFrame, y=None) -> "MissingValueImputer"
def transform(self, X: pd.DataFrame) -> pd.DataFrame
```

- **Does**: imputes missing values — median for numeric columns, mode
  for categorical columns (falls back to `"unknown"` if a column's mode
  is empty). Also flags (does not drop) columns whose missing fraction
  exceeds the threshold.
- **Params**: `high_missing_threshold: float = 0.5` — missing-value
  fraction above which a column is flagged in `high_missing_columns_`.
- **Output**: a `pd.DataFrame` with nulls filled. Fitted attributes:
  `numeric_fill_values_: dict` (col → median), `categorical_fill_values_:
  dict` (col → mode), `high_missing_columns_: list[str]`.
- **Leakage note**: this is the clearest instance of the module-wide
  contract — medians/modes are computed once in `fit` from training
  data only and stored, then the *same* stored values are applied by
  `transform` to any other data (validation/test), so no test-set value
  ever influences an imputed fill.
- **Tested in**: `tests/test_cleaner.py::test_imputer_fills_numeric_with_median`,
  `::test_imputer_fills_categorical_with_mode`,
  `::test_imputer_flags_high_missing_columns`,
  `::test_imputer_learns_stats_only_on_fit_data_not_transform_data`
  (the leakage-prevention test specifically),
  `::test_imputer_handles_all_null_column`.

### `class DataTypeCorrector(BaseEstimator, TransformerMixin)`

```python
def __init__(self, columns_to_check: Optional[list[str]] = None)
def fit(self, X: pd.DataFrame, y=None) -> "DataTypeCorrector"
def transform(self, X: pd.DataFrame) -> pd.DataFrame
```

- **Does**: coerces object-dtype columns that look numeric (e.g. money
  stored as text) into proper numeric dtype via `pd.to_numeric(...,
  errors="coerce")`. A column converts only if more than 90% of its
  non-blank values parse as numeric; blank/whitespace/`"nan"`/`"None"`/
  `"NaN"` strings are treated as already-missing, not as values that
  would be destroyed by conversion. Columns that don't clear the 90%
  bar are left untouched.
- **Params**: `columns_to_check: list[str] | None = None` — which
  columns to attempt; `None` means every non-numeric column (`X.select_dtypes(exclude=[np.number])`).
- **Output**: a `pd.DataFrame` with qualifying columns converted.
  Fitted/computed attribute `converted_columns_: list[str]` — recomputed
  on every `transform` call (not just `fit`), since conversion depends
  only on the column's own content, not on any cross-row training
  statistic.
- **Leakage note**: `fit` is a no-op; the 90%-parse-rate decision is
  re-evaluated independently on whatever `X` is passed to `transform`
  each time — there's no cross-split statistic to leak, since the
  decision is per-column-content, not learned from training data.
- **Tested in**: `tests/test_cleaner.py::test_type_corrector_converts_numeric_like_text_column`,
  `::test_type_corrector_leaves_genuine_text_column_untouched`.

### `class LowVarianceColumnDropper(BaseEstimator, TransformerMixin)`

```python
def __init__(self, threshold: float = 0.99)
def fit(self, X: pd.DataFrame, y=None) -> "LowVarianceColumnDropper"
def transform(self, X: pd.DataFrame) -> pd.DataFrame
```

- **Does**: flags and drops columns where a single value accounts for
  `>= threshold` fraction of all rows (near-constant, low signal).
- **Params**: `threshold: float = 0.99`.
- **Output**: a `pd.DataFrame` with flagged columns dropped. Fitted
  attribute `dropped_columns_: list[str]`.
- **Leakage note**: which columns count as "near-constant" is decided
  once in `fit` on training data only; `transform` just drops whatever
  columns from that list are present, without re-evaluating variance on
  the data being transformed.
- **Tested in**: `tests/test_cleaner.py::test_low_variance_dropper_drops_near_constant_column`,
  `::test_low_variance_dropper_keeps_normal_variance_column`.

---

## `src/tools/feature_tools.py`

Feature-engineering transformers used by `FeatureEngineeringAgent`. Same
sklearn `fit`/`transform` interface and leakage contract as
`data_tools.py`. All four classes additionally respect a
`protected_cols` parameter — columns that must never be modified,
defaulting to module-level `PROTECTED_COLS: frozenset[str] =
frozenset(["is_late_delivery", "order_id"])` (this project's own
target/ID columns; callers pass their own via `protected_cols` for a
different dataset).

### `class RedundantFeatureDropper(BaseEstimator, TransformerMixin)`

```python
def __init__(
    self,
    threshold: float = 0.95,
    protected_cols: Optional[list[str]] = None,
)
def fit(self, X: pd.DataFrame, y=None) -> "RedundantFeatureDropper"
def transform(self, X: pd.DataFrame) -> pd.DataFrame
```

- **Does**: for every pair of numeric columns whose absolute Pearson
  correlation exceeds `threshold`, drops the second-encountered column
  and keeps the first. Protected columns are never drop candidates,
  even if highly correlated with something else.
- **Params**: `threshold: float = 0.95`; `protected_cols: list[str] |
  None = None` (defaults to `PROTECTED_COLS`).
- **Output**: a `pd.DataFrame` with redundant columns dropped. Fitted
  attribute `dropped_columns_: list[str]`.
- **Leakage note**: the correlation matrix (and therefore which columns
  are "redundant") is computed once in `fit` on training data only;
  `transform` just drops that fixed column list from whatever data it's
  given.
- **Tested in**: `tests/test_feature_engineer.py::test_redundant_dropper_drops_correlated_column`,
  `::test_redundant_dropper_keeps_uncorrelated_columns`,
  `::test_redundant_dropper_never_drops_protected_columns`,
  `::test_redundant_dropper_keeps_first_of_correlated_pair`,
  `::test_redundant_dropper_single_column_does_not_crash`.

### `class SkewnessReducer(BaseEstimator, TransformerMixin)`

```python
def __init__(
    self,
    skew_threshold: float = 1.0,
    protected_cols: Optional[list[str]] = None,
)
def fit(self, X: pd.DataFrame, y=None) -> "SkewnessReducer"
def transform(self, X: pd.DataFrame) -> pd.DataFrame
```

- **Does**: applies `np.log1p` to numeric columns whose absolute
  skewness exceeds `skew_threshold`, provided the column's minimum is
  `>= 0` (log1p is undefined for `x < -1` and misleading for mixed-sign
  data, so negative-valued columns are skipped regardless of skew).
- **Params**: `skew_threshold: float = 1.0`; `protected_cols: list[str]
  | None = None` (defaults to `PROTECTED_COLS`).
- **Output**: a `pd.DataFrame` with qualifying columns log1p-transformed.
  Fitted attribute `log_transformed_columns_: list[str]`.
- **Leakage note**: which columns qualify (skewness + sign check) is
  decided once in `fit` on training data; `transform` applies `log1p`
  only to that fixed column list, never re-deciding based on the data
  it's given.
- **Tested in**: `tests/test_feature_engineer.py::test_skewness_reducer_lowers_skewness`,
  `::test_skewness_reducer_skips_column_with_negative_values`,
  `::test_skewness_reducer_skips_low_skew_column`,
  `::test_skewness_reducer_never_transforms_protected_columns`,
  `::test_skewness_reducer_transform_uses_fit_columns_only` (the
  leakage-prevention test specifically).

### `class NumericScaler(BaseEstimator, TransformerMixin)`

```python
def __init__(self, protected_cols: Optional[list[str]] = None)
def fit(self, X: pd.DataFrame, y=None) -> "NumericScaler"
def transform(self, X: pd.DataFrame) -> pd.DataFrame
```

- **Does**: `StandardScaler`-style zero-mean/unit-variance
  normalization: `(x - mean) / std`. Columns with zero variance are
  skipped (avoids division by zero) rather than scaled.
- **Params**: `protected_cols: list[str] | None = None` (defaults to
  `PROTECTED_COLS`).
- **Output**: a `pd.DataFrame` with numeric columns scaled. Fitted
  attributes: `scale_stats_: dict[str, dict[str, float]]` (col →
  `{"mean": ..., "std": ...}`), `scaled_columns_: list[str]`,
  `zero_variance_columns_: list[str]`.
- **Leakage note**: mean/std are computed once in `fit` on training data
  and stored in `scale_stats_`; `transform` applies those exact stored
  values to any other data, never recomputing mean/std from the data
  being transformed — the textbook leakage-prevention pattern for a
  scaler.
- **Tested in**: `tests/test_feature_engineer.py::test_scaler_fit_transform_produces_zero_mean_unit_std`,
  `::test_scaler_leakage_prevention` (the leakage-prevention test
  specifically), `::test_scaler_skips_zero_variance_column`,
  `::test_scaler_never_scales_protected_columns`.

### `class CategoricalEncoder(BaseEstimator, TransformerMixin)`

```python
def __init__(
    self,
    ohe_threshold: int = 20,
    protected_cols: Optional[list[str]] = None,
)
def fit(self, X: pd.DataFrame, y=None) -> "CategoricalEncoder"
def transform(self, X: pd.DataFrame) -> pd.DataFrame
```

- **Does**: encodes categorical columns with one of two strategies
  chosen per-column by cardinality:
  - **One-hot** (cardinality `< ohe_threshold`): `pd.get_dummies`.
    Categories unseen at fit time get all-zero dummies at transform
    time (no new dummy columns are created).
  - **Frequency** (cardinality `>= ohe_threshold`): each value replaced
    by its `count / n_rows` proportion from fit data, landing in
    `[0, 1]`. Values unseen at fit time map to `0.0`.
  The original categorical column is dropped and replaced by its
  encoded form.
- **Params**: `ohe_threshold: int = 20` — cardinality cutoff between the
  two strategies; `protected_cols: list[str] | None = None` (defaults
  to `PROTECTED_COLS`).
- **Output**: a `pd.DataFrame` with categorical columns replaced by
  their encoded form. Fitted attributes: `ohe_columns_: dict[str,
  list[str]]` (col → dummy column names), `freq_columns_: dict[str,
  dict]` (col → `{value: proportion}`), `encoding_map_: dict[str, str]`
  (col → `"one_hot"` or `"frequency"`).
- **Leakage note**: both which strategy applies and the actual encoding
  values (dummy column set, frequency proportions) are fixed once in
  `fit` on training data; `transform` reindexes one-hot output to the
  fit-time dummy columns (`fill_value=0` for anything new) and maps
  frequency values through the fit-time lookup (`fillna(0.0)` for
  anything unseen) — new categories at transform time never leak a
  transform-time proportion or create a transform-time dummy column.
- **Tested in**: `tests/test_feature_engineer.py::test_ohe_encoder_chosen_for_low_cardinality`,
  `::test_frequency_encoder_chosen_for_high_cardinality`,
  `::test_frequency_encoder_correct_proportions`,
  `::test_ohe_encoder_consistent_columns_on_transform`,
  `::test_ohe_encoder_never_encodes_protected_columns`,
  `::test_frequency_encoder_maps_unseen_to_zero`.

---

## `src/tools/ml_tools.py`

Small, pure (side-effect-free) helper functions used by `MLAgent`. No
classes here — every function is independent and trivially testable in
isolation.

### `def detect_task_type(y: pd.Series, multiclass_unique_threshold: int = 20) -> str`

- **Does**: infers whether a target column needs binary classification,
  multiclass classification, or regression, via three ordered rules:
  (1) exactly 2 unique non-null values → `"binary_classification"`;
  (2) integer or object dtype AND unique count `<=
  multiclass_unique_threshold` → `"multiclass_classification"`;
  (3) otherwise → `"regression"`.
- **Params**: `y: pd.Series` — the target column; `multiclass_unique_threshold:
  int = 20` — max distinct values before falling through to regression.
- **Returns**: `str`, one of `"binary_classification"`,
  `"multiclass_classification"`, `"regression"`.
- **Tested in**: `tests/test_ml_agent.py::test_detect_binary_classification`,
  `::test_detect_multiclass_classification_integer`,
  `::test_detect_multiclass_classification_object`,
  `::test_detect_regression_continuous`,
  `::test_detect_regression_when_many_integers`,
  `::test_detect_uses_threshold_parameter`.

### `def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float`

- **Does**: computes adjusted R² from plain R², sample count, and
  feature count: `1 - (1 - r2) * (n_samples - 1) / (n_samples -
  n_features - 1)`. Returns `float("nan")` when the denominator would
  be `<= 0` (i.e. `n_samples <= n_features + 1`).
- **Params**: `r2: float`, `n_samples: int`, `n_features: int`.
- **Returns**: `float` (or `nan` in the degenerate case above).
- **Tested in**: `tests/test_ml_agent.py::test_adjusted_r2_perfect_fit`,
  `::test_adjusted_r2_worse_than_r2_with_many_features`,
  `::test_adjusted_r2_returns_nan_when_degenerate`.

### `def expected_calibration_error(prob_true: list, prob_pred: list, bin_counts: list) -> float`

- **Does**: weighted mean absolute gap between observed and predicted
  probability across calibration bins, weighted by each bin's sample
  count — a single scalar summary of a reliability diagram (0 =
  perfectly calibrated, larger = worse). Weighting by bin count means a
  bin with more test samples counts more toward the final number than a
  sparse one.
- **Params**: `prob_true: list` (observed frequency per bin),
  `prob_pred: list` (mean predicted probability per bin), `bin_counts:
  list` (sample count per bin) — all three same length, one entry per
  calibration bin.
- **Returns**: `float`. Returns `0.0` when `sum(bin_counts) == 0`.
- **Tested in**: `tests/test_ml_agent.py::test_expected_calibration_error_zero_when_perfectly_calibrated`,
  `::test_expected_calibration_error_weighted_by_bin_count`,
  `::test_expected_calibration_error_empty_bins_returns_zero`.

---

## `src/tools/audit_db.py`

The SQLite audit-trail layer (schema fully documented in
[`TECHNICAL_SPEC.md` Section 6.1](TECHNICAL_SPEC.md#61-sqlite-audit-trail-srctoolsauditdbpy));
this entry covers the callable API only.

### `def init_db(db_path: Optional[str] = None) -> None`

- **Does**: creates the `agent_runs` and `ml_experiments` tables if they
  don't exist (`CREATE TABLE IF NOT EXISTS`) — idempotent, safe to call
  repeatedly.
- **Params**: `db_path: str | None = None` — defaults to module-level
  `DEFAULT_DB_PATH` (`workspace/metadata/audit_telemetry.db`),
  re-resolved fresh on every call rather than baked in, so tests can
  monkeypatch `DEFAULT_DB_PATH` and every caller that omits `db_path`
  picks up the override.
- **Returns**: `None`.
- **Tested in**: `tests/test_audit_db.py::test_init_db_creates_table_with_expected_columns`,
  `::test_init_db_is_idempotent`, `::test_init_db_creates_parent_directory`,
  `::test_init_db_creates_ml_experiments_table_with_expected_columns`.

### `def log_agent_run(agent_name: str, started_at: datetime, finished_at: datetime, status: str, input_path: Optional[str], output_path: Optional[str] = None, error_message: Optional[str] = None, db_path: Optional[str] = None) -> None`

- **Does**: inserts one row into `agent_runs`. Calls `init_db()` first,
  so it's safe standalone even before the DB/table exists.
- **Params**: `agent_name: str`; `started_at`, `finished_at: datetime`
  (duration is computed as their difference); `status: str`
  (`"success"`/`"failure"`); `input_path: str | None`; `output_path:
  str | None = None`; `error_message: str | None = None`; `db_path: str
  | None = None`.
- **Returns**: `None`.
- **Tested in**: `tests/test_audit_db.py::test_log_agent_run_inserts_and_get_recent_runs_retrieves`,
  `::test_log_agent_run_records_failure_with_error_message`.

### `def get_recent_runs(limit: int = 50, db_path: Optional[str] = None) -> list[dict]`

- **Does**: returns the most recent `limit` rows from `agent_runs`,
  newest first (`ORDER BY started_at DESC, id DESC` — the `id` tiebreak
  keeps ordering well-defined for sub-second-identical timestamps).
- **Params**: `limit: int = 50`; `db_path: str | None = None`.
- **Returns**: `list[dict]`, one dict per row with keys `id,
  agent_name, started_at, finished_at, status, input_path, output_path,
  error_message, duration_seconds`.
- **Tested in**: `tests/test_audit_db.py::test_get_recent_runs_orders_most_recent_first`,
  `::test_get_recent_runs_respects_limit` (plus indirectly by every
  `log_agent_run` test, which reads back via this function).

### `def log_ml_experiment(data_path: str, target_col: str, task_type: str, best_model_name: str, best_hyperparameters: dict, cv_scores: dict, cv_std: dict, test_metrics: dict, split_strategy: Optional[str] = None, group_col: Optional[str] = None, random_state: Optional[int] = None, n_features: Optional[int] = None, model_selection_note: Optional[str] = None, nested_cv_score: Optional[float] = None, nested_cv_std: Optional[float] = None, report_path: Optional[str] = None, db_path: Optional[str] = None) -> None`

- **Does**: inserts one row into `ml_experiments`, capturing what a
  successful `MLAgent.run()` produced. Dict-valued params
  (`best_hyperparameters`, `cv_scores`, `cv_std`, `test_metrics`) are
  stored as JSON text via `json.dumps`.
- **Params**: as listed above — every dict field is passed through
  exactly as `MLReport` produced it, with no hardcoded metric-name
  assumptions in the schema.
- **Returns**: `None`.
- **Tested in**: `tests/test_audit_db.py::test_log_ml_experiment_inserts_and_get_recent_experiments_retrieves`,
  `::test_log_ml_experiment_round_trips_dict_fields_as_json`,
  `::test_log_ml_experiment_handles_missing_optional_fields`.

### `def get_recent_experiments(limit: int = 50, db_path: Optional[str] = None) -> list[dict]`

- **Does**: returns the most recent `limit` rows from `ml_experiments`,
  newest first (same `ORDER BY ... DESC, id DESC` tiebreak pattern as
  `get_recent_runs`), with the JSON text columns (`best_hyperparameters,
  cv_scores, cv_std, test_metrics`) decoded back into dicts via
  `json.loads`.
- **Params**: `limit: int = 50`; `db_path: str | None = None`.
- **Returns**: `list[dict]`, one dict per row, JSON columns pre-decoded.
- **Tested in**: `tests/test_audit_db.py::test_get_recent_experiments_orders_most_recent_first`,
  `::test_get_recent_experiments_respects_limit`,
  `::test_get_recent_experiments_empty_when_nothing_logged`.

### `def audit_logged(agent_name: str, input_arg: Union[str, tuple[str, ...]] = "data_path", db_path: Optional[str] = None) -> Callable`

- **Does**: decorator factory. Wraps an agent's `run()` method to log one
  `agent_runs` row per call — success or failure — without changing the
  wrapped method's behavior: its `(success, message)` return value (or a
  re-raised exception) passes through unchanged; the decorator only
  observes timing and outcome via `inspect.signature(func).bind_partial`
  to resolve the input-path argument(s) by name.
- **Params**: `agent_name: str` — recorded in the logged row;
  `input_arg: str | tuple[str, ...] = "data_path"` — name(s) of the
  decorated method's parameter(s) identifying its input path(s); a
  tuple is joined with `"; "` for agents reading multiple input files
  (e.g. `VisualizationAgent`'s three report paths); `db_path: str |
  None = None` — passed through to `log_agent_run` per call, left
  unresolved at decoration time so tests can monkeypatch
  `DEFAULT_DB_PATH` after import.
- **Returns**: `Callable` — the wrapped function.
- **Tested in**: `tests/test_audit_db.py::test_audit_logged_logs_success`,
  `::test_audit_logged_logs_failure_without_raising`,
  `::test_audit_logged_logs_and_reraises_unexpected_exception`,
  `::test_audit_logged_supports_multiple_input_args`,
  `::test_audit_logged_does_not_alter_return_value`. Also exercised
  end-to-end by every agent's own test suite, since every agent's
  `run()` is decorated with it.

---

## `src/tools/logging_config.py`

Shared logging setup — no fit/transform contract here (not a
transformer). Every agent module also calls `get_agent_logger` once at
import time, so any test importing an agent module exercises basic
construction incidentally — but that incidental path only ever requests
each real agent's name once per process, so it never actually reaches
the duplicate-handler guard below. That guard is real, load-bearing
behavior (skipping it would double every log line on re-import), so it
gets a dedicated test rather than relying on incidental coverage alone.

### `def get_agent_logger(agent_name: str) -> logging.Logger`

- **Does**: returns a configured `logging.Logger` for a given agent
  name, attaching two handlers on first call — a console
  `StreamHandler` and a `FileHandler` writing to
  `workspace/metadata/agent_activity.log` — both using the shared
  format `"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"`.
  Guards against duplicate handlers on re-import (`if logger.handlers:
  return logger`), since Python caches loggers by name and a naive
  re-configuration would otherwise double every log line on repeated
  imports.
- **Params**: `agent_name: str` — becomes the logger's name and the
  `%(name)s` field in every line it writes.
- **Returns**: `logging.Logger`, level `INFO`, already attached to
  console + shared file handlers.
- **Tested in**: `tests/test_logging_config.py::test_get_agent_logger_attaches_console_and_file_handlers`,
  `::test_get_agent_logger_writes_to_file_with_expected_format`,
  `::test_get_agent_logger_does_not_duplicate_handlers_on_repeat_calls`
  (the duplicate-handler guard specifically). Each test monkeypatches
  the module-level `LOG_FILE` constant to a `tmp_path` location and uses
  a logger name no real agent uses, to avoid colliding with handlers a
  prior import already attached to that name; handlers are closed and
  removed in a `finally` block afterward so no test leaks a `FileHandler`
  pointed at a `tmp_path` pytest has since cleaned up. Also still
  exercised incidentally by every agent module import, as before.
