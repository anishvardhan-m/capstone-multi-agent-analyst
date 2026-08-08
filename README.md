# Multi-Agent AI Data Analyst

A multi-agent system that takes any tabular dataset (CSV) and autonomously
cleans it, explores it, engineers features, trains and evaluates ML models
(classification or regression), generates visualizations, writes business
insights, and compiles an executive PDF report — end to end, through a
self-correcting orchestration layer that recovers from agent failures
automatically. An 8-page Streamlit dashboard sits on top for interactive,
step-by-step or one-click runs.

Capstone project — 6 Month Internship in Data Science, AI & ML
(Techible x IIT Jammu).

**Status:** Feature-complete — all 8 agents, the orchestrator, and the
dashboard are built and tested (384 passing tests), including a dedicated
rigor review (data leakage safety, calibration, seed-robustness, segment
error analysis — see [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)).

## Key features

- **8 specialized agents** — Data Cleaning, EDA, Feature Engineering, ML
  Training, Visualization, Business Insights, Report Generation, and an
  Orchestrator that chains the rest
- **Self-correcting orchestration** — the Orchestrator Agent detects
  per-step failures and applies an LLM-driven retry/skip/abort decision
  rather than crashing the whole pipeline
- **Dataset-agnostic (genericity)** — works on any classification or
  regression CSV, not just the demo dataset; domain labels are
  configurable, not hardcoded
- **Real LLM integration** — the Business Insights Agent makes genuine
  calls to an LLM (via OpenRouter) with a two-model fallback, grounded
  directly in the model's actual performance numbers
- **Full audit trail** — every agent run is logged to a SQLite telemetry
  DB and a structured activity log, browsable from the dashboard
- **8-page dashboard** — ingestion, EDA, ML Studio, visualization
  gallery, insights panel, reports hub, system log explorer, and run
  history
- **Rigorous ML validation** — leakage-safe grouped train/test splitting,
  probability calibration, seed-robustness checks, and segment-level
  error analysis (see [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for
  what this surfaced on the demo model)

## Quick start

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd capstone-multi-agent-analyst
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# then edit .env and set OPENAI_API_KEY to your OpenRouter API key
# (OPENAI_API_BASE_URL should point at OpenRouter's endpoint)
```

**Run the dashboard:**

```bash
streamlit run app/home.py
```

**Run the full pipeline directly (no UI):**

```bash
python -m src.agents.orchestrator <path_to_csv> <target_col> [options]

# Olist demo:
python -m src.agents.orchestrator data/processed/olist_flattened.csv is_late_delivery \
  --group-col customer_unique_id \
  --positive-label "late delivery" --negative-label "on-time delivery" --unit-label order
```

Run `python -m src.agents.orchestrator --help` for the full option list
(`--id-col`, `--group-col`, `--positive-label`, `--negative-label`,
`--unit-label`).

**Run the tests:**

```bash
pytest tests/
```

## Running with Docker

The dashboard can also be run in a container (capstone handbook Section
16.3) — this bundles Python, WeasyPrint's system libraries (Pango, cairo,
GObject), and all Python dependencies into one image, so there's no local
install step beyond Docker itself.

```bash
# 1. Build the image
docker build -t ai-data-analyst .

# 2. Run it, passing your .env file at runtime
#    (.env is gitignored and is NOT baked into the image — supply it with
#    --env-file, or use -e KEY=VALUE for individual variables instead)
docker run -p 8501:8501 --env-file .env ai-data-analyst
```

Then open http://localhost:8501 in a browser.

Notes:

- `OPENAI_API_KEY` (and the other variables in `.env.example`) must be
  supplied via `--env-file .env` or `-e` flags — the container has no
  secrets baked in.
- `workspace/`, `data/raw/`, `data/processed/`, and `models/` are runtime
  artifacts, not baked into the image (see `.dockerignore`); mount them as
  volumes (e.g. `-v $(pwd)/workspace:/app/workspace`) if you want results
  to persist outside the container.
- **A freshly built container starts with an empty `workspace/`** — no
  demo charts, executive report, or audit log, since those are excluded
  from the image by `.dockerignore`. The dashboard handles this
  gracefully (every page shows a "run the pipeline first" message rather
  than erroring), but pages 5–9 (Visualization Gallery, Insights Panel,
  Reports Hub, System Log Explorer, Run History) will show no data until
  the pipeline is run once from the **Dataset Ingestion** page.
  - If you're running Docker on the same machine that already has the
    Olist demo's `workspace/` populated locally, mount it instead of
    starting empty: add `-v $(pwd)/workspace:/app/workspace` to the
    `docker run` command above to see the existing results immediately
    without re-running anything.

## Repository structure

```
capstone-multi-agent-analyst/
├── app/                          # Streamlit multi-page dashboard
│   ├── home.py                   # Entry point (streamlit run app/home.py)
│   ├── dashboard_helpers.py      # Shared UI/session-state helpers
│   └── pages/                    # 8 dashboard pages (ingestion, EDA, ML
│                                  #   Studio, visualization gallery,
│                                  #   insights panel, reports hub, system
│                                  #   log explorer, run history)
├── src/
│   ├── agents/                   # The 8 specialized agents
│   │   ├── cleaner.py            # Data Cleaning Agent
│   │   ├── eda.py                # EDA Agent
│   │   ├── feature_engineer.py   # Feature Engineering Agent
│   │   ├── ml_agent.py           # ML Training/Evaluation Agent
│   │   ├── visualizer.py         # Visualization Agent
│   │   ├── insights.py           # Business Insights Agent (LLM)
│   │   ├── report_generator.py   # Report Generation Agent (PDF)
│   │   └── orchestrator.py       # Orchestrator Agent (chains all 7)
│   └── tools/                    # Shared tool functions agents call
│       ├── data_tools.py
│       ├── feature_tools.py
│       ├── ml_tools.py
│       ├── audit_db.py           # SQLite audit trail
│       └── logging_config.py
├── data/
│   ├── raw/                      # Raw input CSVs (gitignored)
│   ├── processed/                # Flattened/cleaned CSVs (gitignored)
│   └── flatten_olist.py          # One-time Olist join/flatten script
├── models/                       # Serialized trained model artifacts
├── workspace/                    # Runtime outputs (charts, logs, audit DB,
│                                  #   generated reports)
└── tests/                        # Unit and integration tests (pytest)
```

## Demo dataset: Brazilian E-Commerce (Olist)

The platform itself is dataset-agnostic — it works on any CSV a user
uploads. For our demo, we use the [Olist public e-commerce
dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
flattened from its 9 raw relational tables into a single order-level CSV,
predicting **late delivery risk at the moment of purchase**.

### Reproducing the flattened dataset

1. Download the 9 Olist CSVs and place them in `data/raw/`
2. Run:
   ```bash
   python data/flatten_olist.py
   ```
3. Output appears in `data/processed/`:
   - `olist_flattened.csv` — the modeling dataset (leakage-free features only)
   - `olist_undelivered_orders_audit.csv` — orders never delivered/cancelled, kept for transparency
   - `olist_review_analysis_only.csv` — review scores, kept separate since they are only known *after* delivery and would leak into the late-delivery target if used as a feature

**Note on leakage:** the target (`is_late_delivery`) is computed from the
actual delivery date, but the *feature set* only includes information
knowable at purchase time (customer/product/seller info, price, freight,
payment method, estimated delivery window). See the docstring in
`data/flatten_olist.py` for the full rationale.

## Known limitations

The Olist model's rigor testing (calibration, seed robustness, error
analysis across segments) surfaced real, specific limitations — not
generic ML disclaimers. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)
before treating the demo model's numbers as production-ready.

## Testing

405 tests across unit and integration suites, covering every agent plus
the orchestrator's failure-recovery paths.

```bash
pytest tests/
```
