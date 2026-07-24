# Multi-Agent AI Data Analyst

A multi-agent system that takes any tabular dataset (CSV) and autonomously
cleans it, explores it, engineers features, trains and evaluates ML models,
generates visualizations, writes business insights, and compiles an
executive PDF report — with a self-correcting orchestration layer that
recovers from agent errors automatically.

Capstone project — 6 Month Internship in Data Science, AI & ML
(Techible x IIT Jammu).

## Demo dataset: Brazilian E-Commerce (Olist)

The platform itself is dataset-agnostic — it works on any CSV a user
uploads. For our demo, we use the [Olist public e-commerce
dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
flattened from its 9 raw relational tables into a single order-level CSV,
predicting **late delivery risk at the moment of purchase**.

### Reproducing the flattened dataset

1. Download the 9 Olist CSVs and place them in `data/raw/`
2. Run:
   ```
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

## Project status

Build in progress. See commit history for stage-by-stage progress.

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # then fill in your API key
```

## Repository structure

```
capstone-multi-agent-analyst/
├── app/                    # Streamlit multi-page dashboard
├── src/
│   ├── agents/             # The 8 specialized agents
│   └── tools/               # Shared tool functions agents call
├── data/
│   ├── raw/                 # Raw input CSVs (gitignored)
│   ├── processed/            # Flattened/cleaned CSVs (gitignored)
│   └── flatten_olist.py     # One-time Olist join/flatten script
├── models/                  # Serialized trained model artifacts
├── workspace/                # Runtime outputs (charts, logs, audit DB)
└── tests/                    # Unit and integration tests
```
