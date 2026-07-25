"""
src/agents/report_generator.py

Report Generation Agent.

Responsibilities (per the capstone handbook, Sections 6.8, 12.1, 12.2):
compile every upstream artifact -- the cleaning report, the ML report, the
Business Insights Agent's markdown narrative, and the chart images -- into
a single executive PDF, via HTML templating rendered through WeasyPrint.

Design note: unlike the Business Insights Agent, this is NOT an LLM stage.
It is pure deterministic templating: build an HTML string from structured
inputs, then render it to PDF with
``weasyprint.HTML(string=html).write_pdf(path)`` (handbook Section 6.8's
pseudocode pattern). Reproducibility matters here -- the same four inputs
must always produce the same PDF.

The PDF has exactly five sections (handbook Section 12.1):
  1. Executive Summary            <- business_insights.md, "What We Found"
  2. Data Diagnostics Profile     <- cleaning report
  3. Model Performance Leaderboard<- ML report (branches on task_type)
  4. Automated Business Insights  <- business_insights.md, "What Matters
                                      Most", plus the embedded charts
  5. Strategic Recommendations    <- business_insights.md, "Recommendations"

Graceful degradation: a missing/unreadable cleaning report, ML report,
insights markdown, or chart file never crashes the whole report -- that
one piece is skipped/placeholder-ed and a warning is logged and recorded
on the returned report, matching every other agent's philosophy of
producing *something* auditable rather than failing outright on partial
upstream failures.
"""

from __future__ import annotations

import base64
import html as html_escape
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# WeasyPrint's cffi bindings for Pango/cairo/GObject don't reliably find
# Homebrew's dylibs on macOS unless DYLD_LIBRARY_PATH already points at
# Homebrew's lib dir -- macOS strips DYLD_* from the shell environment by
# default (SIP), so set it here, before importing weasyprint, rather than
# requiring every caller to remember to export it themselves.
#
# This ONLY fixes the linker-search-path problem on a machine that already
# has Pango/cairo/GObject installed via Homebrew somewhere -- it does not
# install them. A machine that has never run `brew install pango` (e.g. a
# presentation laptop this project has never touched) will still fail to
# import weasyprint; see the actionable error raised below for that case.
if sys.platform == "darwin":
    _brew_libs = ["/opt/homebrew/lib", "/usr/local/lib"]  # ARM / Intel defaults
    try:
        import subprocess
        _brew_prefix = subprocess.run(
            ["brew", "--prefix"], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if _brew_prefix:
            _brew_libs.append(os.path.join(_brew_prefix, "lib"))
    except Exception:
        pass  # brew not on PATH, or not installed -- fall back to the defaults above

    for _brew_lib in _brew_libs:
        if os.path.isdir(_brew_lib):
            _existing = os.environ.get("DYLD_LIBRARY_PATH", "")
            if _brew_lib not in _existing.split(":"):
                os.environ["DYLD_LIBRARY_PATH"] = (
                    f"{_brew_lib}:{_existing}" if _existing else _brew_lib
                )

try:
    from weasyprint import HTML  # noqa: E402
except OSError as exc:
    raise RuntimeError(
        "WeasyPrint could not load its native Pango/cairo/GObject libraries. "
        "This is a system dependency, not a Python package -- on macOS run "
        "`brew install pango` (installs cairo/gobject-introspection too), on "
        "Debian/Ubuntu run `apt install libpango-1.0-0 libpangoft2-1.0-0`, "
        "then retry. See "
        "https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation "
        f"for other platforms. Original error: {exc}"
    ) from exc

from src.tools.audit_db import audit_logged
from src.tools.logging_config import get_agent_logger

logger = get_agent_logger("ReportGenerationAgent")

_DEFAULT_OUTPUT_PATH = "workspace/executive_report.pdf"


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReportGenerationReport:
    """Structured summary of one report-generation run."""

    output_path: str
    sections_included: list = field(default_factory=list)
    charts_embedded: list = field(default_factory=list)
    charts_skipped: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "sections_included": self.sections_included,
            "charts_embedded": self.charts_embedded,
            "charts_skipped": self.charts_skipped,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Markdown parsing (business_insights.md -> 3 content slots)
# ---------------------------------------------------------------------------

_SECTION_KEYWORDS = {
    "executive_summary": ("found",),
    "business_insights": ("matter",),
    "recommendations": ("recommend",),
}


def _parse_insights_markdown(md_text: str) -> list[tuple[str, str]]:
    """Split markdown text on '## ' headers into ordered (header, body) pairs."""
    sections: list[tuple[str, str]] = []
    current_header: Optional[str] = None
    current_lines: list[str] = []

    for line in md_text.splitlines():
        if line.startswith("## "):
            if current_header is not None:
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_header is not None:
        sections.append((current_header, "\n".join(current_lines).strip()))

    return sections


def _map_insights_sections(sections: list[tuple[str, str]]) -> dict[str, Optional[str]]:
    """Map parsed (header, body) pairs onto the 3 PDF content slots.

    Tries keyword matching on header text first (robust to minor wording
    drift from the LLM -- e.g. "## What We Found" contains "found"); falls
    back to positional order (1st/2nd/3rd section) for any slot keyword
    matching didn't fill, so a malformed or unusually-worded markdown file
    still produces *something* for each slot rather than leaving it empty.
    """
    result: dict[str, Optional[str]] = {
        "executive_summary": None, "business_insights": None, "recommendations": None,
    }
    used_indices: set[int] = set()

    for slot, keywords in _SECTION_KEYWORDS.items():
        for i, (header, _body) in enumerate(sections):
            if i in used_indices:
                continue
            if any(kw in header.lower() for kw in keywords):
                result[slot] = sections[i][1]
                used_indices.add(i)
                break

    remaining_slots = [s for s in result if result[s] is None]
    remaining_sections = [i for i in range(len(sections)) if i not in used_indices]
    for slot, i in zip(remaining_slots, remaining_sections):
        result[slot] = sections[i][1]
        used_indices.add(i)

    return result


# ---------------------------------------------------------------------------
# Minimal markdown -> HTML (just enough for our own LLM narrative output)
# ---------------------------------------------------------------------------

def _inline_markdown_to_html(text: str) -> str:
    """Escape text, then re-enable **bold** markers as <strong>."""
    escaped = html_escape.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def _markdown_to_html(text: Optional[str]) -> str:
    """Minimal markdown -> HTML for BusinessInsightsAgent's narrative text:
    paragraphs, **bold**, and numbered/bulleted lists. Not a general-purpose
    markdown parser -- just enough to render this project's own LLM output
    as real HTML instead of a wall of asterisks and raw newlines.
    """
    if not text or not text.strip():
        return ""

    blocks = re.split(r"\n\s*\n", text.strip())
    html_parts = []
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        if all(re.match(r"^(\d+\.|-)\s+", l) for l in lines):
            is_ordered = bool(re.match(r"^\d+\.\s+", lines[0]))
            tag = "ol" if is_ordered else "ul"
            items = [re.sub(r"^(\d+\.|-)\s+", "", l) for l in lines]
            items_html = "".join(f"<li>{_inline_markdown_to_html(i)}</li>" for i in items)
            html_parts.append(f"<{tag}>{items_html}</{tag}>")
        else:
            paragraph = " ".join(lines)
            html_parts.append(f"<p>{_inline_markdown_to_html(paragraph)}</p>")
    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# Section 2 — Data Diagnostics Profile
# ---------------------------------------------------------------------------

def _fmt_shape(shape) -> str:
    if not shape or len(shape) != 2:
        return "unknown"
    rows, cols = shape
    try:
        return f"{int(rows):,} rows &times; {int(cols)} columns"
    except (TypeError, ValueError):
        return f"{rows} rows &times; {cols} columns"


def _fmt_list(items: Optional[list]) -> str:
    return ", ".join(str(i) for i in items) if items else "None"


def _build_data_diagnostics_html(cleaning: dict) -> str:
    if not cleaning:
        return "<p class='placeholder'>Data diagnostics unavailable (cleaning report missing or unreadable).</p>"

    rows = [
        ("Input shape", _fmt_shape(cleaning.get("input_shape"))),
        ("Output shape", _fmt_shape(cleaning.get("output_shape"))),
        ("Duplicate rows removed", f"{cleaning.get('n_duplicates_removed', 0):,}"),
        ("High-missing columns flagged", _fmt_list(cleaning.get("high_missing_columns_flagged"))),
        ("Low-variance columns dropped", _fmt_list(cleaning.get("low_variance_columns_dropped"))),
        ("Columns type-corrected", _fmt_list(cleaning.get("columns_type_corrected"))),
        ("Numeric columns imputed", _fmt_list(cleaning.get("numeric_columns_imputed"))),
        ("Categorical columns imputed", _fmt_list(cleaning.get("categorical_columns_imputed"))),
    ]
    rows_html = "".join(
        f"<tr><th>{html_escape.escape(label)}</th><td>{value}</td></tr>"
        for label, value in rows
    )
    return f"<table class='diagnostics-table'>{rows_html}</table>"


# ---------------------------------------------------------------------------
# Section 3 — Model Performance Leaderboard
# ---------------------------------------------------------------------------

_REGRESSION_METRIC_LABELS = {"rmse": "RMSE", "mae": "MAE", "r2": "R²", "adjusted_r2": "Adjusted R²"}
_CLASSIFICATION_METRIC_LABELS = {
    "f1_macro": "F1 (macro)", "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)", "roc_auc": "ROC-AUC",
}


def _build_metrics_table(test_metrics: dict, labels: dict) -> str:
    rows = "".join(
        f"<tr><th>{labels.get(k, k)}</th><td>{v:.4f}</td></tr>"
        for k, v in test_metrics.items()
        if isinstance(v, (int, float))
    )
    return f"<table class='metrics-table'>{rows}</table>" if rows else ""


def _build_cv_scores_table(cv_scores: dict, best_model_name: str) -> str:
    if not cv_scores:
        return ""
    rows = "".join(
        f"<tr class=\"{'best-row' if name == best_model_name else ''}\">"
        f"<td>{html_escape.escape(name)}</td><td>{score:.4f}</td></tr>"
        for name, score in sorted(cv_scores.items(), key=lambda kv: kv[1], reverse=True)
    )
    return (
        "<table class='leaderboard-table'>"
        "<thead><tr><th>Candidate Model</th><th>CV Score</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _build_confusion_matrix_html(cm: Optional[list]) -> str:
    if not cm or len(cm) != 2 or len(cm[0]) != 2:
        return ""
    tn, fp = cm[0]
    fn, tp = cm[1]
    return (
        "<table class='confusion-table'>"
        "<thead><tr><th></th><th>Predicted Negative (0)</th><th>Predicted Positive (1)</th></tr></thead>"
        "<tbody>"
        f"<tr><th>Actual Negative (0)</th><td>{tn:,}</td><td>{fp:,}</td></tr>"
        f"<tr><th>Actual Positive (1)</th><td>{fn:,}</td><td>{tp:,}</td></tr>"
        "</tbody></table>"
    )


def _build_threshold_table_html(threshold_metrics: Optional[list]) -> str:
    if not threshold_metrics:
        return ""
    rows = "".join(
        f"<tr><td>{t['threshold']}</td><td>{t['precision_minority']:.3f}</td>"
        f"<td>{t['recall_minority']:.3f}</td><td>{t['f1_macro']:.3f}</td></tr>"
        for t in threshold_metrics
    )
    return (
        "<table class='threshold-table'>"
        "<thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1 (macro)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _build_model_performance_html(ml: dict) -> str:
    """Branch on task_type: classification gets the CV table + confusion
    matrix + threshold sweep; regression gets the CV table + RMSE/MAE/R^2
    instead. Never mixes the two vocabularies."""
    if not ml:
        return "<p class='placeholder'>Model performance data unavailable (ML report missing or unreadable).</p>"

    task_type = ml.get("task_type", "unknown")
    best_model_name = ml.get("best_model_name", "unknown")
    parts = [
        f"<p><strong>Task type:</strong> {html_escape.escape(task_type)}"
        f" &nbsp;|&nbsp; <strong>Best model:</strong> {html_escape.escape(best_model_name)}</p>"
    ]

    cv_table = _build_cv_scores_table(ml.get("cv_scores") or {}, best_model_name)
    if cv_table:
        parts.append(cv_table)

    test_metrics = ml.get("test_metrics") or {}

    if task_type == "regression":
        metrics_table = _build_metrics_table(test_metrics, _REGRESSION_METRIC_LABELS)
        if metrics_table:
            parts.append(metrics_table)
    else:
        metrics_table = _build_metrics_table(test_metrics, _CLASSIFICATION_METRIC_LABELS)
        if metrics_table:
            parts.append(metrics_table)
        cm_table = _build_confusion_matrix_html(ml.get("confusion_matrix"))
        if cm_table:
            parts.append(cm_table)
        threshold_table = _build_threshold_table_html(ml.get("threshold_metrics"))
        if threshold_table:
            parts.append(threshold_table)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chart embedding
# ---------------------------------------------------------------------------

def _chart_to_data_uri(path: str) -> Optional[str]:
    """Read a chart PNG and return it as a base64 data URI, or None on failure.

    Base64-embedding (rather than a local file:// reference) is what
    WeasyPrint handles most reliably here: HTML(string=...) has no
    filesystem base_url by default, so relative/local image paths are
    fragile to the caller's working directory, while a self-contained data
    URI always renders regardless of where the PDF is generated from.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return None
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def _build_charts_html(chart_paths: Optional[list], report: ReportGenerationReport) -> str:
    figures = []
    for path in chart_paths or []:
        data_uri = _chart_to_data_uri(path)
        if data_uri is None:
            logger.warning("Chart file missing or unreadable, skipped: %s", path)
            report.charts_skipped.append({"path": path, "reason": "missing or unreadable"})
            continue
        caption = os.path.splitext(os.path.basename(path))[0]
        caption = re.sub(r"^\d+_", "", caption).replace("_", " ").title()
        figures.append(
            "<figure class='chart-figure'>"
            f"<img src='{data_uri}' alt='{html_escape.escape(caption)}'>"
            f"<figcaption>{html_escape.escape(caption)}</figcaption>"
            "</figure>"
        )
        report.charts_embedded.append(path)

    if not figures:
        return "<p class='placeholder'>No charts available.</p>"

    return f"<div class='chart-gallery'>{''.join(figures)}</div>"


# ---------------------------------------------------------------------------
# HTML document assembly
# ---------------------------------------------------------------------------

_CSS = """
@page {
    size: A4;
    margin: 2.2cm 1.8cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 8pt;
        color: #8a94a6;
    }
}
* { box-sizing: border-box; }
body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    color: #1f2937;
    font-size: 10.5pt;
    line-height: 1.5;
}
header.cover {
    border-bottom: 3px solid #1a2b4c;
    padding-bottom: 14px;
    margin-bottom: 24px;
}
header.cover h1 {
    font-size: 22pt;
    color: #1a2b4c;
    margin: 0 0 6px 0;
}
header.cover .project-id {
    font-size: 10.5pt;
    font-weight: bold;
    color: #2c6e91;
    margin: 0;
    letter-spacing: 0.03em;
}
header.cover .generated-at {
    font-size: 9pt;
    color: #6b7280;
    margin: 2px 0 0 0;
}
section.report-section {
    margin-bottom: 22px;
}
section.report-section.page-break {
    page-break-before: always;
}
section.report-section h2 {
    font-size: 14pt;
    color: #ffffff;
    background: #1a2b4c;
    padding: 6px 12px;
    border-radius: 3px;
    margin: 0 0 12px 0;
}
p { margin: 0 0 8px 0; }
p.placeholder {
    color: #9ca3af;
    font-style: italic;
}
ol, ul { margin: 0 0 10px 22px; padding: 0; }
li { margin-bottom: 4px; }
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 16px 0;
    font-size: 9.5pt;
}
table th, table td {
    padding: 5px 8px;
    border: 1px solid #dbe1ea;
    text-align: left;
}
table thead th {
    background: #2c6e91;
    color: #ffffff;
}
.diagnostics-table th, .confusion-table th {
    background: #eef2f7;
    color: #1a2b4c;
}
table tr:nth-child(even) td { background: #f7f9fc; }
tr.best-row td {
    background: #dff3e3 !important;
    font-weight: bold;
}
.chart-gallery {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 8px;
}
figure.chart-figure {
    width: 47%;
    margin: 0 0 10px 0;
    text-align: center;
}
figure.chart-figure img {
    width: 100%;
    border: 1px solid #dbe1ea;
    border-radius: 3px;
}
figure.chart-figure figcaption {
    font-size: 8.5pt;
    color: #6b7280;
    margin-top: 3px;
}
"""


def _build_html_document(
    project_identifier: str,
    generated_at: str,
    executive_summary_html: str,
    data_diagnostics_html: str,
    model_performance_html: str,
    business_insights_html: str,
    charts_html: str,
    recommendations_html: str,
) -> str:
    """Assemble the full self-contained HTML document (inline CSS, base64
    images) that gets handed to weasyprint.HTML(string=...).write_pdf()."""
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Executive Data Analysis Report</title>
<style>{_CSS}</style>
</head>
<body>
<header class="cover">
    <h1>Executive Data Analysis Report</h1>
    <p class="project-id">Project Identifier: {html_escape.escape(project_identifier)}</p>
    <p class="generated-at">Generated {html_escape.escape(generated_at)}</p>
</header>

<section class="report-section" id="executive-summary">
    <h2>1. Executive Summary</h2>
    {executive_summary_html}
</section>

<section class="report-section page-break" id="data-diagnostics">
    <h2>2. Data Diagnostics Profile</h2>
    {data_diagnostics_html}
</section>

<section class="report-section page-break" id="model-performance">
    <h2>3. Model Performance Leaderboard</h2>
    {model_performance_html}
</section>

<section class="report-section page-break" id="business-insights">
    <h2>4. Automated Business Insights</h2>
    {business_insights_html}
    {charts_html}
</section>

<section class="report-section page-break" id="recommendations">
    <h2>5. Strategic Recommendations</h2>
    {recommendations_html}
</section>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# ReportGenerationAgent
# ---------------------------------------------------------------------------

def _load_json_report(path: str, label: str, warnings: list) -> dict:
    """Load a JSON report, returning {} and logging+recording a warning on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        msg = f"Could not load {label} from {path}: {exc}"
        logger.warning(msg)
        warnings.append(msg)
        return {}


class ReportGenerationAgent:
    """Compiles the cleaning report, ML report, business-insights markdown,
    and chart images into a single executive PDF via HTML templating.

    Purely deterministic -- no LLM call. Every upstream input is optional
    in practice: a missing/unreadable file degrades that section to a
    placeholder (and is recorded in the returned report's `warnings` /
    `charts_skipped`) rather than crashing the whole run.
    """

    def __init__(self):
        self.report_: Optional[ReportGenerationReport] = None

    @audit_logged(
        "ReportGenerationAgent",
        input_arg=("cleaning_report_path", "ml_report_path", "insights_md_path"),
    )
    def run(
        self,
        cleaning_report_path: str,
        ml_report_path: str,
        insights_md_path: str,
        chart_paths: list,
        output_path: str = _DEFAULT_OUTPUT_PATH,
        project_identifier: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Generate the executive PDF report.

        Parameters
        ----------
        cleaning_report_path : str
            Path to the DataCleaningAgent's JSON report.
        ml_report_path : str
            Path to the MLAgent's JSON report.
        insights_md_path : str
            Path to the BusinessInsightsAgent's markdown narrative.
        chart_paths : list[str]
            Paths to chart PNGs (e.g. from VisualizationAgent) to embed.
        output_path : str
            Where to write the rendered PDF.
        project_identifier : str | None
            Shown in the report's title block. Defaults to an
            auto-generated identifier stamped with the current time.

        Returns
        -------
        (success, output_path_or_error_message)
        """
        logger.info(
            "Starting report generation  cleaning=%s  ml=%s  insights=%s  charts=%d",
            cleaning_report_path, ml_report_path, insights_md_path, len(chart_paths or []),
        )

        report = ReportGenerationReport(output_path=output_path)

        cleaning = _load_json_report(cleaning_report_path, "cleaning report", report.warnings)
        ml = _load_json_report(ml_report_path, "ML report", report.warnings)

        insights_sections: dict = {
            "executive_summary": None, "business_insights": None, "recommendations": None,
        }
        try:
            with open(insights_md_path) as f:
                md_text = f.read()
            parsed = _parse_insights_markdown(md_text)
            insights_sections = _map_insights_sections(parsed)
            if not any(insights_sections.values()):
                msg = f"Business insights markdown at {insights_md_path} had no parseable '## ' sections"
                logger.warning(msg)
                report.warnings.append(msg)
        except Exception as exc:
            msg = f"Could not load business insights markdown from {insights_md_path}: {exc}"
            logger.warning(msg)
            report.warnings.append(msg)

        executive_summary_html = (
            _markdown_to_html(insights_sections["executive_summary"])
            or "<p class='placeholder'>Executive summary unavailable.</p>"
        )
        business_insights_html = (
            _markdown_to_html(insights_sections["business_insights"])
            or "<p class='placeholder'>Business insights unavailable.</p>"
        )
        recommendations_html = (
            _markdown_to_html(insights_sections["recommendations"])
            or "<p class='placeholder'>Recommendations unavailable.</p>"
        )

        data_diagnostics_html = _build_data_diagnostics_html(cleaning)
        model_performance_html = _build_model_performance_html(ml)
        charts_html = _build_charts_html(chart_paths, report)

        if project_identifier is None:
            project_identifier = f"AI-DATA-ANALYST-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        html_doc = _build_html_document(
            project_identifier=project_identifier,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            executive_summary_html=executive_summary_html,
            data_diagnostics_html=data_diagnostics_html,
            model_performance_html=model_performance_html,
            business_insights_html=business_insights_html,
            charts_html=charts_html,
            recommendations_html=recommendations_html,
        )

        report.sections_included = [
            "executive_summary", "data_diagnostics", "model_performance",
            "business_insights", "recommendations",
        ]

        try:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            HTML(string=html_doc).write_pdf(target=output_path)
        except Exception as exc:
            logger.error("Failed to render PDF: %s", exc, exc_info=True)
            return False, f"Failed to render PDF: {exc}"

        logger.info(
            "Executive report written to %s (%d charts embedded, %d skipped, %d warnings)",
            output_path, len(report.charts_embedded), len(report.charts_skipped), len(report.warnings),
        )
        self.report_ = report
        return True, output_path


if __name__ == "__main__":
    import glob

    agent = ReportGenerationAgent()
    success, result = agent.run(
        cleaning_report_path="data/processed/olist_flattened_cleaned_report.json",
        ml_report_path="data/processed/olist_flattened_cleaned_features_ml_report.json",
        insights_md_path="workspace/business_insights.md",
        chart_paths=sorted(glob.glob("workspace/visualizations/*.png")),
        output_path="workspace/executive_report.pdf",
        project_identifier="OLIST-LATE-DELIVERY-RISK",
    )
    if success:
        size_kb = os.path.getsize(result) // 1024
        print(f"Success. Report at: {result} ({size_kb} KB)")
        print(json.dumps(agent.report_.to_dict(), indent=2))
    else:
        print(f"Failed: {result}")
        raise SystemExit(1)
