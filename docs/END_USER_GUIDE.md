# End User Guide

A friendly walkthrough for anyone who wants to analyze their own data with
this tool — no coding experience required.

---

## What this tool does

This is a self-service data analysis tool. You give it a spreadsheet of
data (as a CSV file) and tell it what you'd like to predict — for
example, whether a customer will cancel their subscription, how much a
house will sell for, or whether a delivery will arrive late. The tool
then does the rest of the work for you: it cleans up messy data, looks
for patterns, builds a predictive model, draws charts, writes up a
plain-English summary of what it found, and puts it all together into a
downloadable report. You don't need to know any statistics or
programming to use it — you just upload your file, answer a couple of
simple questions, and click one button.

---

## Before you start: preparing your dataset

### File format

Your data needs to be a **CSV file** (Comma-Separated Values). This is a
plain spreadsheet format that both Excel and Google Sheets can save to —
look for "Save As" or "Export" and choose "CSV" as the file type. Each
row in your file should be one example (one customer, one house, one
order), and each column should be one piece of information about that
example.

### How much data you need

- **Rows**: somewhere between **5,000 and 150,000** rows works best.
  Fewer than that and the tool won't have enough examples to learn a
  reliable pattern; a lot more than that and it'll simply take longer to
  run without adding much benefit.
- **Columns**: somewhere between **8 and 45 columns** is the sweet spot.
  Too few columns and there isn't much for the model to learn from; a
  huge number of columns tends to add noise and slows things down
  without necessarily improving results.

Your dataset doesn't have to sit exactly inside these ranges to work —
but the further outside them you are, the less reliable your results are
likely to be.

### What "the target column" means

Every dataset you upload needs one column that is **the thing you want
to predict**. This is called the "target column." Think of it as the
answer key — it's the piece of information you'd want to know in advance
for a brand-new row of data you hadn't seen yet.

A few examples:
- A column called `will_cancel` (yes/no) → the tool predicts whether a
  customer will cancel.
- A column called `sale_price` (a dollar amount) → the tool predicts
  what a house will sell for.
- A column called `is_late` (yes/no) → the tool predicts whether a
  delivery will be late.

### What makes a good target column — and what doesn't

**Good target columns:**
- Something that varies across your rows (not the same value every
  time).
- Something you'd genuinely want to know ahead of time for a new,
  unseen example.
- A yes/no outcome, a category (like "small / medium / large"), or a
  meaningful number (like a price, a score, or a count).

**Target columns to avoid:**
- A column that's really just a row number or a unique ID (e.g.
  `customer_id`) — there's no pattern to learn because every value is
  different.
- A column that's almost always the same value (e.g. 99.9% "no," 0.1%
  "yes") with very few examples of the rare outcome — the tool can still
  attempt this, but it will have very few real examples to learn the
  rare case from, so treat those results cautiously.
- A column that's only known *after the fact* and wouldn't actually be
  available at the moment you'd want the prediction (for example, don't
  use "customer's final review score" to predict something you'd want to
  know *before* the order ships — that information doesn't exist yet at
  that point in time).

If you're not sure, just try it — you'll be able to see in your results
whether the tool found a real pattern or not.

---

## Step-by-step walkthrough

### 1. Uploading your file (Dataset Ingestion page)

Open the **Dataset Ingestion** page from the sidebar. Click the file
upload box and select your CSV. Once it loads, you'll see a preview of
your first several rows, plus the total row and column count, so you can
double-check it's the right file.

### 2. Picking your target column

Right below the preview, you'll see a dropdown asking **"Target column
(what should the model predict?)"** — pick the column you identified
above.

You'll also see two optional dropdowns:
- **ID / row-identifier column** — if your data has a column that's just
  a unique ID (like an order number or a customer number), select it
  here so the tool knows to ignore it as a prediction clue.
- **Group column** — only relevant if the same real-world person or
  entity can appear more than once in your data (e.g. one customer with
  several orders as separate rows). Selecting the right column here
  keeps all of one customer's rows together when the tool tests itself,
  which gives you a more honest sense of how well it will really
  perform on a customer it's never seen before. If this doesn't apply
  to your data, just leave it as "(none)."

There's also an optional section where you can type in plain-English
names for your outcome (for example, typing "late delivery" and
"on-time delivery" instead of the defaults) — this makes your charts and
written summary read more naturally, but it's entirely optional.

### 3. Running the pipeline

Click **"Save & Run Full Pipeline."** This kicks off the whole analysis:
cleaning your data, exploring it, building and testing a model, drawing
charts, writing a summary, and assembling your final report — all
automatically, in that order.

**How long it takes**: this can range from under a minute for a small
file to a few minutes for a larger one, since the tool tries out several
different modeling approaches behind the scenes before picking the best
one. A progress spinner will stay on screen while it works — just leave
the page open.

**What "success" looks like**: when it finishes, you'll see a green
"Pipeline complete" message along with a summary table showing every
step it went through and whether each one succeeded. If something goes
wrong partway through, the tool tries to recover automatically and
continue rather than stopping cold — you'll see that reflected in the
summary table too. If it can't recover, you'll see a clear error message
explaining what happened and where.

### 4. Reading your results (Data Analysis and ML Studio pages)

**Data Analysis** shows you what the cleaning step did to your data
(how many duplicate rows it removed, which columns needed fixing, and
so on) and a plain statistical overview of your columns — averages,
how spread out the values are, which columns tend to move together, and
which have unusual outlier values.

**ML Studio** shows you how well the model actually performs. A couple
of terms you'll see there, explained simply:

- **Task type** — whether the tool treated your problem as a yes/no
  question ("classification"), a multiple-choice question ("multiclass
  classification"), or a number-prediction question ("regression"). It
  figures this out automatically from your target column.
- **Confusion matrix** (yes/no predictions only) — a simple table
  showing four counts: how many times the model correctly said "yes,"
  correctly said "no," incorrectly said "yes" when the answer was "no,"
  and incorrectly said "no" when the answer was "yes." It's the clearest
  way to see exactly where the model's mistakes are happening.
- **RMSE / MAE** (number predictions only) — these tell you, on average,
  how far off the model's predictions typically are, measured in the
  same units as what you're predicting. For example, an MAE of 5,000 on
  house prices means predictions are typically off by about $5,000 in
  either direction. Smaller is better.
- **Feature importances** — a ranked list of which columns in your data
  mattered most to the model's predictions. Columns near the top had
  the biggest influence; columns not shown didn't meaningfully help.
- **Confidence / calibration** — see the note on limitations below
  before treating any percentage the model reports as an exact
  probability.

### 5. Viewing your charts (Visualization Gallery page)

This page lays out every chart the tool generated — distributions of
your key columns, how columns relate to each other, which features
mattered most, and (depending on your task type) either a confusion
matrix picture or an actual-vs-predicted chart. No setup needed here;
just browse.

### 6. Reading the written insights (Insights Panel page)

This page shows a plain-English written summary explaining what the
model found and what it might mean for your business, along with a
couple of concrete recommendations.

**Important**: this narrative is written by an AI, not a person, based
on your model's actual numbers. It's generally grounded and specific,
but AI-written summaries can occasionally overstate a pattern or phrase
something a bit too strongly. If a claim in the summary surprises you or
sounds like a stretch, it's worth cross-checking it against the numbers
on the ML Studio page before repeating it to someone else.

### 7. Downloading your report (Reports Hub page)

This is where you download the finished PDF — a single polished document
combining your data summary, model results, charts, and written insights
in one place, ready to share. This page also has an expandable section
where you can read (or download) the plain-language notes on this
model's limitations — worth a skim before you present your results to
anyone else.

### 8. If you're curious: System Log Explorer and Run History

These two pages aren't required for a normal analysis, but they're there
if you want to dig deeper:

- **System Log Explorer** is a running record of everything the tool
  has done, step by step, with timestamps — useful mainly if something
  didn't work and you want to see exactly what happened.
- **Run History** keeps a permanent record of every model you've ever
  trained with this tool, so you can compare how a new run stacks up
  against previous ones over time, even after you've re-run the
  pipeline and its "latest results" have moved on.

---

## Common questions / troubleshooting

**My upload failed or gave an error — what do I do?**
Double check the file is actually saved as a `.csv` file, not `.xlsx` or
another spreadsheet format. If it was exported from an unusual system,
try opening it in Excel or Google Sheets and re-saving it as CSV from
there — that usually clears up formatting quirks that trip up the
upload.

**What if one of my columns is missing a lot of data?**
The tool automatically flags columns that are missing more than half
their values — you'll see them called out in the Data Analysis page's
cleaning report. By default it doesn't delete them outright; it fills in
the gaps using the rest of the column's data, since sometimes the fact
that a value is missing is itself a useful clue. If a column is missing
almost everything, though, its predictive value will likely be limited
regardless.

**What does "task type" mean, and why didn't I get to choose it?**
The tool figures out automatically whether your target column is a
yes/no question, a multiple-choice question, or a number to predict,
just by looking at the values already in that column. You don't need to
tell it which — just make sure you picked the right target column, and
it takes care of the rest.

**Why did some of my columns disappear after cleaning?**
A few columns get automatically dropped along the way, and this is
normal:
- Columns that are almost entirely empty.
- Columns where nearly every row has the exact same value — these don't
  give the model anything useful to learn from.
- Columns that are near-duplicates of another column you already have
  (carrying basically the same information twice).

You can always see exactly which columns were removed, and why, in the
Data Analysis page's cleaning report.

---

## A note on limitations

Like any predictive model, the one this tool builds for you isn't
perfect, and it's worth understanding its rough edges before you act on
its results. A few honest, plain-language examples:

- The AI's confidence percentages (e.g. "73% chance of X") aren't
  perfectly precise — treat them as a rough signal of relative
  confidence, not an exact probability.
- A model can sometimes score lower on plain "percent correct" than a
  trivial guess would — that isn't always a bug; it's often the
  trade-off made to catch more of a rare-but-important outcome.
- Which specific column the model calls "most important" can shift a
  bit depending on random chance in how the model was trained,
  especially when several columns carry similar information.

None of this means the results aren't useful — it means they should be
read with the same common sense you'd apply to any forecast. For the
full, specific list of what this project's own testing uncovered (with
real numbers, not generic disclaimers), see
[`KNOWN_LIMITATIONS.md`](../KNOWN_LIMITATIONS.md), also downloadable
from the Reports Hub page.
