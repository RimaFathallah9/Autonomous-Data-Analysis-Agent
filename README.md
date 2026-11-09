# ADA — Autonomous Data-Analysis Agent + Evaluation Framework

A learning-oriented project: build an agent that **cleans data, detects anomalies and visualises**,
and — more importantly — **measure it honestly**.

> Stack: Python · Pandas · NumPy · Scikit-learn · LLMs/SLMs · Function Calling · LangGraph · FastAPI · Plotly · Docker

This README is written as a course. Read it top to bottom once; later use it as a reference.

---

## 0. Run it in 3 minutes (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"

.\.venv\Scripts\python -m pytest                      # 16 tests, ~1 min
.\.venv\Scripts\ada datasets                          # registry + licenses
.\.venv\Scripts\ada run experiments/smoke.yaml        # seconds
.\.venv\Scripts\ada run experiments/default.yaml      # ~2.5 min, 10 seeds
.\.venv\Scripts\ada serve                             # API at http://127.0.0.1:8000/docs
```

Results land in `results/<run_name>_<confighash>/` (open `report.html` in a browser).

Docker: `docker compose up --build` (API) · `docker compose --profile tools run --rm experiment`.

---

## 1. The big idea (why this project is structured this way)

Most "AI agent" demos show one nice example. That proves nothing. A **research-grade** project answers:

1. *Compared to what?* → **baselines** (naive, no-cleaning, oracle).
2. *On what data, legally?* → **dataset registry** with licenses + checksums.
3. *Is the answer exact?* → **fault injection**: we break clean data ourselves, so we *know* the truth.
4. *Is it luck?* → **many seeds, bootstrap confidence intervals, paired tests**.
5. *Can I rerun it?* → **config files, pinned versions, fingerprints, Docker**.

The agent itself is only ~250 lines. The evaluation scaffolding is the real contribution.

### Architecture

```
data/registry.yaml ──► datasets/registry.py ──► clean DataFrame
                                                    │
                          corruption.py  (seeded fault injection + exact error mask)
                                                    │ dirty data
        ┌───────────────────────┬───────────────────┴──────────┐
     baselines            agent (LangGraph)                  oracle / no-op
   (naive z-score…)   plan → act(tools) → plan … → finalize   (reference points)
        └───────────────────────┴───────────────────┬──────────┘
                                                    ▼
                        eval/metrics.py  vs ground truth  →  stats.py (CI, Wilcoxon)
                                                    ▼
                          results.csv · summary.csv · paired.csv · report.html
```

### Folder map

| Path | What it is | Why it exists |
|---|---|---|
| `data/registry.yaml` | dataset metadata: license, source, sha256 | legality + reproducibility |
| `src/ada/datasets/` | loaders (sklearn / synthetic / checksum-verified URL) | one door for all data |
| `src/ada/corruption.py` | breaks data in recorded ways | **creates ground truth** |
| `src/ada/agent/detectors.py` | pure functions: robust z, IsolationForest, imputation | shared by baselines *and* tools |
| `src/ada/agent/tools.py` | whitelisted functions + pydantic schemas | **function calling** |
| `src/ada/agent/llm.py` | LLM client abstraction (OpenAI-compatible / scripted) | plug LLMs *or* SLMs |
| `src/ada/agent/graph.py` | LangGraph state machine | the agent loop |
| `src/ada/eval/` | metrics, stats, runner | the evaluation framework |
| `src/ada/viz.py` | Plotly figures | insights/visualisation |
| `src/ada/api.py` | FastAPI service | turn agent into a service |
| `experiments/*.yaml` | experiment configs | rerunnable science |
| `tests/` | 16 tests | executable documentation |
| `Dockerfile`, `docker-compose.yml` | packaging | "works on my machine" → "works" |

---

## 2. Data infrastructure

### 2.1 Registry (`data/registry.yaml`)
**What:** one YAML entry per dataset (`name, loader, license, source, sha256, has_labels, redistribution_allowed, task_tags…`).
**Why:** you must be able to say *which bytes* and *under what license*. Pydantic (`DatasetSpec`) validates the file, so typos fail immediately.
**How remote data stays honest:** `sha256: null` at first → `ada fetch uci_adult` downloads and prints the hash → you paste it in → afterwards a silent upstream change raises a *checksum mismatch* and deletes the file. This is "trust on first use", the same idea as `package-lock.json`.
**Where files go:** `data/raw/` (gitignored, immutable). We commit only metadata, never data whose license forbids redistribution.

Included datasets (licenses are in the registry; **re-verify on the source page before redistributing**):

| Name | Use | Note |
|---|---|---|
| `wine`, `breast_cancer_wdbc` | cleaning base + row-anomaly benchmark | bundled with scikit-learn, UCI, CC BY 4.0 |
| `synthetic_tabular` | cleaning (has a categorical column → typos) | generated, CC0 |
| `synthetic_timeseries` | time-series anomalies | generated, CC0 |
| `uci_adult` | optional real mixed-type data | needs network + pinning |

### 2.2 Why synthetic + injected faults?
Real public data has natural dirt nobody labelled, so you can't compute precision/recall.
We take *clean* data and inject: missing values (MCAR or MAR), outliers (6–10σ), typos, duplicate rows, and for time series spikes / level shifts / flatlines.
`corruption.py` returns an **`error_mask`** (a True/False table of exactly which cells we broke). Every metric below is computed against it.

* **MCAR** = Missing Completely At Random (coin flip per cell). **MAR** = Missing At Random (chance depends on another *observed* column). MAR is harder and more realistic.
* Everything is driven by `np.random.default_rng(seed)`: same seed ⇒ identical dirt (tested).
* Limitation (be honest in your write-up): injected faults are *cleaner* than real ones. Wine/breast-cancer also have natural heavy tails that look like outliers, so some "false positives" are arguably real anomalies.

---

## 3. The agent, step by step

### 3.1 Function calling (`tools.py`)
An LLM can't touch your DataFrame. Instead we give it a **menu of tools** described in JSON schema; it replies "call `detect_outliers` with `{threshold: 5}`"; *our* code runs it and sends the result back.

Design choices and why:

* **No arbitrary code execution.** The model only picks from a whitelist (`profile_dataframe`, `drop_duplicates`, `detect_outliers`, `detect_category_typos`, `impute_missing`, `detect_row_anomalies`, `detect_ts_anomalies`, `make_plot`). Safer, loggable, and small models (SLMs) are much better at "choose tool + 2 args" than writing correct pandas.
* **Pydantic models define arguments.** One class gives (a) the JSON schema shown to the LLM and (b) runtime validation of what it sends (e.g. `threshold` must be `> 0`).
* **Errors go back to the model, not up the stack.** `call_tool` never raises; invalid args return `{"error": ...}` so the LLM can self-correct. (Tested.)
* **`Session` object** holds the DataFrame outside the chat history — never paste a DataFrame into a prompt.
* **Detect, then repair.** Detection tools only *flag*; `impute_missing` is the only one that changes values. This separation lets us score detection and repair independently.

### 3.2 LLM backends (`llm.py`) — LLMs *and* SLMs
`LLMClient.chat(messages, tools)` is the only thing the graph knows.

* `OpenAICompatibleClient` — OpenAI, Azure, or a **local small model** via Ollama / vLLM / LM Studio. Only `base_url` and `model` change:

  ```powershell
  $env:ADA_LLM_PROVIDER="openai"; $env:ADA_LLM_MODEL="gpt-4o-mini"; $env:OPENAI_API_KEY="sk-..."
  # local SLM (Ollama exposes an OpenAI-compatible endpoint; the model must support tool calling):
  $env:ADA_LLM_PROVIDER="openai"; $env:ADA_LLM_BASE_URL="http://localhost:11434/v1"; $env:ADA_LLM_MODEL="qwen2.5:7b"
  ```
* `ScriptedClient` — a deterministic fake LLM that follows a fixed plan. Three jobs: **offline tests**, **zero-cost reproducible default**, and a **control group** (if a real LLM can't beat a script, the LLM adds nothing).
* Temperature defaults to 0 for determinism.

### 3.3 LangGraph (`graph.py`)
```
START → plan ──tool calls & budget left──► act ──► plan …
              └─no tool calls / budget gone─► finalize → END
```
* `plan` calls the LLM; `act` runs the requested tools; `route` decides.
* **Why a graph and not a `while` loop?** explicit, testable transitions; easy to later add checkpoints, human approval or a "critic" node.
* **`max_steps` hard budget** — a confused model can't loop forever or burn money (tested with a deliberately broken model).
* `operator.add` reducer on `messages`: nodes return only *new* messages; LangGraph appends.
* Output `AgentResult` includes the full trace (every prompt/tool call/result) for failure analysis.

### 3.4 Visualisation (`viz.py`, Plotly)
Functions return `go.Figure`, reused by the `make_plot` tool, the HTML report (error-bar charts) and the API (figure JSON a front-end can render).

### 3.5 FastAPI (`api.py`)
* `GET /health`, `GET /datasets`, `POST /analyze` (body: `{"dataset":"wine"}` or `{"csv":"a,b\n1,2"}`).
* Pydantic request/response models ⇒ validation + free docs at `/docs`.
* Untrusted input is capped (`ADA_MAX_CSV_BYTES`, `ADA_MAX_ROWS`).
* Sync `def` endpoints because pandas/sklearn are CPU-bound (FastAPI uses a thread pool).

```powershell
curl.exe -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" -d "{\"dataset\":\"wine\"}"
```

### 3.6 Docker
* `pip install .` is placed *before* copying the rest, so the slow layer is cached.
* Runs as a non-root user, has a `HEALTHCHECK`.
* `docker-compose.yml`: `api` service + `experiment` one-shot (mounts `./results`).
* LLM selection is entirely env vars — same image for scripted, OpenAI or local SLM.

---

## 4. Evaluation strategy

### 4.1 Three suites

| Suite | Task | Methods compared |
|---|---|---|
| `cleaning` | find + repair injected faults | `no_cleaning` (floor), `naive` (z>3 baseline), `agent`, `oracle` (ceiling) |
| `row_anomaly` | rank anomalous rows (minority class = anomaly, 5%) | `zscore_max`, `lof`, `isolation_forest`, `agent` |
| `timeseries` | find injected spikes/shifts/flatlines | `naive_zscore`, `agent` |

**Reference points matter:** `no_cleaning` and `oracle` tell you the *range* of possible scores, so "0.96" can be read as "98% of the way to perfect".

### 4.2 Metrics (all in `eval/metrics.py`, unit-tested by hand-computed examples)

| Metric | Meaning | Why this one |
|---|---|---|
| `detect_precision/recall/f1` | cell-level error detection vs `error_mask` | exact because we know the truth |
| `repair_nrmse` | error on repaired numeric cells ÷ column std | comparable across columns/datasets |
| `repair_cat_acc` | fraction of categorical errors restored exactly | |
| `over_cleaning_rate` | % of *correct* cells the method changed | catches "fix everything" cheating |
| `dup_removed_frac` | injected duplicates removed | |
| `downstream_acc` | train a classifier on the cleaned data, test on **clean held-out** rows | does cleaning actually help? |
| `auroc`, `auprc` | ranking quality of anomaly scores | **AUPRC** is the one to trust under heavy class imbalance |
| `event_f1` | time-series *events* found, false alarms penalised | avoids the inflated point-adjusted F1 |
| `verify_numeric_claims` | re-compute numbers an LLM wrote | faithfulness / hallucination check |

### 4.3 Statistics (`stats.py`)
* 10 seeds per cell; each seed changes the corruption and the train/test split.
* **95% bootstrap CI** of the mean (resample runs 2000×, take 2.5%/97.5% quantiles).
* **Paired Wilcoxon signed-rank** (agent vs each method on *the same seeds*). Needs ≥6 seeds, otherwise `NaN` — we refuse to fake significance.
* Leakage control: downstream test rows are never corrupted or used in cleaning.

### 4.4 Reproducibility artefacts
Each run folder has `config.yaml`, `env.json` (python/package versions, git commit, **data fingerprints**), `results.csv` (raw), `summary.csv`, `paired.csv`, `summary.md`, `report.html`. The folder name includes a hash of the config, so changing the config makes a new folder instead of overwriting.

---

## 5. Actual results from `experiments/default.yaml` (10 seeds, scripted agent)

Means shown; CIs/p-values are in `summary.csv` / `paired.csv`.

**Cleaning, "high" corruption (15% MAR missing, 5% outliers/typos, 10% duplicates), breast_cancer_wdbc**

| metric | no_cleaning | naive | agent | oracle |
|---|---|---|---|---|
| detect_f1 | 0.874 | 0.982 | 0.977 | 1.000 |
| repair_nrmse ↓ | 8.10 | 1.65 | **1.18** | 0 |
| dup_removed_frac | 0 | 0 | **1.00** | 1.00 |
| over_cleaning_rate ↓ | 0 | 0.001 | 0.010 | 0 |
| downstream_acc | 0.953 | 0.964 | 0.965 | 0.972 |

**Time series:** agent AUPRC 0.742 vs naive 0.185; event-F1 0.968 vs 0.296 (Wilcoxon p = 0.002, n = 10).

**How to read these honestly**
* Agent ≫ naive on **repair error**, **duplicates**, **time series** — the real wins (robust statistics, seasonal decomposition, dedupe).
* On **detection F1** the naive baseline is *as good or better* (higher precision); the agent's robust threshold trades precision for recall and over-cleans ~1%.
* `downstream_acc` differences between naive and agent are within noise — a logistic regression is tolerant to this much dirt. Don't claim a gain you can't show. A harsher classifier or higher corruption would be needed.
* Row anomalies: IsolationForest ≈ agent *by construction* (the scripted agent calls that exact tool). The interesting comparison appears when a real LLM chooses tools/arguments.
* The scripted agent isn't "intelligent"; these numbers are the **tool-library ceiling** and the **control** for real LLM runs.

---

## 6. How to extend (suggested learning path)

1. **Run a real LLM** with the env vars in §3.2; run the same config and compare with the scripted control. Look at `AgentResult.trace` to see where it fails (wrong threshold? skipped dedupe? looped?).
2. **Add an SLM** via Ollama and compare cost/accuracy vs. a large model — add `tokens` and `latency` columns to `results.csv`.
3. **Add a tool** (e.g. `detect_unit_mismatch`): write a pydantic args model + function with `@tool(...)` — it is automatically exposed to the LLM and the API.
4. **Add a dataset**: append to the registry, run `ada fetch`, pin the sha256, add it to a config.
5. **Add a corruption** (e.g. MNAR missingness, unit errors) in `corruption.py` + a test that the mask is exact.
6. **Insight suite:** have the LLM write a report, extract numeric claims, and score them with `verify_numeric_claims`.
7. **Real benchmarks:** ODDS/ADBench (tabular anomalies), NAB / UCR Archive (time series), Raha/Baran cleaning sets (Hospital, Flights, Beers). Check each license and pin checksums first.
8. Ablations: remove `drop_duplicates` or the typos tool from the plan and measure the drop.

## 7. Known limitations
* Injected faults ≠ real-world faults; synthetic time series are easy compared to real ones.
* Imputation is median/mode only (a deliberately simple baseline; try IterativeImputer/KNN).
* Only one scripted policy has been evaluated here; LLM results depend on provider/model/version, so always log them.
* Dataset licenses were written from memory — verify before redistributing.
