# External Repositories Integration Plan (Kronos, MemPalace, Vibe-Trading, RTK, NVIDIA QPO, FinRL, MiroFish)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define how this repository (`quant-system`: Dockerized, cache-aware backtesting with `BaseStrategy`, external strategy discovery, lib-pybroker evaluation) can adopt or interoperate with seven upstream projects—without collapsing unrelated systems into one unmaintainable monolith.

**Architecture:** Treat each upstream repo as an **integration surface** with one of four modes: **(A) optional Python dependency + thin adapter**, **(B) subprocess / sidecar service with stable JSON/CSV contracts**, **(C) research notebook / GPU pipeline outside the main wheel**, **(D) out of scope / developer tooling only**. The core product remains: YAML collections → data sources → `generate_signals` → cached evaluation → reports.

**Tech Stack (this repo today):** Python 3.12+, Poetry, Pandas/NumPy, Typer, Pydantic, PyBroker (`lib-pybroker`), Optuna, FastAPI dashboard, external strategies via `STRATEGIES_PATH`.

---

## 0. Scope check and naming correction

| Upstream | Primary domain | Fit to `quant-system` |
|----------|----------------|------------------------|
| [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) | K-line foundation model (forecasting / tokenized OHLCV) | **High** — signal generation or scenario paths |
| [MemPalace/mempalace](https://github.com/MemPalace/mempalace) | Local-first AI memory (RAG / knowledge graph) | **Medium** — agent UX, research notes, not core backtest math |
| [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) | Full-stack AI trading agent (FastAPI + React, MCP, many skills) | **Low overlap** — duplicate product surface; prefer **interop** not merge |
| [rtk-ai/rtk](https://github.com/rtk-ai/rtk) | **Rust Token Killer** — CLI output compression for AI coding | **Out of scope** for trading runtime (optional dev ergonomics only) |
| [NVIDIA-AI-Blueprints/quantitative-portfolio-optimization](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization) | GPU portfolio optimization (cuDF/cuML/cuOpt), notebooks | **Medium** — offline / HPC pipeline feeding weights or constraints |
| [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) | DRL environments and training for trading | **Medium** — train externally; import policies or signals via adapter |
| [666ghj/MiroFish](https://github.com/666ghj/MiroFish) | Swarm / multi-agent simulation (AGPL-3.0) | **Low / legal-sensitive** — sandbox or separate service; **do not** AGPL-link into default wheel without legal review |

**Recommendation:** Split into **four implementation plans** if execution proceeds: (1) forecasting + signals (Kronos), (2) portfolio optimization GPU (NVIDIA), (3) RL training bridge (FinRL), (4) agent/memory/UI (MemPalace, Vibe-Trading, MiroFish) — each shippable alone.

---

## 1. Integration principles (bind to this codebase)

1. **Strategy contract:** Anything that participates in the main backtest loop must expose `param_grid()` and `generate_signals(df, params)` per `src/strategies/base.py`, or live in the external strategies repo as a `BaseStrategy` subclass discovered by `discover_external_strategies`.
2. **No silent network in tests:** Upstream deps that pull models or call APIs must be **optional extras** in Poetry and **mocked** in pytest (see workspace testing rules).
3. **Execution boundary:** Live trading, broker APIs, and OMS/RMS are **out of scope** unless explicitly extended; ADRs under `docs/adr/` already frame execution boundaries—new features should reference them.
4. **Cognitive complexity:** New orchestration must keep functions at cognitive complexity ≤ 15 (project rule); split helpers.
5. **Licensing:** Flag **AGPL (MiroFish)** and any **copyleft** train/serve path before vendoring code into this repo.

---

## 2. Per-upstream integration design

### 2.1 Kronos (`shiyu-coder/Kronos`)

- **Value:** Probabilistic multi-step OHLCV forecasts from a specialized tokenizer + transformer; useful for **regime features**, **synthetic stress paths**, or **signal priors**.
- **Mode:** **(A)** optional extra `kronos` (or `forecasting`) in `pyproject.toml` + **adapter strategy** in external strategies repo that:
  - Normalizes `df` to Kronos input format.
  - Runs `KronosPredictor` (or official inference API) to produce forward windows.
  - Maps predictions to **boolean or float features** consumed inside `generate_signals` (e.g., threshold on expected return, quantile bands).
- **Artifacts:** Model weights via Hugging Face cache; document cache dir under `.cache/` conventions.
- **Risks:** Heavy torch stack, GPU memory, reproducibility—pin versions and record model id in run metadata (`summary.json` extension or sidecar).

### 2.2 MemPalace (`MemPalace/mempalace`)

- **Value:** Durable, semantic retrieval over research notes, run logs, and ADRs for **human + agent** workflows.
- **Mode:** **(B)** CLI / local DB used by **documentation and support tooling**, not hot path of `runner.py`. Optional Typer command group e.g. `quant-system research memory ...` that shells to `mempalace` or uses its Python API if stable.
- **Risks:** Version drift of storage schema; keep integration behind subprocess to avoid coupling.

### 2.3 Vibe-Trading (`HKUDS/Vibe-Trading`)

- **Value:** Rich agent UX, MCP tools, multi-market connectors, Pine/MQL exports.
- **Mode:** **(B) interop** — treat as a **separate deployable**. Exchange artifacts only:
  - **Inputs:** `summary.csv` / `all_results.csv` / `summary.json` from this repo’s `reports/<timestamp>/`.
  - **Outputs:** strategy code or parameters consumed by external strategies, not duplicated inside `src/dashboard`.
- **Risks:** Large dependency tree; merging frontends would fork maintenance. Prefer **documented bridge** + shared file contract.

### 2.4 RTK (`rtk-ai/rtk`)

- **Clarification:** **Not a trading library.** It reduces LLM token use for CLI output.
- **Mode:** **(D)** developer-only — document in `AGENTS.md` / contributor docs how to pipe verbose logs through `rtk`; **no** runtime dependency in Poetry.

### 2.5 NVIDIA quantitative-portfolio-optimization (`NVIDIA-AI-Blueprints/quantitative-portfolio-optimization`)

- **Value:** GPU scenario generation + **Mean-CVaR** + backtest refinement patterns.
- **Mode:** **(C)** standalone conda/docker image or notebook repo that outputs **weights time series** or **constraint files** (CSV/Parquet). This repo adds a **reader** that maps weights to a strategy implementing `generate_signals` (e.g., rebalance when weight delta exceeds threshold) **or** a pre-run data enrichment step.
- **Risks:** CUDA capability, data residency; do not require GPU in default `make tests`.

### 2.6 FinRL (`AI4Finance-Foundation/FinRL`)

- **Value:** Gym-style environments and DRL algorithms for policy search.
- **Mode:** **(C)** external training; export **deterministic inference** function or discrete signals. Bridge via:
  - Saved model + lightweight wrapper strategy, **or**
  - Precomputed action series joined on index in a data prep step.
- **Risks:** FinRL env assumptions vs this repo’s OHLC schema—define a **single canonical column schema** in an ADR or `docs/architecture/` note.

### 2.7 MiroFish (`666ghj/MiroFish`)

- **Value:** Scenario / narrative simulation for qualitative shocks.
- **Mode:** **(B)** HTTP or CLI sidecar; **strictly optional**. AGPL: avoid static linking of AGPL code into distributed binary images without review.
- **Risks:** LLM cost, non-determinism, governance—treat outputs as **hypothesis artifacts**, not validated alpha.

---

## 3. Phased roadmap (all tracks)

### Phase A — Contracts and documentation (no heavy deps)

**Files:**
- Create: `docs/architecture/external-repo-interop.md`
- Modify: `AGENTS.md` (short pointer to interop doc and optional extras)

- [ ] **Step A1:** Document the four integration modes (A–D) and the canonical OHLCV column names this system expects end-to-end.
- [ ] **Step A2:** Add a table mapping each of the seven URLs to mode, license, and recommended boundary (runtime vs offline vs dev-only).
- [ ] **Step A3:** Commit

```bash
git add docs/architecture/external-repo-interop.md AGENTS.md
git commit -m "docs: add external repository interoperability map"
```

### Phase B — Kronos forecasting bridge (highest quant fit)

**Files (illustrative; exact paths follow repo layout):**
- Modify: `pyproject.toml` (optional dependency group)
- Create: `src/integrations/kronos/__init__.py` (package namespace)
- Create: `src/integrations/kronos/schema.py` (column renames, timeframe alignment)
- Create: `tests/integrations/test_kronos_schema.py`
- External strategies repo: `strategies/kronos_adapter.py` implementing `BaseStrategy` (not in this workspace if strategies remain external)

- [ ] **Step B1:** Add optional Poetry group, e.g. `[tool.poetry.group.kronos.dependencies]` with pinned `torch` / upstream package versions compatible with Python 3.12.
- [ ] **Step B2:** Implement pure-Pandas **schema adapter** functions with unit tests (no model download in CI).

```python
# tests/integrations/test_kronos_schema.py
import pandas as pd

from src.integrations.kronos.schema import to_kronos_frame, assert_monotonic_index


def test_assert_monotonic_index_rejects_unsorted() -> None:
    df = pd.DataFrame({"open": [1, 1]}, index=pd.to_datetime(["2020-01-02", "2020-01-01"]))
    try:
        assert_monotonic_index(df)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_to_kronos_frame_renames_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "Open": [1.0, 1.1],
            "High": [1.2, 1.2],
            "Low": [0.9, 1.0],
            "Close": [1.1, 1.15],
            "Volume": [100, 110],
        },
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )
    out = to_kronos_frame(df)
    assert set(out.columns) >= {"open", "high", "low", "close", "volume"}
```

- [ ] **Step B3:** Run `poetry run pytest tests/integrations/test_kronos_schema.py -q` — expect PASS after implementation.
- [ ] **Step B4:** Add **gated** integration test marked `@pytest.mark.kronos` that skips without `RUN_KRONOS_INTEGRATION=1` and Hugging Face cache, to avoid CI network.
- [ ] **Step B5:** Document model cache location and reproducibility fields for reports.
- [ ] **Step B6:** Commit in small slices (schema first, optional deps second).

### Phase C — FinRL / NVIDIA offline bridges

**Files:**
- Create: `docs/architecture/offline-policy-import.md`
- Create: `src/integrations/portfolio/io.py` (read weights Parquet/CSV into aligned Series)
- Create: `tests/integrations/test_portfolio_weights_io.py`

- [ ] **Step C1:** Define file contract: columns `timestamp`, `symbol` or `asset`, `weight`, optional `gross_exposure`.
- [ ] **Step C2:** Implement `load_weights_frame(path: Path) -> pd.DataFrame` with strict validation (missing symbols, non-sum-to-one warnings).
- [ ] **Step C3:** Tests with synthetic Parquet using `tmp_path` (no FinRL install in default CI).
- [ ] **Step C4:** Commit

### Phase D — MemPalace / Vibe-Trading / MiroFish agent layer

- [ ] **Step D1:** Add `docs/architecture/agent-tooling.md` describing subprocess boundaries and artifact exchange only.
- [ ] **Step D2 (optional):** Typer command `quant-system interop export-latest-run --reports-dir reports` producing a zip/json bundle for Vibe-Trading ingestion (pure stdlib + existing reporting modules).
- [ ] **Step D3:** Commit

### Phase E — RTK developer note

- [ ] **Step E1:** One paragraph in contributor-facing doc: using `rtk` to shrink logs is supported; no Poetry dependency.

---

## 4. Testing and CI policy

- Default `make tests` / `pytest` must remain **offline** and **fast**.
- Any GPU, Hugging Face, or broker-backed path uses **markers** and **env gates** (`RUN_*_INTEGRATION=1`).
- Coverage floor (≥80%) must be preserved; add narrow modules with high testability (schema I/O) rather than importing entire upstream stacks in unit tests.

---

## 5. Security and safety

- Secrets remain in `.env` only; upstream tools must not encourage committing API keys.
- Treat MiroFish and cloud LLM flows as **untrusted content generators**—sanitize any markdown rendered in the dashboard.
- Trading safety ADRs (`docs/adr/0003-*`, `0006-*`) apply if any bridge moves from backtest toward live execution.

---

## 6. Spec coverage self-review

| Requirement (user list) | Covered by |
|-------------------------|------------|
| Kronos | §2.1, Phase B |
| MemPalace | §2.2, Phase D |
| Vibe-Trading | §2.3, Phase D |
| RTK | §0, §2.4, Phase E |
| NVIDIA QPO | §2.5, Phase C |
| FinRL | §2.6, Phase C |
| MiroFish | §2.7, Phase D + licensing |

**Placeholder scan:** None intentional; concrete file paths are suggestive—adjust to match final package naming during implementation.

**Type consistency:** Use a single canonical timestamp index name (`DatetimeIndex` on OHLCV frames) across `schema.py` and `portfolio/io.py`.

---

## Execution handoff

**Plan complete and saved to** `docs/superpowers/plans/2026-04-19-external-quant-ai-repos-integration.md`.

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per phase (A→E), review between phases.

**2. Inline Execution** — run phases sequentially in one session with checkpoints after each `git commit`.

**Which approach?**

If neither is chosen, default to **Phase A only** as the next mergeable increment.
