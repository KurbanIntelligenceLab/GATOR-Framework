# GAS Framework

This repository implements the **GAS workflow** used in our paper for screening **LLM-assisted thermodynamic evaluation for reversible hydrogen storage on alkaline-earth--modified TiO$_2$ nanoparticles: a Grounded Agentic Screening framework**.

The codebase is organized to match the paper’s three stages:

- **Stage 1 (Surrogate prediction)**: an **E(3)-equivariant attention GNN** (implemented via a light wrapper around **GotenNet**) maps optimized xyz geometries to base electronic targets (HOMO/LUMO and total energies).
- **Stage 2 (Deterministic descriptors)**: numeric screening tables (tab1–tab4) are computed deterministically from Stage 1 predictions + geometries. No LLM is involved.
- **Stage 3 (Agentic screening)**: a constrained “computational chemist” LLM consumes only a compact **JSON payload** built from tab1–tab4 and generates a ranked screening report. The agent **does not invent numbers**.

---

## Repository layout (what matters for the pipeline)

- `data/geometries/`: xyz inputs (pristine/doped + adsorbed) and `H2.xyz`
- `data/metrics_tab1.csv`: electronic descriptors (HOMO/LUMO → conceptual DFT fields)
- `data/metrics_tab2.csv`: adsorption geometry metrics + adsorption mode labels
- `data/metrics_tab3.csv`: total energies + adsorption energies + heuristic DOE window
- `data/metrics_tab4.csv`: finite‑T thermodynamic projections (ΔH_corr, T50, ΔG°)
- `scripts/e3_attention_dataloader.py`: xyz→graph loader + label normalization + `.npz` cache
- `scripts/train_model.py`: leave‑one‑out training for **single‑target** equivariant attention models
- `modules/`: deterministic calculators + the chemist-agent implementation
- `scripts/run_chemist_agent.py`: CLI entrypoint for Stage 3

---

## Installation

This repo vendors the `gotennet/` package (ICLR 2025). The Stage 1 training script imports `gotennet.*`, so you should make it importable by installing the vendored package.

```bash
python -m venv .venv
source .venv/bin/activate

# Install vendored gotennet (recommended)
pip install -e "gotennet[full]"
```

Notes:
- If you only run the Stage 3 agent on existing `data/metrics_tab*.csv`, you do **not** need the full GNN stack.
- `gotennet/requirements.txt` contains one pinned configuration (CUDA wheels) that can be useful as a reference for GPU installs.

---

## Quickstart (Stage 3 only: generate a screening report)

If `data/metrics_tab{1..4}.csv` already exist, you can run the chemist-agent immediately:

```bash
python scripts/run_chemist_agent.py --llm ollama --model llama3.2:3b
```

Provider options:
- **Ollama (default)**: local HTTP call (recommended for offline runs)
- **OpenAI**: set `OPENAI_API_KEY` and run with `--llm openai`
- **Transformers**: local HF inference with `--llm transformers` (optional dependency + may download weights)

Repeated-run logging for robustness analysis:

```bash
python scripts/run_chemist_agent.py --llm ollama --model llama3.2:3b --runs 100 --outdir results/chemist_agent_runs
```

This writes a batch folder containing:
- `manifest.json` (backend/model/temperature/etc.)
- `prompt.md` and `payload.json` (saved once per batch)
- per-run `run_*.json` + `run_*.md` plus an aggregate `runs.jsonl`

---

## Stage 1: Generating predictions (E(3)-equivariant attention via GotenNet)

### What is predicted

The Stage 1 surrogate models are **single-target regressors** trained independently, matching the paper’s “no cross‑target leakage” setup. Supported targets are:

- `homo`, `lumo`
- `e_np_h2` (total energy of NP+H₂)
- `e_np` (total energy of NP)
- `e_h2` (standalone H₂ energy; uses `H2.xyz` if present)

### Data ingestion + caching

`scripts/e3_attention_dataloader.py`:
- builds radius graphs from xyz with cutoff \(r_c\) (default 5.0 Å)
- normalizes targets (mean/std) for training
- stores a compressed dataset cache under `data/` as:
  `orbital_attention_dataset_cutoff{cutoff}_t1{mtime}_t3{mtime}.npz`

System naming convention (CSV → xyz):
- `Pristine` → `TiO2-H2.xyz`
- `Ba-TiO2` → `Ba-TiO2-H2.xyz` (and similarly for other dopants)
- `H2` → `H2.xyz`

### Leave-one-out training (paper protocol)

Run leave-one-out (LOO) training for one or more targets:

```bash
python scripts/train_model.py --targets homo,lumo,e_np_h2,e_np,e_h2 --seeds 42 --epochs 1000
```

Outputs (default `--results_dir results_e3_attention`):

```
results_e3_attention/{material_slug}/{target}/{seed}/results.json
results_e3_attention/{material_slug}/{target}/{seed}/best.pt
```

`results.json` records the full configuration, training loss curve, and the held‑out prediction (both normalized and denormalized).

---

## Stage 2: Deterministic descriptor construction (tab1–tab4)

Stage 2 is implemented as deterministic “tools” under `modules/`. Given the same inputs, these computations are fully reproducible and contain **no LLM calls**.

### Generating / refreshing the CSV tables

This repo focuses on the deterministic calculators and the agentic screening step. It does **not** currently ship a single “one-click” CLI that rebuilds `data/metrics_tab{1..4}.csv` end-to-end.

Typical usage is:
- **tab1** from HOMO/LUMO (Stage 1 predictions or reference values)
- **tab2** from xyz geometries (always deterministic)
- **tab3** from total energies (Stage 1 predictions or reference values)
- **tab4** from tab3 \(E_{\mathrm{ads}}^{(\mathrm{kJ/mol})}\) (deterministic; can be recomputed at load time)

If you only care about making the agent robust to missing tab4 values, you can rely on the built-in recomputation in `modules.chemist_agent.load_system_records(...)` (see “tab4” below).

### tab1 — Electronic + conceptual-DFT descriptors (from HOMO/LUMO)

Implemented in `modules/electronic_calculations.py` as `ElectronicProperties`:
- \(E_g = \varepsilon_{\mathrm{LUMO}}-\varepsilon_{\mathrm{HOMO}}\)
- \(\mathrm{IP}\approx-\varepsilon_{\mathrm{HOMO}}\), \(\mathrm{EA}\approx-\varepsilon_{\mathrm{LUMO}}\)
- \(\chi, \eta, \omega\) with safeguards for small \(\eta\)

### tab2 — Geometry-based adsorption metrics + mode classification (from xyz)

Implemented in `modules/adsorption_metrics.py` as `AdsorptionMetrics`:
- \(r_{\mathrm{H-H}}\), \(\Delta r_{\mathrm{H-H}}\) relative to 0.741 Å
- adsorption mode thresholds:
  - molecular: \(r_{\mathrm{H-H}}<0.80\) Å
  - activated: \(0.80\le r_{\mathrm{H-H}}\le 0.95\) Å
  - dissociative: \(r_{\mathrm{H-H}}>0.95\) Å
- optional AE proximity metrics (`r_H-AE_A`, `R_O-AE_A`)

### tab3 — Adsorption energetics + heuristic DOE window (from total energies)

Implemented in `modules/adsorption_energies.py` as `AdsorptionEnergies`:
- \(E_{\mathrm{ads}} = E_{\mathrm{NP+H_2}} - E_{\mathrm{NP}} - E_{\mathrm{H_2}}\)
- \(E_{\mathrm{ads}}^{(\mathrm{kJ/mol})} = 96.485\,E_{\mathrm{ads}}^{(\mathrm{eV})}\)
- DOE window heuristic based on \(15\le |E_{\mathrm{ads}}^{(\mathrm{kJ/mol})}|\le 25\)

### tab4 — Finite‑T thermodynamic projections (deterministic recomputation supported)

Implemented in `modules/thermodynamic_projections.py` as `ThermodynamicProjections`:
- corrected enthalpies with \(\Delta S_T\in\{6.0,\ 25.96\}\) kJ/mol
- DOE labels with endothermic handling
- \(T_{50}(p)=\frac{\Delta H_{\mathrm{corr}}}{\Delta S_{\mathrm{loss}}+R\ln p}\) at 1 and 30 bar
- \(\Delta G^\circ(298\,\mathrm{K})=\Delta H_{\mathrm{corr}} - 298\,\Delta S_{\mathrm{loss}}\)

**Important**: the Stage 3 agent will deterministically **recompute tab4** for any system whose tab4 row is missing (or missing `Delta_H_plus25_96_kJ_mol`), using `E_ads_kJ_mol` from tab3.

---

## Stage 3: Agentic screening and decision support (chemist-agent)

The agent is implemented in `modules/chemist_agent.py` and invoked by `scripts/run_chemist_agent.py`.

Key design constraints (mirrors the paper):
- the agent reads only tab1–tab4 and builds a compact structured JSON payload
- the prompt requires explicit citation of numeric fields (E_ads, ΔH_corr, ΔG°, T50)
- the output is fixed-format markdown: executive summary → ranked shortlist → per-system notes → follow-ups
- each run records provenance metadata; repeated runs support stability statistics (Top‑k rates, rank correlations, Kendall’s \(\tau\))

Example (subset of systems):

```bash
python scripts/run_chemist_agent.py --systems Ca-TiO2,Sr-TiO2,Ba-TiO2 --llm ollama --model llama3.2:3b
```

---

## Minimal API notes (for scripting)

Everything is re-exported from `modules/__init__.py`, so you can do:

```python
from modules import (
    ElectronicProperties,
    AdsorptionMetrics,
    AdsorptionEnergies,
    ThermodynamicProjections,
    analyze_reversible_storage_screening,
)
```
