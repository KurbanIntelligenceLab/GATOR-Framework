# GATOR: Gate-Augmented Thermodynamic Ordering and Retrieval

A physics-gated screening pipeline with RAG for LLM-assisted thermodynamic evaluation of hydrogen storage materials.

GATOR replaces monolithic LLM prompts with a hybrid architecture: **configurable deterministic physics gates** handle binary decisions reproducibly, **RAG** retrieves experimental literature for grounding, and a **constrained LLM** synthesizes nuanced interpretation, comparative analysis, and experimental design suggestions.

---

## Pipeline

```
screening_config.yaml
        │
        ▼
Stage 1: Deterministic Gates
  Apply configurable physics gates to descriptor data
  Output: PhysicsProfile per system (pass / borderline / fail)
        │
        ▼
Stage 2: Literature Retrieval (RAG)
  Gate-informed queries → local vector store
  FAIL systems skip retrieval
        │
        ▼
Stage 3: LLM Synthesis
  Three-section prompt: IMMUTABLE FACTS → LITERATURE → INSTRUCTIONS
  Output: A. Comparative Analysis
          B. Constrained Ranking (respects gate verdicts)
          C. Experimental Design Suggestions
        │
        ▼
Stage 4: Post-hoc Validation
  Verify LLM ranking doesn't contradict gate verdicts
```

---

## Repository layout

```
src/gator/                     # Python package
├── gate_engine.py             # Configurable gate chain + PhysicsProfile
├── llm_providers.py           # Ollama / OpenAI / Transformers
├── screening_agent.py         # Pipeline orchestrator + prompt builder
├── literature_retrieval.py    # RAG query + ExperimentalContext
├── corpus_builder.py          # Semantic Scholar/arXiv search + indexing
├── cli.py                     # Unified CLI
├── electronic_calculations.py # HOMO/LUMO → conceptual DFT
├── adsorption_metrics.py      # XYZ → H-H bond + mode classification
├── adsorption_energies.py     # E_ads + DOE window
└── thermodynamic_projections.py # ΔH_corr, T₅₀, ΔG°

data/                          # Computed descriptors (unchanged)
├── labels.csv                 # Merged 29-column descriptor table
├── metrics_tab{1-4}.csv       # Individual descriptor tables
└── geometries/                # XYZ structure files

configs/
└── screening_config.yaml      # Gate thresholds, labels, mappings

literature/                    # Curated experimental papers (local)
├── papers/                    # PDFs
├── index/                     # ChromaDB vector store
└── approved.json              # Approved paper manifest

tests/                         # pytest test suite
```

---

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # core (gates + LLM screening) + dev tools
uv sync --extra rag              # with RAG support
uv sync --extra transformers     # with local HF model support
uv sync --all-extras             # everything
```

---

## Quickstart

### Run the screening pipeline

```bash
# Single run with Ollama (default)
uv run gator screen --llm ollama --model llama3.2:3b

# With a specific config
uv run gator screen --config configs/screening_config.yaml

# Subset of systems
uv run gator screen --systems Pristine,Mg-TiO2

# Batch mode (100 runs for stability analysis)
uv run gator screen --runs 100 --outdir results/gator_runs

# Skip RAG (gates + LLM only)
uv run gator screen --no-rag

# Using OpenAI
OPENAI_API_KEY=... uv run gator screen --llm openai
```

### Manage the literature corpus

```bash
# Search for papers
uv run gator corpus search "H2 adsorption TiO2 nanoparticle"

# Approve papers interactively
uv run gator corpus approve

# Index approved papers into vector store
uv run gator corpus index
```

---

## Configuration

All physics thresholds, labels, and screening mappings are in `configs/screening_config.yaml`. No magic numbers in code.

**Adding a new gate** = adding a YAML block. **Changing a threshold** = editing one number. **Switching entropy correction** = changing a column name.

### Gate types

| Type | Description | Example |
|------|-------------|---------|
| `range_multi` | N thresholds → N+1 bins | Mode: [0.80, 0.95] → molecular/activated/dissociative |
| `threshold` | Single comparison | ΔG° < 0 → spontaneous/non-spontaneous |
| `doe_window` | Inside/outside a range | \|E_ads\| in [15, 25] kJ/mol |
| `entropy_corrected_doe` | DOE with endothermic handling | ΔH+25.96: Endothermic/Inside/Outside |

### Key configurable parameters

| Parameter | Config key | Default |
|-----------|-----------|---------|
| Mode thresholds (Å) | `gates[adsorption_mode].thresholds` | [0.80, 0.95] |
| Regime thresholds (kJ/mol) | `gates[regime].thresholds` | [18, 50, 100] |
| DOE window (kJ/mol) | `gates[doe_*].doe_bounds` | [15, 25] |
| Deliverability T window (K) | `gates[deliverability].thresholds` | [200, 400] |
| RAG top-k | `rag.top_k` | 5 |
| RAG embedding model | `rag.embedding_model` | all-MiniLM-L6-v2 |

---

## Python API

```python
from gator import load_config, load_records_from_csv, run_gates

config = load_config("configs/screening_config.yaml")
records = load_records_from_csv("data/labels.csv")
profiles = run_gates(config, records)

for p in profiles:
    print(f"{p.system}: {p.overall} (pass={p.pass_count}, flag={p.flag_count}, fail={p.fail_count})")
```

---

## Testing

```bash
uv sync
uv run pytest
```

---

## License

Apache 2.0
