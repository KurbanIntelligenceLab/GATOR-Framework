"""modules/chemist_agent.py

LLM "computational chemist" agent for thermodynamic screening.

This module:
- Loads/merges computed descriptors from data/metrics_tab{1..4}.csv
- Optionally recomputes thermodynamic projections from E_ads (kJ/mol)
- Sends a compact, structured payload to an LLM for expert-style analysis

Design goals:
- Keep dependencies minimal (stdlib + numpy already used elsewhere)
- Be provider-light: default local Ollama (open-weights models) via HTTP (stdlib),
  with optional OpenAI Responses API support.

Environment variables (OpenAI):
- OPENAI_API_KEY: required to call the API
- OPENAI_MODEL: optional (default: gpt-4.1-mini)
- OPENAI_BASE_URL: optional (default: https://api.openai.com/v1)

Environment variables (Ollama):
- OLLAMA_BASE_URL: optional (default: http://localhost:11434)
- OLLAMA_MODEL: optional (default: llama3.2:3b)

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from functools import lru_cache
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence
import time

import csv
import json
import os
import uuid
import urllib.error
import urllib.request

from .thermodynamic_projections import ThermodynamicProjections


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _index_by_system(rows: Sequence[Dict[str, str]], system_key: str = "System") -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        sys_name = (r.get(system_key) or "").strip()
        if not sys_name:
            continue
        out[sys_name] = r
    return out


@dataclass(frozen=True)
class SystemRecord:
    """Merged descriptor record for one system."""

    system: str
    tab1: Dict[str, Any]
    tab2: Dict[str, Any]
    tab3: Dict[str, Any]
    tab4: Dict[str, Any]

    def to_payload(self) -> Dict[str, Any]:
        """Compact payload with the most screening-relevant fields."""
        t3 = self.tab3
        t2 = self.tab2
        t4 = self.tab4
        t1 = self.tab1

        return {
            "system": self.system,
            "electronic": {
                "epsilon_homo_eV": t1.get("epsilon_HOMO_eV"),
                "epsilon_lumo_eV": t1.get("epsilon_LUMO_eV"),
                "band_gap_eV": t1.get("E_g_eV"),
                "ip_koop_eV": t1.get("IP_Koop_eV"),
                "ea_koop_eV": t1.get("EA_Koop_eV"),
                "chi_eV": t1.get("chi_eV"),
                "eta_eV": t1.get("eta_eV"),
                "omega_eV": t1.get("omega_eV"),
            },
            "adsorption_geometry": {
                "r_HH_A": t2.get("r_H-H_A"),
                "delta_r_HH_A": t2.get("Delta_r_H-H_A"),
                "percent_elongation": t2.get("percent_elongation"),
                "mode": t2.get("Mode"),
                "r_H_AE_A": t2.get("r_H-AE_A"),
                "r_O_AE_A": t2.get("R_O-AE_A"),
            },
            "adsorption_energetics": {
                "E_ads_eV": t3.get("E_ads_eV"),
                "E_ads_kJ_mol": t3.get("E_ads_kJ_mol"),
                "regime": t3.get("Regime"),
                "DOE_window_electronic": t3.get("DOE_window"),
            },
            "thermo_projections": {
                "Delta_H_plus6_kJ_mol": t4.get("Delta_H_plus6_kJ_mol"),
                "DOE_plus6": t4.get("DOE_plus6"),
                "Delta_H_plus25_96_kJ_mol": t4.get("Delta_H_plus25_96_kJ_mol"),
                "DOE_plus25_96": t4.get("DOE_plus25_96"),
                "T50_1bar_K": t4.get("T50_at_1bar_K"),
                "T50_30bar_K": t4.get("T50_at_30bar_K"),
                "Delta_G_std_298K_kJ_mol": t4.get("Delta_G_std_kJ_mol"),
            },
        }

def build_screening_payload(records: Sequence[SystemRecord]) -> Dict[str, Any]:
    """Build the structured JSON payload sent to the LLM."""
    return {
        "task": "Thermodynamic screening for reversible hydrogen storage on alkaline-earth modified TiO2 nanoparticles",
        "assumptions": {
            "goal": "near-ambient reversibility and practical operating window",
            "red_flags": [
                "too-strong binding (very negative E_ads / strongly negative ΔG°)",
                "dissociative adsorption (risk of irreversible hydride/chemisorption)",
                "too-weak binding (endothermic or positive ΔG° at 298 K)",
            ],
            "use_tabs": {
                "tab3": "electronic adsorption energy + DOE window (15–25 kJ/mol) based on |E_ads|",
                "tab4": "finite-T projections (ΔH_corr, T50 at 1/30 bar, ΔG° at 298 K)",
                "tab2": "H–H activation mode via bond elongation",
            },
        },
        "systems": [r.to_payload() for r in records],
    }


def load_system_records(
    tab1_path: str = "data/metrics_tab1.csv",
    tab2_path: str = "data/metrics_tab2.csv",
    tab3_path: str = "data/metrics_tab3.csv",
    tab4_path: str = "data/metrics_tab4.csv",
    systems: Optional[Sequence[str]] = None,
    recompute_tab4_if_missing: bool = True,
) -> List[SystemRecord]:
    """Load + merge tab1–tab4 by system name.

    Notes:
    - CSV parsing keeps unknown columns but normalizes numerics to float/None.
    - If `tab4` is missing for a system and `recompute_tab4_if_missing` is True,
      it will be recomputed from `E_ads_kJ_mol` using ThermodynamicProjections.
    """

    tab1 = _index_by_system(_read_csv_rows(tab1_path))
    tab2 = _index_by_system(_read_csv_rows(tab2_path))
    tab3 = _index_by_system(_read_csv_rows(tab3_path))
    tab4 = _index_by_system(_read_csv_rows(tab4_path))

    all_systems = sorted(set(tab1) | set(tab2) | set(tab3) | set(tab4))
    if systems is not None:
        wanted = {s.strip() for s in systems if s and s.strip()}
        all_systems = [s for s in all_systems if s in wanted]

    def _normalize_row(row: Dict[str, str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in row.items():
            if k == "System":
                out[k] = (v or "").strip()
                continue
            f = _safe_float(v)
            out[k] = f if f is not None else (v.strip() if isinstance(v, str) else v)
        return out

    records: List[SystemRecord] = []
    for sys_name in all_systems:
        t1 = _normalize_row(tab1.get(sys_name, {"System": sys_name}))
        t2 = _normalize_row(tab2.get(sys_name, {"System": sys_name}))
        t3 = _normalize_row(tab3.get(sys_name, {"System": sys_name}))
        t4 = _normalize_row(tab4.get(sys_name, {"System": sys_name}))

        if recompute_tab4_if_missing and ("Delta_H_plus25_96_kJ_mol" not in t4 or _safe_float(t4.get("Delta_H_plus25_96_kJ_mol")) is None):
            e_ads_kj = _safe_float(t3.get("E_ads_kJ_mol"))
            if e_ads_kj is not None:
                proj = ThermodynamicProjections(e_ads_kj_mol=float(e_ads_kj), material_name=sys_name).to_dict()
                t4 = {**t4, **proj}

        records.append(SystemRecord(system=sys_name, tab1=t1, tab2=t2, tab3=t3, tab4=t4))

    return records


def build_screening_prompt(records: Sequence[SystemRecord]) -> str:
    payload = build_screening_payload(records)

    return (
        "You are a careful computational chemist specializing in hydrogen adsorption thermodynamics on oxide nanoclusters.\n"
        "Analyze the provided computed descriptors and write a screening assessment for reversible H2 storage on AE-modified TiO2 nanoparticles.\n\n"
        "Requirements:\n"
        "- Be quantitative: refer to E_ads (kJ/mol), corrected ΔH, ΔG°(298 K), and T50 (1 bar and 30 bar).\n"
        "- Explicitly address reversibility risk using adsorption mode (molecular/activated/dissociative) and adsorption strength.\n"
        "- Rank the candidates for reversible storage and explain the ranking.\n"
        "- Call out any inconsistencies (e.g., electronic DOE window vs corrected enthalpy becoming endothermic).\n"
        "- Propose 3–6 concrete next computations/validation checks (e.g., NEB barriers, vibrational ZPE/entropy, multiple adsorption sites, coverage effects).\n\n"
        "Output format (markdown):\n"
        "## Executive summary\n"
        "## Ranked shortlist (best → worst)\n"
        "## Per-system notes (one subsection per system)\n"
        "## Follow-up computations\n\n"
        "Here is the data as JSON:\n"
        f"```json\n{json.dumps(payload, indent=2, sort_keys=False)}\n```\n"
    )


def _openai_extract_text(resp: Dict[str, Any]) -> str:
    # Responses API often includes top-level output_text.
    if isinstance(resp, dict) and isinstance(resp.get("output_text"), str):
        return resp["output_text"]

    # Otherwise attempt to assemble from output[] content blocks.
    chunks: List[str] = []
    out = resp.get("output") if isinstance(resp, dict) else None
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") in {"output_text", "text"} and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
    return "\n".join(chunks).strip()


def call_openai(
    prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: float = 90.0,
) -> str:
    """Call OpenAI Responses API via HTTPS (stdlib)."""

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    url = base_url.rstrip("/") + "/responses"
    body = {
        "model": model,
        "input": prompt,
    }

    req = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"OpenAI API error: HTTP {getattr(e, 'code', '???')}\n{detail}")

    resp = json.loads(raw)
    text = _openai_extract_text(resp)
    if not text:
        # Fall back to raw JSON so the user can debug the schema.
        return json.dumps(resp, indent=2, sort_keys=False)
    return text


def call_openai_raw(
    prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: float = 90.0,
) -> Dict[str, Any]:
    """Call OpenAI Responses API via HTTPS (stdlib) and return the parsed JSON response."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    url = base_url.rstrip("/") + "/responses"
    body = {
        "model": model,
        "input": prompt,
    }

    req = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"OpenAI API error: HTTP {getattr(e, 'code', '???')}\n{detail}")

    resp = json.loads(raw)
    if not isinstance(resp, dict):
        raise RuntimeError("Unexpected OpenAI response schema (expected JSON object)")
    return resp


def call_ollama(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    timeout_s: float = 180.0,
) -> str:
    """Call a local Ollama server (open-weights models) via HTTP (stdlib).

    Requires Ollama running locally (default): `ollama serve`
    """
    model = model or os.getenv("OLLAMA_MODEL") or "llama3.2:3b"
    base_url = base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"

    url = base_url.rstrip("/") + "/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }

    req = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        # Common Ollama case: model not pulled yet -> HTTP 404 with JSON {"error": "... not found"}
        hint = ""
        try:
            j = json.loads(detail)
            if isinstance(j, dict) and isinstance(j.get("error"), str) and "not found" in j["error"].lower():
                hint = (
                    "\nHint: the model isn't installed in Ollama yet.\n"
                    f"- Pull it: `ollama pull {model}`\n"
                    "- Or list installed: `ollama list`\n"
                    "- Or pick another model via --model / OLLAMA_MODEL\n"
                )
        except Exception:
            pass
        raise RuntimeError(f"Ollama API error: HTTP {getattr(e, 'code', '???')}\n{detail}{hint}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Could not reach Ollama server. Is it running?\n"
            "- Install: https://ollama.com/\n"
            "- Start server: `ollama serve`\n"
            f"- Base URL tried: {base_url}\n"
            f"Original error: {e}"
        )

    resp = json.loads(raw)
    text = resp.get("response") if isinstance(resp, dict) else None
    if isinstance(text, str) and text.strip():
        return text.strip()
    # Fall back to raw JSON so the user can debug the schema.
    return json.dumps(resp, indent=2, sort_keys=False)


def call_ollama_raw(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    timeout_s: float = 180.0,
) -> Dict[str, Any]:
    """Call a local Ollama server and return the parsed JSON response."""
    model = model or os.getenv("OLLAMA_MODEL") or "llama3.2:3b"
    base_url = base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"

    url = base_url.rstrip("/") + "/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }

    req = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        hint = ""
        try:
            j = json.loads(detail)
            if isinstance(j, dict) and isinstance(j.get("error"), str) and "not found" in j["error"].lower():
                hint = (
                    "\nHint: the model isn't installed in Ollama yet.\n"
                    f"- Pull it: `ollama pull {model}`\n"
                    "- Or list installed: `ollama list`\n"
                    "- Or pick another model via --model / OLLAMA_MODEL\n"
                )
        except Exception:
            pass
        raise RuntimeError(f"Ollama API error: HTTP {getattr(e, 'code', '???')}\n{detail}{hint}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Could not reach Ollama server. Is it running?\n"
            "- Install: https://ollama.com/\n"
            "- Start server: `ollama serve`\n"
            f"- Base URL tried: {base_url}\n"
            f"Original error: {e}"
        )

    resp = json.loads(raw)
    if not isinstance(resp, dict):
        raise RuntimeError("Unexpected Ollama response schema (expected JSON object)")
    return resp


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def call_transformers(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_new_tokens: int = 900,
    trust_remote_code: bool = False,
) -> str:
    """Run a local open-weight model via Hugging Face Transformers (optional dependency).

    Notes:
    - Requires `transformers` and `torch` installed.
    - If the model isn't already cached locally, Transformers may try to download it.
    - Some model repos require `trust_remote_code=True` (custom architectures/tokenizers).
    """
    model = model or os.getenv("HF_MODEL") or "Qwen/Qwen2.5-3B-Instruct"

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Transformers backend requires `transformers` + `torch`.\n"
            "Install (cpu): `pip install transformers torch`\n"
            "Install (apple silicon): `pip install transformers torch`\n"
            f"Original error: {e}"
        )

    # Prefer Apple Silicon GPU (MPS) when available; otherwise CPU.
    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    dtype_name = "float16" if device == "mps" else "float32"

    @lru_cache(maxsize=2)
    def _load_transformers(model_id: str, device_name: str, dtype_key: str, trust: bool):
        dtype = torch.float16 if (device_name == "mps" and dtype_key == "float16") else None
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=trust)
        mdl.eval()
        if device_name != "cpu":
            mdl.to(device_name)
        return tok, mdl

    tok, mdl = _load_transformers(model, device, dtype_name, bool(trust_remote_code))

    # Keep it simple: use raw prompt as a single turn.
    inputs = tok(prompt, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = mdl.generate(
            **inputs,
            do_sample=float(temperature) > 0,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tok.eos_token_id,
        )

    # Decode only the newly generated part when possible.
    gen_ids = out_ids[0]
    prompt_len = inputs["input_ids"].shape[-1]
    new_ids = gen_ids[prompt_len:] if gen_ids.shape[-1] > prompt_len else gen_ids
    text = tok.decode(new_ids, skip_special_tokens=True)
    return text.strip()


def run_reversible_storage_screening(
    systems: Optional[Sequence[str]] = None,
    llm: str = "ollama",
    model: Optional[str] = None,
    temperature: float = 0.2,
    include_prompt: bool = True,
    include_payload: bool = True,
    include_provider_response: bool = True,
    timeout_s: float = 300.0,
    retries: int = 2,
    retry_backoff_s: float = 2.0,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    """Run the screening agent once and return a structured record suitable for logging."""
    records = load_system_records(systems=systems)
    payload = build_screening_payload(records)
    prompt = build_screening_prompt(records)

    llm_norm = (llm or "").strip().lower()
    t0 = perf_counter()
    provider_resp = None
    output = ""
    model_used = model or ""

    # Retry wrapper for flaky/slow local inference calls.
    for attempt in range(max(0, int(retries)) + 1):
        try:
            if llm_norm == "openai":
                provider_resp = call_openai_raw(prompt, model=model, timeout_s=min(float(timeout_s), 300.0)) if include_provider_response else None
                output = _openai_extract_text(provider_resp) if isinstance(provider_resp, dict) else call_openai(prompt, model=model, timeout_s=min(float(timeout_s), 300.0))
                model_used = model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
            elif llm_norm == "ollama":
                provider_resp = call_ollama_raw(prompt, model=model, temperature=temperature, timeout_s=float(timeout_s)) if include_provider_response else None
                if isinstance(provider_resp, dict) and isinstance(provider_resp.get("response"), str):
                    output = str(provider_resp["response"]).strip()
                else:
                    output = call_ollama(prompt, model=model, temperature=temperature, timeout_s=float(timeout_s))
                model_used = model or os.getenv("OLLAMA_MODEL") or "llama3.2:3b"
            elif llm_norm == "transformers":
                provider_resp = None
                output = call_transformers(prompt, model=model, temperature=temperature, trust_remote_code=trust_remote_code)
                model_used = model or os.getenv("HF_MODEL") or "Qwen/Qwen2.5-3B-Instruct"
            else:
                raise ValueError(f"Unknown llm provider: {llm!r} (expected 'ollama', 'transformers', or 'openai')")
            break
        except Exception as e:
            # Only retry timeouts / transient URL errors.
            transient = isinstance(e, TimeoutError) or ("timed out" in str(e).lower())
            if llm_norm == "ollama" and transient and attempt < int(retries):
                time.sleep(float(retry_backoff_s) * (2**attempt))
                continue
            raise
    elapsed_s = perf_counter() - t0

    out_hash = sha256(output.encode("utf-8")).hexdigest()
    rec: Dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "created_at_utc": _utc_now_iso(),
        "llm": llm_norm,
        "model": model_used,
        "temperature": float(temperature),
        "timeout_s": float(timeout_s),
        "retries": int(retries),
        "systems": list(systems) if systems is not None else None,
        "systems_count": len(payload.get("systems", [])) if isinstance(payload.get("systems"), list) else None,
        "elapsed_s": float(elapsed_s),
        "output_sha256": out_hash,
        "output": output,
    }
    if include_prompt:
        rec["prompt"] = prompt
    if include_payload:
        rec["payload"] = payload
    if include_provider_response:
        rec["provider_response"] = provider_resp
    return rec


def analyze_reversible_storage_screening(
    systems: Optional[Sequence[str]] = None,
    llm: str = "ollama",
    model: Optional[str] = None,
    temperature: float = 0.2,
    timeout_s: float = 300.0,
    retries: int = 2,
    trust_remote_code: bool = False,
) -> str:
    """High-level helper: load records -> build prompt -> call LLM."""
    return run_reversible_storage_screening(
        systems=systems,
        llm=llm,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        retries=retries,
        trust_remote_code=trust_remote_code,
    )["output"]
