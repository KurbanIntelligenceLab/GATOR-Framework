"""scripts/run_chemist_agent.py

CLI entrypoint for the LLM computational-chemist screening agent.

Examples:
  # Analyze all systems using local Ollama (free/open-weights model)
  python scripts/run_chemist_agent.py --llm ollama --model llama3.2:3b

  # Analyze all systems using OpenAI (optional)
  OPENAI_API_KEY=... python scripts/run_chemist_agent.py --llm openai --model gpt-4.1-mini

  # Analyze a subset
  python scripts/run_chemist_agent.py --systems Ca-TiO2,Sr-TiO2,Ba-TiO2

  # Run and record 100 repeats for analysis
  python scripts/run_chemist_agent.py --llm ollama --model llama3.2:3b --runs 100 --outdir results/chemist_agent_runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import traceback

# Ensure repo root is on sys.path so `import modules` works when running as a script:
#   python scripts/run_chemist_agent.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.chemist_agent import (  # noqa: E402
    analyze_reversible_storage_screening,
    build_screening_payload,
    build_screening_prompt,
    load_system_records,
    run_reversible_storage_screening,
)


def main() -> None:
    p = argparse.ArgumentParser(description="LLM screening report for reversible H2 storage on AE-modified TiO2 NPs")
    p.add_argument("--systems", type=str, default="", help="Comma-separated system names (default: all)")
    p.add_argument("--llm", type=str, default="ollama", help="LLM provider: ollama (default), transformers, or openai")
    p.add_argument("--model", type=str, default="", help="Model name (provider-specific; default via env/module)")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (used by ollama; default: 0.2)")
    p.add_argument("--timeout-s", type=float, default=300.0, help="Per-run LLM request timeout in seconds (default: 300)")
    p.add_argument("--retries", type=int, default=2, help="Retries for transient timeouts (default: 2)")
    p.add_argument("--trust-remote-code", action="store_true", help="For --llm transformers: allow Hugging Face model repos with custom code")
    p.add_argument("--runs", type=int, default=1, help="Number of repeated runs to execute and record (default: 1)")
    p.add_argument("--outdir", type=str, default="results/chemist_agent_runs", help="Output directory for run logs")
    p.add_argument("--tag", type=str, default="", help="Optional tag to include in the batch folder name")
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar even if installed")
    p.add_argument("--print", dest="do_print", action="store_true", help="Print the last run's report to stdout")
    p.add_argument("--no-print", dest="do_print", action="store_false", help="Do not print to stdout")
    p.set_defaults(do_print=True)
    args = p.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()] or None
    model = args.model.strip() or None
    llm = (args.llm or "").strip() or "ollama"
    runs = int(args.runs)
    if runs < 1:
        raise SystemExit("--runs must be >= 1")

    # Single-run convenience: behave like the original script.
    if runs == 1:
        report = analyze_reversible_storage_screening(
            systems=systems,
            llm=llm,
            model=model,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            retries=args.retries,
            trust_remote_code=args.trust_remote_code,
        )
        if args.do_print:
            print(report)
        return

    # Multi-run mode: record each run for later analysis.
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag.strip()}" if args.tag.strip() else ""
    batch_dir = outdir / f"{ts}{tag}_{llm}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_at_utc": ts,
        "llm": llm,
        "model": model,
        "temperature": float(args.temperature),
        "trust_remote_code": bool(args.trust_remote_code),
        "systems": systems,
        "runs": runs,
        "cwd": os.getcwd(),
        "batch_dir": str(batch_dir),
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")

    # Save the exact prompt + payload once per batch (keeps per-run files compact),
    # without calling any model (important for offline analysis/repro).
    records = load_system_records(systems=systems)
    (batch_dir / "prompt.md").write_text(build_screening_prompt(records), encoding="utf-8")
    (batch_dir / "payload.json").write_text(
        json.dumps(build_screening_payload(records), indent=2, sort_keys=False),
        encoding="utf-8",
    )

    jsonl_path = batch_dir / "runs.jsonl"
    last: Optional[Dict[str, Any]] = None

    # Optional progress bar.
    tqdm_iter = None
    if not args.no_tqdm:
        try:
            from tqdm import tqdm  # type: ignore

            tqdm_iter = tqdm
        except Exception:
            tqdm_iter = None

    with jsonl_path.open("w", encoding="utf-8") as f:
        iterator = range(runs)
        if tqdm_iter is not None:
            iterator = tqdm_iter(iterator, total=runs, desc="chemist_agent runs", unit="run")

        for i in iterator:
            try:
                rec = run_reversible_storage_screening(
                    systems=systems,
                    llm=llm,
                    model=model,
                    temperature=args.temperature,
                    include_prompt=False,
                    include_payload=False,
                    include_provider_response=True,
                    timeout_s=args.timeout_s,
                    retries=args.retries,
                    trust_remote_code=args.trust_remote_code,
                )
            except Exception as e:
                rec = {
                    "run_id": f"error_{i:04d}",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "llm": llm,
                    "model": model,
                    "temperature": float(args.temperature),
                    "timeout_s": float(args.timeout_s),
                    "retries": int(args.retries),
                    "trust_remote_code": bool(args.trust_remote_code),
                    "systems": systems,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                }
            last = rec

            # Persist per-run JSON + a convenient markdown copy of the report.
            run_stub = f"run_{i:04d}_{rec['run_id']}"
            (batch_dir / f"{run_stub}.json").write_text(json.dumps(rec, indent=2, sort_keys=False), encoding="utf-8")
            (batch_dir / f"{run_stub}.md").write_text(rec.get("output", ""), encoding="utf-8")

            # Append to JSONL (easy for pandas/duckdb).
            f.write(json.dumps(rec, sort_keys=False) + "\n")
            f.flush()

            # Minimal progress to stderr-like behavior (keeps stdout clean for piping).
            if tqdm_iter is None:
                print(f"[{i+1}/{runs}] wrote {run_stub}", file=sys.stderr)
            else:
                # Show the latest run id in the tqdm postfix without spamming stdout.
                try:
                    rid = str(rec.get("run_id", ""))[:8]
                    if "error" in rec:
                        rid = "ERR"
                    iterator.set_postfix_str(rid)  # type: ignore[attr-defined]
                except Exception:
                    pass

    if args.do_print and last is not None:
        print(last.get("output", ""))


if __name__ == "__main__":
    main()
