"""Unified CLI for GATOR: Gate-Augmented Thermodynamic Ordering and Retrieval.

Usage:
    python -m gator screen [options]       # Run the screening pipeline
    python -m gator corpus search QUERY    # Search for literature
    python -m gator corpus approve         # Approve papers for indexing
    python -m gator corpus index           # Index approved papers
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="gator",
        description="GATOR: Gate-Augmented Thermodynamic Ordering and Retrieval",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- screen subcommand ---
    screen_parser = subparsers.add_parser("screen", help="Run the gated screening pipeline")
    screen_parser.add_argument(
        "--config",
        type=str,
        default="configs/screening_config.yaml",
        help="Path to screening config YAML (default: configs/screening_config.yaml)",
    )
    screen_parser.add_argument(
        "--data",
        type=str,
        default="data/labels.csv",
        help="Path to labels CSV (default: data/labels.csv)",
    )
    screen_parser.add_argument(
        "--systems",
        type=str,
        default="",
        help="Comma-separated system names (default: all)",
    )
    screen_parser.add_argument(
        "--llm",
        type=str,
        default="ollama",
        help="LLM provider: ollama (default), openai, or transformers",
    )
    screen_parser.add_argument(
        "--model", type=str, default="", help="Model name (provider-specific)"
    )
    screen_parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    screen_parser.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="Per-run LLM timeout in seconds (default: 300)",
    )
    screen_parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries for transient timeouts (default: 2)",
    )
    screen_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="For --llm transformers: allow HF repos with custom code",
    )
    screen_parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs (default: 1)",
    )
    screen_parser.add_argument(
        "--outdir",
        type=str,
        default="results/gator_runs",
        help="Output directory for run logs (default: results/gator_runs)",
    )
    screen_parser.add_argument(
        "--tag", type=str, default="", help="Optional tag for batch folder name"
    )
    screen_parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(
            "Comma-separated model IDs for multi-model benchmark "
            "(e.g. anthropic/claude-opus-4.6,openai/gpt-5.4,google/gemini-3.1-pro-preview)"
        ),
    )
    screen_parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    screen_parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip RAG retrieval (gates + LLM only)",
    )
    screen_parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        default=True,
        help="Print report to stdout (default)",
    )
    screen_parser.add_argument(
        "--no-print",
        dest="do_print",
        action="store_false",
        help="Do not print to stdout",
    )
    screen_parser.add_argument(
        "--structured",
        action="store_true",
        help="Use two-call structured analysis mode (JSON analysis → constrained synthesis)",
    )

    # --- corpus subcommand ---
    corpus_parser = subparsers.add_parser("corpus", help="Manage the literature corpus")
    corpus_subparsers = corpus_parser.add_subparsers(dest="corpus_command", help="Corpus commands")
    search_parser = corpus_subparsers.add_parser("search", help="Search Semantic Scholar / arXiv")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    corpus_subparsers.add_parser("approve", help="Approve papers for indexing")
    corpus_subparsers.add_parser("index", help="Index approved papers")
    auto_parser = corpus_subparsers.add_parser(
        "auto", help="Auto-search, approve, and index literature"
    )
    auto_parser.add_argument(
        "--queries",
        type=str,
        default="",
        help="Comma-separated custom queries (default: domain-relevant H2/TiO2 queries)",
    )
    auto_parser.add_argument(
        "--limit", type=int, default=10, help="Max results per query per source (default: 10)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "screen":
        _run_screen(args)
    elif args.command == "corpus":
        _run_corpus(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_screen(args: argparse.Namespace) -> None:
    """Execute the screening pipeline."""
    # Lazy imports to keep CLI snappy
    from gator.gate_engine import load_config, load_records_from_csv
    from gator.screening_agent import run_multi_model_benchmark, run_screening_pipeline

    config = load_config(args.config)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()] or None
    records = load_records_from_csv(args.data, systems=systems)

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if models:
        run_multi_model_benchmark(
            config=config,
            records=records,
            models=models,
            llm=args.llm,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            retries=args.retries,
            trust_remote_code=args.trust_remote_code,
            runs=args.runs,
            outdir=args.outdir,
            tag=args.tag,
            use_tqdm=not args.no_tqdm,
            use_rag=not args.no_rag,
            do_print=args.do_print,
            structured=args.structured,
        )
    else:
        run_screening_pipeline(
            config=config,
            records=records,
            llm=args.llm,
            model=args.model.strip() or None,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            retries=args.retries,
            trust_remote_code=args.trust_remote_code,
            runs=args.runs,
            outdir=args.outdir,
            tag=args.tag,
            use_tqdm=not args.no_tqdm,
            use_rag=not args.no_rag,
            do_print=args.do_print,
            structured=args.structured,
        )


def _run_corpus(args: argparse.Namespace) -> None:
    """Execute corpus management commands."""
    if args.corpus_command is None:
        print("Usage: python -m gator corpus {search|approve|index}")
        sys.exit(1)

    from gator.corpus_builder import (
        approve_papers,
        auto_build_corpus,
        index_corpus,
        search_literature,
    )
    from gator.gate_engine import load_config

    if args.corpus_command == "search":
        search_literature(args.query, limit=args.limit)
    elif args.corpus_command == "approve":
        approve_papers()
    elif args.corpus_command == "index":
        # Thread embedding model from config if available
        try:
            config = load_config("configs/screening_config.yaml")
            index_corpus(embedding_model=config.rag_embedding_model)
        except FileNotFoundError:
            index_corpus()
    elif args.corpus_command == "auto":
        queries = [q.strip() for q in args.queries.split(",") if q.strip()] or None
        try:
            config = load_config("configs/screening_config.yaml")
            auto_build_corpus(
                queries=queries, limit=args.limit, embedding_model=config.rag_embedding_model
            )
        except FileNotFoundError:
            auto_build_corpus(queries=queries, limit=args.limit)


if __name__ == "__main__":
    main()
