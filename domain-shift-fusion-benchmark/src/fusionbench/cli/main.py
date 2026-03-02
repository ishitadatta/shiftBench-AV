from __future__ import annotations

import argparse
import json
import sys

from fusionbench.bench.runner import benchmark_result_to_dict, run_benchmark
from fusionbench.core.utils import read_json, write_json
from fusionbench.data.sqlite_store import create_demo_database


def _cmd_run(args: argparse.Namespace) -> int:
    config = read_json(args.config)
    result = run_benchmark(config)
    payload = benchmark_result_to_dict(result)
    write_json(args.output, payload)

    print(json.dumps(payload, indent=2))
    print(f"\nWrote benchmark results to: {args.output}")
    return 0


def _cmd_make_demo_db(args: argparse.Namespace) -> int:
    db_path = create_demo_database(db_path=args.db_path, n_samples=args.n_samples, seed=args.seed)
    print(f"Demo database created at: {db_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fusionbench",
        description="Domain-shift and uncertainty-aware sensor-fusion robustness benchmarking",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run benchmark from JSON config")
    run_parser.add_argument("--config", required=True, help="Path to benchmark config JSON")
    run_parser.add_argument("--output", required=True, help="Path to output JSON")
    run_parser.set_defaults(func=_cmd_run)

    db_parser = subparsers.add_parser("make-demo-db", help="Create a synthetic SQLite dataset for benchmarking")
    db_parser.add_argument("--db-path", required=True, help="Output SQLite DB path")
    db_parser.add_argument("--n-samples", type=int, default=1500, help="Number of synthetic samples")
    db_parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    db_parser.set_defaults(func=_cmd_make_demo_db)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

