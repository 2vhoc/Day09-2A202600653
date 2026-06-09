from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.graph import ShoppingAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Student scaffold CLI.")
    parser.add_argument("--question", help="Run one question through the graph.")
    parser.add_argument("--test-file", default="data/test.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trace-file", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    assistant = ShoppingAssistant()

    if args.batch:
        output_dir = Path(args.output_dir) if args.output_dir else assistant.settings.traces_dir
        summary = assistant.run_batch(
            Path(args.test_file),
            output_dir,
            rebuild_index=args.rebuild_index,
        )
        if args.as_json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return

        print(f"Total: {summary['total']}")
        print(f"Route matches: {summary['route_matches']}/{summary['total']}")
        print(f"Status matches: {summary['status_matches']}/{summary['total']}")
        print(f"Trace directory: {output_dir}")
        mismatches = [
            row
            for row in summary["results"]
            if not row["route_match"] or not row["status_match"]
        ]
        if mismatches:
            print("Mismatches:")
            for row in mismatches:
                print(
                    "- {id}: route {actual_route} vs {expected_route}; "
                    "status {actual_status} vs {expected_status}".format(**row)
                )
        return

    if args.question:
        trace_file = Path(args.trace_file) if args.trace_file else None
        result = assistant.ask(
            args.question,
            trace_file=trace_file,
            rebuild_index=args.rebuild_index,
        )
        if args.as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        print(f"Status: {result['status']}")
        print(f"Route: {result['route'].get('selected_workers', [])}")
        print(result["final_answer"])
        if trace_file:
            print(f"Trace: {trace_file}")
        return

    parser.error("Provide --question or --batch.")


if __name__ == "__main__":
    main()
