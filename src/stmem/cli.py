"""Offline CLI for the public memory-core demo. No perception models are loaded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stmem import QueryEngine, STMemParams, load_memory
from stmem.pipeline import build_memory, run_demo, view_counts


def _query(args, parser):
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be specified together")
    time_range = (args.start, args.end) if args.start is not None else None
    engine = QueryEngine(load_memory(args.memory))
    common = {"top_k": args.top_k, "scene": args.scene}
    if args.type == "nlq":
        return engine.query_nlq(
            args.text, object_id=args.object_id, time_range=time_range, **common
        )
    if args.type == "str":
        return engine.query_str(
            args.text, object_id=args.object_id, time_range=time_range, **common
        )
    if args.type == "lor":
        if time_range is not None:
            parser.error("LOR uses --as-of and --lookback, not --start and --end")
        return engine.query_lor(
            args.label,
            object_id=args.object_id,
            as_of=args.as_of,
            lookback_seconds=args.lookback,
            **common,
        )
    if not args.feature_file:
        parser.error("VQ2D requires --feature-file with feature_space and feature fields")
    data = json.loads(Path(args.feature_file).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not {"feature", "feature_space"} <= data.keys():
        parser.error("feature JSON must contain feature and feature_space")
    return engine.query_vq2d(
        data["feature"], feature_space=data["feature_space"], time_range=time_range, **common
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="stmem", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the synthetic CPU demo without model weights")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    build = commands.add_parser("build", help="Build memory from provided offline observations")
    build.add_argument("--observations", type=Path, required=True)
    build.add_argument(
        "--output", type=Path, required=True, help="New directory for memory and crops"
    )
    build.add_argument("--captions", choices=["provided", "template"], default="provided")
    inspect = commands.add_parser("inspect", help="Inspect the saved memory and five view counts")
    inspect.add_argument("--memory", type=Path, required=True)
    query = commands.add_parser("query", help="Run one query against a saved memory")
    query.add_argument("--memory", type=Path, required=True)
    query.add_argument("--type", choices=["nlq", "vq2d", "str", "lor"], required=True)
    query.add_argument("--text", default="")
    query.add_argument("--object-id")
    query.add_argument("--label")
    query.add_argument("--scene")
    query.add_argument("--start", type=float)
    query.add_argument("--end", type=float)
    query.add_argument("--lookback", type=float)
    query.add_argument("--as-of", type=float)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--feature-file", type=Path, help="JSON with feature_space and feature")
    verify = commands.add_parser(
        "verify", help="Check portable reload and four fresh-process queries"
    )
    verify.add_argument("--memory", type=Path, required=True)
    verify.add_argument("--object-id", default="cup_1")
    verify.add_argument("--text", default="cup")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = run_demo(args.output)
        elif args.command == "build":
            result = {
                "memory": str(
                    build_memory(
                        args.observations, args.output, params=STMemParams(captions=args.captions)
                    )
                )
            }
        elif args.command == "inspect":
            memory = load_memory(args.memory)
            result = {
                "video_id": memory.video_id,
                "feature_space": memory.feature_space,
                "provenance": memory.provenance,
                "views": view_counts(memory),
            }
        elif args.command == "verify":
            from stmem.verify import verify

            report = verify(args.memory, args.object_id, args.text)
            result = {
                "check": "Functional smoke test, not benchmark evaluation",
                "portable_reload": report["portable_reload"],
                "views": report["views"],
                "query_counts": {
                    key: len(value) for key, value in report["fresh_process_queries"].items()
                },
            }
        else:
            result = _query(args, parser)
    except (OSError, ValueError) as error:
        parser.exit(2, f"stmem: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
