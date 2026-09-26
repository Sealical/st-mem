"""ST-Mem core demo and optional local video perception. Models load only on request."""

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
    if bool(args.feature_file) == bool(args.image):
        parser.error("VQ2D requires exactly one of --feature-file or --image")
    if args.feature_file:
        data = json.loads(Path(args.feature_file).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not {"feature", "feature_space"} <= data.keys():
            parser.error("feature JSON must contain feature and feature_space")
    else:
        if not args.dinov2_checkpoint or not args.dinov2_repo:
            parser.error("--image requires --dinov2-checkpoint and --dinov2-repo")
        from stmem.dataset import load_rgb
        from stmem.perception.dinov2 import DINOv2Encoder
        from stmem.perception.runtime import seed_everything

        seed_everything(42)
        encoder = DINOv2Encoder(args.dinov2_checkpoint, device=args.device, repo=args.dinov2_repo)
        try:
            data = {
                "feature": encoder.forward({"image": load_rgb(args.image)})["embedding"],
                "feature_space": encoder.feature_space,
            }
        finally:
            encoder.close()
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
    build.add_argument("--captions", choices=["provided", "template", "vlm"], default="provided")
    build.add_argument("--vlm-checkpoint", help="Complete local Qwen3-VL model snapshot")
    build.add_argument("--device", default="cuda:0")
    build.add_argument("--max-new-tokens", type=int, default=100)
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
    query.add_argument("--image", type=Path, help="Object crop to encode with DINOv2")
    query.add_argument("--dinov2-checkpoint")
    query.add_argument("--dinov2-repo")
    query.add_argument("--device", default="cuda:0")
    verify = commands.add_parser(
        "verify", help="Check portable reload and four fresh-process queries"
    )
    verify.add_argument("--memory", type=Path, required=True)
    verify.add_argument("--object-id", default="cup_1")
    verify.add_argument("--text", default="cup")
    from stmem.perception.cli import add_commands

    add_commands(commands)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = run_demo(args.output)
        elif args.command == "build":
            if (args.captions == "vlm") != bool(args.vlm_checkpoint):
                parser.error("Use --captions vlm together with --vlm-checkpoint")
            if args.max_new_tokens < 1:
                parser.error("--max-new-tokens must be positive")
            if args.output.exists():
                raise FileExistsError("Choose a new memory output directory")
            captioner = None
            try:
                if args.captions == "vlm":
                    from stmem.perception.captioner import MotionCaptioner
                    from stmem.perception.runtime import seed_everything

                    seed_everything(42)
                    captioner = MotionCaptioner(
                        args.vlm_checkpoint, args.device, max_new_tokens=args.max_new_tokens
                    )
                result = {
                    "memory": str(
                        build_memory(
                            args.observations,
                            args.output,
                            params=STMemParams(captions=args.captions),
                            captioner=captioner,
                        )
                    )
                }
            finally:
                if captioner is not None:
                    captioner.close()
        elif args.command == "video":
            from stmem.perception.cli import run

            result = run(args)
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
    except ImportError as error:
        parser.exit(
            2, f"stmem: optional runtime is unavailable ({error}). See docs/video-demo.md.\n"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
