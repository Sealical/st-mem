"""Argument registration for optional video stages without importing model runtimes."""


def add_commands(commands):
    video = commands.add_parser("video", help="Optional real-video perception stages")
    stages = video.add_subparsers(dest="stage", required=True)
    sample = stages.add_parser("sample", help="Create a bounded frame-aligned video bundle")
    sample.add_argument("--video", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--fps", type=float, default=2.0)
    sample.add_argument("--max-frames", type=int, default=32)
    sample.add_argument("--start-seconds", type=float, default=0.0)
    sample.add_argument("--max-size", type=int, default=640)

    vipe = stages.add_parser("vipe", help="Run geometry in the separate ViPE environment")
    vipe.add_argument("--video", required=True)
    vipe.add_argument("--output", required=True)
    vipe.add_argument("--depth-model", choices=["metric3d-small"], default="metric3d-small")
    vipe.add_argument("--seed", type=int, default=42)

    track = stages.add_parser("track", help="Run SAM3 and save video-bound tracking masks")
    track.add_argument("--video", required=True)
    track.add_argument("--checkpoint", required=True)
    track.add_argument("--output", required=True)
    track.add_argument("--prompts", nargs="+", required=True)
    track.add_argument("--device", default="cuda:0")
    track.add_argument("--seed", type=int, default=42)

    prepare = stages.add_parser("prepare", help="Combine SAM3, ViPE and DINOv2 observations")
    for name in ("video-bundle", "vipe-root", "output", "dinov2-checkpoint", "dinov2-repo"):
        prepare.add_argument(f"--{name}", required=True)
    prepare.add_argument("--sam3-checkpoint")
    prepare.add_argument("--tracking-cache")
    prepare.add_argument("--prompts", nargs="+", required=True)
    prepare.add_argument("--scene-label", default="unknown")
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--seed", type=int, default=42)


def run(args):
    options = {key: value for key, value in vars(args).items() if key not in {"command", "stage"}}
    if args.stage == "sample":
        from stmem.perception.video import sample_video

        return {"video_bundle": str(sample_video(**options))}
    if args.stage == "vipe":
        from stmem.perception.vipe import run_vipe

        return {"geometry_provenance": str(run_vipe(**options))}
    if args.stage == "prepare":
        from stmem.perception.prepare import prepare_observations

        return {"observations": str(prepare_observations(**options))}
    from pathlib import Path

    from stmem.perception.runtime import seed_everything
    from stmem.perception.sam3 import Sam3VideoTracker
    from stmem.perception.tracking import save_tracking_cache

    if Path(args.output).exists():
        raise FileExistsError("Choose a new tracking output directory")
    seed_everything(args.seed)
    tracker = Sam3VideoTracker(args.checkpoint, args.device)
    try:
        tracks = tracker.forward({"video": args.video, "prompts": args.prompts})["frames"]
        path = save_tracking_cache(tracks, args.video, args.output, args.prompts)
        return {"tracking": str(path), "frames": len(tracks)}
    finally:
        tracker.close()
