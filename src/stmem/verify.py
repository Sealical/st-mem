"""Functional acceptance check: copy a memory and run four queries in fresh processes.

This is a smoke check, not a benchmark evaluator. VQ2D uses an existing anchor as
its query to check feature-space consistency, not generalization accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from stmem import MemoryViews, load_memory


def verify(memory_path: Path, object_id: str, text: str) -> dict:
    path = memory_path.resolve()
    if path.is_dir():
        path = path / "memory.json"
    memory = load_memory(path)
    obj = memory.objects[object_id]
    anchor = next(a for a in obj.visual_anchors if a.feature is not None)
    views = MemoryViews(memory)
    counts = {
        name: len(getattr(views, name))
        for name in ("objects", "scenes", "texts", "events", "images")
    }
    if not all(counts.values()):
        raise ValueError("acceptance fixture must exercise all five views")
    with tempfile.TemporaryDirectory(prefix="stmem-replay-") as temporary:
        root = Path(temporary)
        shutil.copytree(path.parent, root / "portable")
        portable = root / "portable" / path.name
        if load_memory(portable) != memory:
            raise AssertionError("portable memory differs after JSON reload")
        feature_path = root / "query_feature.json"
        feature_path.write_text(
            json.dumps({"feature_space": memory.feature_space, "feature": anchor.feature})
        )
        arguments = {
            "nlq": ["--text", text, "--object-id", object_id],
            "vq2d": ["--feature-file", str(feature_path)],
            "str": ["--object-id", object_id],
            "lor": ["--object-id", object_id],
        }
        results = {}
        for query_type, extra in arguments.items():
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "stmem.cli",
                    "query",
                    "--memory",
                    str(portable),
                    "--type",
                    query_type,
                    *extra,
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=root,
            )
            results[query_type] = json.loads(process.stdout)
            if not results[query_type]:
                raise AssertionError(f"{query_type} returned no results")
        if results["lor"][0]["timestamp_s"] != obj.last_seen:
            raise AssertionError("LOR does not point to the actual last observation")
        if results["vq2d"][0]["score"] < 0.999:
            raise AssertionError("VQ2D exact-feature match failed")
    return {
        "check": "portable four-query smoke; not benchmark evaluation",
        "memory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "views": counts,
        "portable_reload": "passed",
        "fresh_process_queries": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument(
        "--text", required=True, help="Text expected to match this fixture's captions"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("Choose a new report path")
    report = verify(args.memory, args.object_id, args.text)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
