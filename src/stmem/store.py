"""Versioned JSON memory storage. No pickle or model state is loaded."""

from pathlib import Path

from stmem.schema import STMemory


def save_memory(memory: STMemory, path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"memory already exists: {path}")
    # Validate links again after any caller-side mutation.
    payload = STMemory.model_validate(memory.model_dump()).model_dump_json(indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")
    temporary.replace(path)
    return path


def load_memory(path: str | Path, *, validate_assets=True) -> STMemory:
    path = Path(path).resolve()
    if path.is_dir():
        path /= "memory.json"
    memory = STMemory.model_validate_json(path.read_text(encoding="utf-8"))
    for obj in memory.objects.values():
        for anchor in obj.visual_anchors:
            if anchor.crop_path:
                asset = (path.parent / anchor.crop_path).resolve()
                if not asset.is_relative_to(path.parent):
                    raise ValueError("memory crop path escapes its directory")
                if validate_assets and not asset.is_file():
                    raise FileNotFoundError(f"missing memory crop: {asset}")
    return memory
