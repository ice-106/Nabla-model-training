import json
from pathlib import Path


def load_inference_manifest(path, src):
    """Load aligned names and texts for one source from a comparison manifest."""
    manifest_path = Path(path)
    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in inference manifest {manifest_path}: {exc}") from exc

    if not isinstance(records, list):
        raise ValueError(f"inference manifest {manifest_path} must contain a JSON list")

    required = {"id", "text", "src"}
    ids = []
    selected = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"manifest record {index} must be an object")
        missing = required - record.keys()
        if missing:
            raise ValueError(f"manifest record {index} missing fields: {sorted(missing)}")
        clip_id = str(record["id"])
        if not clip_id:
            raise ValueError(f"manifest record {index} has an empty id")
        ids.append(clip_id)
        if record["src"] == src:
            selected.append((clip_id, str(record["text"])))

    duplicates = sorted({clip_id for clip_id in ids if ids.count(clip_id) > 1})
    if duplicates:
        raise ValueError(f"inference manifest contains duplicate ids: {duplicates}")
    if not selected:
        raise ValueError(f"inference manifest has no records for source {src!r}")

    names, texts = zip(*selected)
    return list(texts), list(names)
