#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200449
#SBATCH -J SOKECompare
#SBATCH --output="/project/lt200449-ttsign/01_SOKE/01_submitResult/R-%x-%j-compare.out"

# =============================================================================
# Run SOKE text-only and matched-KWS inference for the shared 5+5 manifest.
# Prereq: generate manifest.json with SignSparK tools/dump_manifest.py.
# Submit: sbatch lanta/01_CompareSubset.sh
# =============================================================================
# Conda activation is deliberately not run under `set -u`; inherited cluster
# environments can reference unset variables. Cell failures are collected so a
# single failure does not waste the remainder of the allocation.
set -o pipefail

PROJECT_ROOT=/project/lt200449-ttsign/01_SOKE
REPO_DIR="$PROJECT_ROOT/SOKE"
ENV_PREFIX="$PROJECT_ROOT/envs"
MANIFEST="${MANIFEST:-/project/lt200449-ttsign/04_SignSparK/02_result/compare/manifest.json}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/02_result/compare}"
SOURCES="${SOURCES:-csl how2sign}"
SCENARIOS="${SCENARIOS:-nokws kws}"

module load Mamba/23.11.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
_guard=0
while [ -n "${CONDA_PREFIX:-}" ] && [ "$_guard" -lt 8 ]; do
  conda deactivate 2>/dev/null || break
  _guard=$((_guard + 1))
done
conda activate "$ENV_PREFIX" || { echo "conda activate failed"; exit 1; }

PY="$ENV_PREFIX/bin/python"
[ -x "$PY" ] || { echo "no python at $PY"; exit 1; }
cd "$REPO_DIR" || { echo "no repo at $REPO_DIR"; exit 1; }
[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST"; exit 1; }

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYOPENGL_PLATFORM=egl

mkdir -p "$OUT_ROOT"

echo "=== node: $(hostname) | job: ${SLURM_JOB_ID:-none} | $(date) ==="
echo "manifest: $MANIFEST"
echo "outputs:  $OUT_ROOT"
nvidia-smi
"$PY" -c 'import torch; print("torch", torch.__version__, "| cuda", torch.version.cuda, "| sm", torch.cuda.get_device_capability())'

# Validate the comparison contract before spending GPU time: five unique clips
# per language and complete test-split retrieval coverage.
"$PY" - "$MANIFEST" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

manifest_path = Path(sys.argv[1])
records = json.loads(manifest_path.read_text(encoding="utf-8"))
required = {"id", "text", "src"}
for index, record in enumerate(records):
    missing = required - record.keys()
    if missing:
        raise SystemExit(f"manifest record {index} missing: {sorted(missing)}")

counts = Counter(record["src"] for record in records)
for src in ("csl", "how2sign"):
    if counts[src] != 5:
        raise SystemExit(f"expected 5 {src} records, found {counts[src]}")

ids = [str(record["id"]) for record in records]
if len(ids) != len(set(ids)):
    raise SystemExit("manifest contains duplicate clip IDs")

lookup = json.loads(Path("scripts/name2kws_test.json").read_text(encoding="utf-8"))
missing = [clip_id for clip_id in ids if clip_id not in lookup]
if missing:
    raise SystemExit(f"IDs missing from name2kws_test.json: {missing}")
print(f"manifest validation: OK ({len(records)} records, complete KWS coverage)")
PY
if [ "$?" -ne 0 ]; then
  echo "manifest validation failed"
  exit 1
fi

read_field() {
  src="$1"
  field="$2"
  "$PY" - "$MANIFEST" "$src" "$field" <<'PY'
import json
import sys
from pathlib import Path

records = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
src, field = sys.argv[2], sys.argv[3]
for record in records:
    if record["src"] == src:
        sys.stdout.buffer.write(str(record[field]).encode("utf-8") + b"\0")
PY
}

rename_outputs() {
  src="$1"
  out_dir="$2"
  "$PY" - "$MANIFEST" "$src" "$out_dir" <<'PY'
import json
import re
import sys
from pathlib import Path

records = [
    record
    for record in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if record["src"] == sys.argv[2]
]
out_dir = Path(sys.argv[3])
for index, record in enumerate(records):
    prefix = f"infer_{index}_"
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", str(record["id"]))
    for extension in ("pkl", "mp4"):
        matches = list(out_dir.glob(f"{prefix}*.{extension}"))
        if len(matches) != 1:
            raise SystemExit(
                f"expected one {prefix}*.{extension} in {out_dir}, found {len(matches)}"
            )
        target = out_dir / f"{safe_id}_{matches[0].name[len(prefix):]}"
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target}")
        matches[0].rename(target)
print(f"renamed {len(records)} output pairs with clip IDs in {out_dir}")
PY
}

FAILED=""
for src in $SOURCES; do
  mapfile -d '' -t IDS < <(read_field "$src" id)
  mapfile -d '' -t TEXTS < <(read_field "$src" text)
  if [ "${#IDS[@]}" -ne 5 ] || [ "${#TEXTS[@]}" -ne 5 ]; then
    echo "ERROR: expected five IDs/texts for $src"
    exit 1
  fi

  for scen in $SCENARIOS; do
    case "$scen" in
      nokws) cfg=configs/soke-01-full-dataset.yaml ;;
      kws)   cfg=configs/soke-01-full-dataset-kws.yaml ;;
      *)
        echo "ERROR: unsupported scenario '$scen' (expected: nokws or kws)"
        exit 2
        ;;
    esac

    out_dir="$OUT_ROOT/${src}_${scen}"
    if [ -d "$out_dir" ] && find "$out_dir" -mindepth 1 -print -quit | grep -q .; then
      echo "ERROR: output directory is not empty: $out_dir"
      echo "Choose a new OUT_ROOT to avoid overwriting prior results."
      exit 1
    fi
    mkdir -p "$out_dir"

    echo ""
    echo "================ src: $src | scenario: $scen ================"
    if srun "$PY" -m test \
        --cfg "$cfg" \
        --infer \
        --src "$src" \
        --text "${TEXTS[@]}" \
        --name "${IDS[@]}" \
        --output_dir "$out_dir" \
        --fps 20 \
        --use_gpus 0 \
        --device 0
    then
      if rename_outputs "$src" "$out_dir"; then
        echo "--- $src/$scen: OK"
      else
        rc=$?
        echo "--- $src/$scen: output verification FAILED (rc=$rc)"
        FAILED="$FAILED $src/$scen"
      fi
    else
      rc=$?
      echo "--- $src/$scen: inference FAILED (rc=$rc)"
      FAILED="$FAILED $src/$scen"
    fi
  done
done

echo ""
echo "=== done $(date) ==="
find "$OUT_ROOT" -maxdepth 2 -type f \( -name '*.mp4' -o -name '*.pkl' \) -print | sort

if [ -n "$FAILED" ]; then
  echo "FAILED cells:$FAILED"
  exit 1
fi
