#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A lt200449
#SBATCH -J SOKEDTWView
#SBATCH --output="/project/lt200449-ttsign/01_SOKE/01_submitResult/R-%x-%j-dtwviewer.out"

# =============================================================================
# Build the DTW alignment viewer: evaluate -> render -> assemble one HTML.
#
# For every selected sample this computes DTW for all 3 parts x BOTH cost modes,
# renders an SMPL-X mesh thumbnail per reference and generated frame, and packs
# the lot into a single self-contained HTML you can scp back and open offline.
#
# Submit:  sbatch lanta/02_DTWViewer.sh
# Resume:  just submit again -- samples already complete are skipped.
# Redo:    FORCE=1 sbatch lanta/02_DTWViewer.sh
# =============================================================================
# Conda activation is deliberately not run under `set -u`; inherited cluster
# environments can reference unset variables. Run failures are collected so one
# bad run does not waste the remainder of the allocation.
set -o pipefail

PROJECT_ROOT=/project/lt200449-ttsign/01_SOKE
REPO_DIR="$PROJECT_ROOT/SOKE"
ENV_PREFIX="$PROJECT_ROOT/envs"

OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/02_result/dtw_viewer}"
DATA_DIR="$OUT_ROOT/data"
HTML_OUT="$OUT_ROOT/dtw_viewer.html"

# -----------------------------------------------------------------------------
# tag | results dir (relative to REPO_DIR) | recorded-mode | select
#
# BOTH cost modes are always computed -- every sample gets all 6 combinations
# (body/lhand/rhand x jpe/pa). The recorded-mode field does NOT choose what to
# evaluate. It names the run's METRIC.DTW_ALIGN_MODE so the extractor knows which
# of the two is the metric recorded in test_scores.json and reconciles that one;
# the other is kept as the diagnostic the viewer's PA section renders. The
# preflight below checks this field against the run's own saved config and
# refuses to start if they disagree.
#
# The results dir may be a single rank dir (.../test_rank_0) or a whole run dir,
# in which case every {SPLIT}_rank_* under it is discovered and their
# test_scores.json merged -- which is what you want for a DDP run, so that
# best/median/worst rank over the run rather than over one GPU's share.
#
# The tag prefixes every output file and labels the run in the viewer. It is not
# cosmetic: sample keys are NOT unique across runs (the Thai clips appear in both
# the Thai run and the 4-dataset run), so without distinct tags the second run
# would overwrite the first.
# -----------------------------------------------------------------------------
RUNS=(
  "thai|results/mgpt/SOKE-Thai-Hand4WholePP-From-Scratch/test_rank_0|jpe|best,median,worst"
  "multi|results/mgpt/SOKE-3-Dataset-One-Go-last-ckpt|jpe|best,median,worst"
)

SPLIT="${SPLIT:-test}"
TILE="${TILE:-128}"
JPEG_QUALITY="${JPEG_QUALITY:-72}"
DEVICE="${DEVICE:-cuda:0}"
CHUNK="${CHUNK:-32}"
FORCE="${FORCE:-0}"
MEAN_PATH="${MEAN_PATH:-../data/CSL-Daily/mean.pt}"
STD_PATH="${STD_PATH:-../data/CSL-Daily/std.pt}"

# --- environment -------------------------------------------------------------
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

export PYTHONPATH=.
export PYOPENGL_PLATFORM=egl          # headless mesh rendering; no X on compute nodes
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$DATA_DIR"

echo "=== node: $(hostname) | job: ${SLURM_JOB_ID:-none} | $(date) ==="
echo "repo:    $REPO_DIR"
echo "outputs: $OUT_ROOT"
echo "resume:  $([ "$FORCE" = "1" ] && echo 'OFF (FORCE=1, re-extracting everything)' \
                                   || echo 'ON (complete samples are skipped)')"
nvidia-smi
"$PY" -c 'import torch; print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| sm", torch.cuda.get_device_capability())' 2>/dev/null

# =============================================================================
# Preflight. Each of these has already cost a debugging session; all of them are
# cheap, and every one of them fails hundreds of log lines later if left to
# chance.
# =============================================================================
echo ""
echo "=== preflight ==="
PREFLIGHT_OK=1
fail() { echo "  FAIL: $*"; PREFLIGHT_OK=0; }

for f in "$MEAN_PATH" "$STD_PATH"; do
  [ -f "$f" ] && echo "  ok: $f" || fail "missing normalisation stats: $f"
done

# Headless render smoke test. Deliberately trimesh-free: trimesh 3.9.24 is broken
# under numpy 2 (ndarray.ptp was removed), so importing it dies before pyrender is
# ever exercised. The extractor builds Primitives directly for the same reason.
if "$PY" - <<'PY'
import sys
import numpy as np, pyrender
v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], np.float32)
f = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], np.int32)
prim = pyrender.Primitive(positions=v, indices=f, mode=4,
    material=pyrender.MetallicRoughnessMaterial(metallicFactor=0.0))
s = pyrender.Scene(); s.add(pyrender.Mesh(primitives=[prim]))
cam = pyrender.camera.IntrinsicsCamera(fx=200, fy=200, cx=32, cy=32)
p = np.eye(4); p[:3,3] = [0,0,3]; s.add(cam, pose=p)
s.add(pyrender.DirectionalLight(color=[1,1,1], intensity=300), pose=np.eye(4))
r = pyrender.OffscreenRenderer(64, 64); c, d = r.render(s); r.delete()
sys.exit(0 if (d > 0).any() else 1)
PY
then echo "  ok: pyrender EGL offscreen render"
else fail "pyrender EGL render failed -- rendering will not work on this node"
fi

for entry in "${RUNS[@]}"; do
  IFS='|' read -r tag rdir rmode rsel <<< "$entry"
  [ -n "$tag" ] || fail "run entry has an empty tag: $entry"

  if [ ! -d "$rdir" ]; then
    fail "[$tag] results dir not found: $rdir"
    continue
  fi

  # A rank dir holds test_scores.json directly; a run dir holds rank subdirs.
  if [ -f "$rdir/test_scores.json" ]; then
    run_root="$(dirname "$rdir")"
    echo "  ok: [$tag] single rank dir $rdir"
  else
    nrank=$(find "$rdir" -maxdepth 1 -type d -name "${SPLIT}_rank_*" \
              -exec test -f '{}/test_scores.json' \; -print | wc -l)
    if [ "$nrank" -eq 0 ]; then
      fail "[$tag] no test_scores.json in $rdir, and no ${SPLIT}_rank_*/ under it"
      continue
    fi
    run_root="$rdir"
    echo "  ok: [$tag] run dir with $nrank rank(s) under $rdir"
  fi

  # The recorded mode must match the config the run was actually produced with.
  # Get this wrong and every sample reports MISMATCH and is refused -- a failure
  # that only shows up after the SMPL-X forward has run for every sample.
  cfg=$(ls -1 "$run_root"/config_*_test.yaml 2>/dev/null | head -1)
  if [ -n "$cfg" ]; then
    # Strip the trailing comment BEFORE splitting on the colon: these lines read
    #   DTW_ALIGN_MODE: 'pa'  # Options: 'jpe' (align to root joint) or 'pa' (...)
    # and a greedy `.*:` would match the colon in "Options:" and return the comment.
    actual=$(grep -E '^[[:space:]]*DTW_ALIGN_MODE:' "$cfg" | head -1 \
             | sed -E "s/#.*//; s/^[^:]*:[[:space:]]*//; s/[\"']//g; s/[[:space:]]*$//")
    if [ -z "$actual" ]; then
      echo "  warn: [$tag] no DTW_ALIGN_MODE in $(basename "$cfg"); assuming '$rmode'"
    elif [ "$actual" != "$rmode" ]; then
      fail "[$tag] recorded-mode '$rmode' but the run's config says '$actual' ($(basename "$cfg"))"
    else
      echo "  ok: [$tag] DTW_ALIGN_MODE '$actual' confirmed from $(basename "$cfg")"
    fi
  else
    echo "  warn: [$tag] no config_*_test.yaml under $run_root; cannot verify recorded-mode '$rmode'"
  fi
done

if [ "$PREFLIGHT_OK" -ne 1 ]; then
  echo ""
  echo "preflight failed -- nothing was run."
  exit 1
fi

# =============================================================================
# Extract. One invocation per run, all writing into the shared $DATA_DIR so the
# build step produces a single viewer spanning every run.
# =============================================================================
FAILED=""
for entry in "${RUNS[@]}"; do
  IFS='|' read -r tag rdir rmode rsel <<< "$entry"
  [ -n "$rsel" ] || rsel="best,median,worst"

  echo ""
  echo "================ extract: $tag ================"
  echo "results-dir : $rdir"
  echo "select      : $rsel   recorded-mode: $rmode"

  extra=""
  [ "$FORCE" = "1" ] && extra="--force"

  if srun "$PY" scripts/03_dtw_extract.py \
      --results-dir "$rdir" \
      --run-tag "$tag" \
      --select "$rsel" \
      --recorded-mode "$rmode" \
      --split "$SPLIT" \
      --device "$DEVICE" \
      --chunk "$CHUNK" \
      --tile "$TILE" \
      --jpeg-quality "$JPEG_QUALITY" \
      --mean-path "$MEAN_PATH" \
      --std-path "$STD_PATH" \
      --out-dir "$DATA_DIR" \
      $extra
  then
    echo "--- $tag: OK"
  else
    rc=$?
    echo "--- $tag: extraction FAILED (rc=$rc)"
    FAILED="$FAILED $tag"
  fi
done

# =============================================================================
# Build. CPU-only, and worth doing even if a run failed: a viewer over whatever
# did extract beats no viewer at all.
# =============================================================================
echo ""
echo "================ build viewer ================"
if ! "$PY" scripts/03_dtw_build_viewer.py --data-dir "$DATA_DIR" --out "$HTML_OUT"; then
  echo "viewer build FAILED"
  FAILED="$FAILED build"
fi

echo ""
echo "=== done $(date) ==="
ls -lh "$HTML_OUT" 2>/dev/null
echo "samples in $DATA_DIR: $(ls -1 "$DATA_DIR"/*.npz 2>/dev/null | wc -l)"
du -sh "$DATA_DIR" 2>/dev/null

echo ""
echo "Pull the viewer back and open it locally:"
echo "  scp $USER@lanta.nstda.or.th:$HTML_OUT ."

if [ -n "$FAILED" ]; then
  echo ""
  echo "FAILED:$FAILED"
  exit 1
fi
