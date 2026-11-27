#!/usr/bin/env bash
# Submit a command as a Renku job (non-interactive session) of this project.
#
# Jobs start from a fresh clone of the default branch, so the job first checks
# out the pinned commit, syncs the uv env from that commit's lockfile, then runs
# the command from the repo root with CODIT_DATA_PATH on the persistent store.
# Output of the command and job metadata go to
#   $CODIT_DATA_PATH/jobs/<name>/{job.log,meta.txt}
# The commit must be on origin (the job fetches it from there).
# GPU jobs build a fresh venv (and uv cache) on the node disk under /tmp,
# because the CUDA wheels do not fit on the GPU launcher's work disk.
#
# Usage:
#   bash scripts/dev/renku/submit_job.sh --cpu --name bl-test-1 [--ref dev] \
#       [--extra parcels] [--wait] [--dry-run] -- python scripts/... --flag value
#
# The launcher's command is `uv run`; this script passes the job arguments.

set -euo pipefail

CPU_LAUNCHER="01M8E1GQXDGCYTG3C4RK35ACY3"
GPU_LAUNCHER="01M8E1E094GP2VMEAMQ79AS4GW"
JOB_REPO_DIR="/home/renku/work/CoDiT"
JOB_DATA_PATH="/home/renku/work/store/codit_data"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LAUNCHER=""
NAME=""
REF="HEAD"
EXTRAS=()
WAIT=0
DRY_RUN=0

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) LAUNCHER="$CPU_LAUNCHER"; shift ;;
    --gpu) LAUNCHER="$GPU_LAUNCHER"; EXTRAS+=(cuda); shift ;;
    --name) NAME="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --extra) EXTRAS+=("$2"); shift 2 ;;
    --wait) WAIT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h | --help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "submit_job: unknown argument $1" >&2; usage >&2; exit 1 ;;
  esac
done

[[ -n "$LAUNCHER" ]] || { echo "submit_job: pass --cpu or --gpu" >&2; exit 1; }
[[ $# -gt 0 ]] || { echo "submit_job: no command given after --" >&2; exit 1; }
if [[ ! "$NAME" =~ ^[a-z][a-z0-9-]{3,19}$ ]]; then
  echo "submit_job: --name must be 4-20 chars of [a-z0-9-], starting with a letter" >&2
  exit 1
fi

git -C "$REPO_ROOT" fetch --quiet origin
SHA="$(git -C "$REPO_ROOT" rev-parse --verify "$REF^{commit}")"
if [[ -z "$(git -C "$REPO_ROOT" branch -r --contains "$SHA")" ]]; then
  echo "submit_job: commit $SHA ($REF) is not on origin; push it first" >&2
  exit 1
fi

EXTRA_FLAGS=""
for e in "${EXTRAS[@]+"${EXTRAS[@]}"}"; do EXTRA_FLAGS+=" --extra $e"; done
CMD="$(printf '%q ' "$@")"
UV_ENV=""
if [[ "$LAUNCHER" == "$GPU_LAUNCHER" ]]; then
  UV_ENV=$'export UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/tmp/venv\nexport VIRTUAL_ENV=/tmp/venv PATH=/tmp/venv/bin:$PATH'
fi

read -r -d '' BOOTSTRAP <<EOF || true
set -euo pipefail
export CODIT_DATA_PATH=$JOB_DATA_PATH
OUT=\$CODIT_DATA_PATH/jobs/$NAME
mkdir -p "\$OUT"
exec > >(tee -a "\$OUT/job.log") 2>&1
{
  echo "name=$NAME"
  echo "commit=$SHA"
  echo "ref=$REF"
  echo "launcher=$LAUNCHER"
  echo "command=$CMD"
  echo "started=\$(date -Iseconds)"
} > "\$OUT/meta.txt"
trap 'rc=\$?; { echo "finished=\$(date -Iseconds)"; echo "exit_code=\$rc"; } >> "\$OUT/meta.txt"' EXIT
$UV_ENV
cd $JOB_REPO_DIR
git fetch --quiet origin
git checkout --quiet --detach $SHA
uv sync --frozen --inexact$EXTRA_FLAGS
$CMD
EOF

ARGS=(job start --launcher "$LAUNCHER" --submission-id "$NAME")
[[ $WAIT -eq 1 ]] && ARGS+=(--wait)
ARGS+=(-- bash -c "$BOOTSTRAP")

if [[ $DRY_RUN -eq 1 ]]; then
  echo "rnk job start --launcher $LAUNCHER --submission-id $NAME$([[ $WAIT -eq 1 ]] && echo ' --wait') -- bash -c <<'BOOTSTRAP'"
  echo "$BOOTSTRAP"
  echo "BOOTSTRAP"
  exit 0
fi

RNK="$(bash "$REPO_ROOT/scripts/dev/renku/ensure_rnk.sh")"
"$RNK" "${ARGS[@]}"
