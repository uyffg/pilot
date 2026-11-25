#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./launch_opd.sh [General|Specific] --agent-env /path/to/env [options]

Defaults: General, 4 GPUs, artifacts/<prompt>/full.
Use --smoke for a short run in separate smoke directories.
Runner options such as --out, --scratch and --gpus are forwarded unchanged.
Environment: MODEL_PATH, AGENT_ENV, OPD_PYTHON, OPD_FRONTEND_PYTHON, OPD_SCRATCH_ROOT.
EOF
  exit 0
fi

profile=General
if [[ $# -gt 0 && "$1" != -* ]]; then
  profile="$1"
  shift
fi
case "$profile" in
  General|Specific) ;;
  *) echo "Prompt must be General or Specific." >&2; exit 2 ;;
esac

phase=full
for arg in "$@"; do
  [[ "$arg" != "--smoke" ]] || phase=smoke
done

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scratch_root="${OPD_SCRATCH_ROOT:-${TMPDIR:-/tmp}/opd-${USER:-user}}"
args=(--config "$repo_dir/experiments/bcb_opd/$profile.json"
      --out "$repo_dir/artifacts/$profile/$phase"
      --scratch "$scratch_root/$profile/$phase")
if [[ -n "${AGENT_ENV:-}" ]]; then
  args+=(--agent-env "$AGENT_ENV")
fi

cd -- "$repo_dir"
exec "${OPD_PYTHON:-python3}" -m experiments.bcb_opd.run "${args[@]}" "$@"
