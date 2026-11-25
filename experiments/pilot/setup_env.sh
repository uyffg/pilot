#!/usr/bin/env bash
# Build what one node needs before any run. Idempotent: every part has a completion marker and
# is skipped when it is already built from the same package list.
#
#   bash experiments/pilot/setup_env.sh                 # everything, in order
#   bash experiments/pilot/setup_env.sh venv frontend   # only the named parts
#
# Parts:
#   venv      training and serving environment, Python 3.12   (requirements/train-serve.lock.txt)
#   frontend  Mini-SWE environment, Python 3.12               (requirements/frontend.lock.txt)
#   agent     self-contained Python 3.9 with the BigCodeBench packages and NLTK data, mounted
#             read-only in the sandbox                        (requirements/agent.lock.txt)
#   bwrap     a bubblewrap binary if /usr/bin has none, then sandbox_probe.py --install
#   record    package lists and versions of this node, under $PILOT_OUT/env/<host>/
#
# Needs network: PyPI, GitHub (Python builds, NLTK data) and, only if bwrap is missing, the
# distribution's package archive. Nothing outside PILOT_ROOT and PILOT_OUT is written.
#
# The three lock files name every package and version, so every node installs the same set.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PILOT_ONLINE=1
# shellcheck source=experiments/pilot/env.sh
source "$HERE/env.sh"
REQ=$HERE/requirements
export UV_CACHE_DIR=$PILOT_ROOT/cache/uv UV_PYTHON_INSTALL_DIR=$PILOT_ROOT/pythons
# Managed Python builds ship their headers; the Triton kernels of FLA and Liger compile against them.
export UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-managed

say() { echo "[pilot setup $(date -u +%T)] $*"; }
digest() { sha256sum "$1" | cut -d' ' -f1; }

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return
    fi
    if [[ ! -x $PILOT_TOOLS/uv/bin/uv ]]; then
        say "installing uv into $PILOT_TOOLS/uv" >&2
        python3 -m pip install -q --target "$PILOT_TOOLS/uv" uv >&2 ||
            { curl -LsSf https://astral.sh/uv/install.sh |
                env UV_INSTALL_DIR="$PILOT_TOOLS/uv/bin" UV_NO_MODIFY_PATH=1 sh >&2; }
    fi
    echo "$PILOT_TOOLS/uv/bin/uv"
}

# build_python_env <directory> <lock file> <what>: a fresh Python 3.12 environment with exactly the locked packages.
build_python_env() {
    local directory=$1 lock=$2 what=$3 marker=$1/.pilot_ready want
    want=$(digest "$lock")
    if [[ -f $marker && $(cat "$marker") == "$want" ]]; then
        say "$what is ready"
        return
    fi
    say "building $what from $(basename "$lock")"
    rm -rf "$directory"
    "$UV" venv --python "${PILOT_PYTHON:-3.12}" "$directory"
    "$UV" pip install --python "$directory/bin/python" -r "$lock"
    echo "$want" >"$marker"
}

build_venv() {
    build_python_env "$PILOT_VENV" "$REQ/train-serve.lock.txt" "the training/serving environment"
}

build_frontend() {
    build_python_env "$PILOT_FRONTEND" "$REQ/frontend.lock.txt" "the Mini-SWE environment"
    "$PILOT_FRONTEND/bin/python" -c 'import minisweagent, litellm'
}

build_agent() {
    local lock=$REQ/agent.lock.txt marker=$PILOT_AGENT_ENV/.bcb_deps_ready want source prefix
    want=$(digest "$lock")
    if [[ -f $marker && $(cat "$marker") == "$want" ]]; then
        say "the agent environment is ready"
        return
    fi
    say "building the agent environment: Python 3.9 and the BigCodeBench packages"
    "$UV" python install 3.9
    source=$("$UV" python find 3.9)
    prefix=$(cd -- "$(dirname -- "$(readlink -f "$source")")/.." && pwd -P)
    # The sandbox mounts only this directory, so the interpreter itself must live in it:
    # a copy of the relocatable Python build, with the packages installed into it.
    rm -rf "$PILOT_AGENT_ENV"
    cp -a "$prefix" "$PILOT_AGENT_ENV"
    [[ -x $PILOT_AGENT_ENV/bin/python ]] || ln -s python3.9 "$PILOT_AGENT_ENV/bin/python"
    echo "uv" >"$PILOT_AGENT_ENV/.pilot_installer"
    if ! "$UV" pip install --python "$PILOT_AGENT_ENV/bin/python3.9" --break-system-packages -r "$lock"; then
        say "uv could not install the benchmark packages; trying pip with the same lock file"
        "$PILOT_AGENT_ENV/bin/python3.9" -m pip install --break-system-packages --no-deps -r "$lock"
        echo "pip" >"$PILOT_AGENT_ENV/.pilot_installer"
    fi
    AGENT_ENV=$PILOT_AGENT_ENV "$PILOT_AGENT_ENV/bin/python" - <<'PY'
import nltk, os
root = os.path.join(os.environ['AGENT_ENV'], 'share', 'nltk_data')
for name in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'words', 'vader_lexicon']:
    assert nltk.download(name, download_dir=root, quiet=True), name
PY
    echo "$want" >"$marker"
}

build_bwrap() {
    local target=$PILOT_TOOLS/bwrap work
    if [[ ! -x /usr/bin/bwrap && ! -x /bin/bwrap && ! -x $target/usr/bin/bwrap && -z ${PILOT_BWRAP_REAL:-} ]]; then
        say "no bwrap in /usr/bin; unpacking the distribution's bubblewrap package into $target"
        work=$(mktemp -d)
        (
            cd "$work"
            apt-get -o APT::Sandbox::User=root update -qq >/dev/null 2>&1 || true
            apt-get -o APT::Sandbox::User=root download bubblewrap
        ) || {
            say "could not download bubblewrap; install it, or set PILOT_BWRAP_REAL to a bwrap binary"
            exit 3
        }
        mkdir -p "$target"
        dpkg-deb --fsys-tarfile "$work"/bubblewrap_*.deb | tar x --no-same-owner -C "$target"
        rm -rf "$work"
    fi
    "$PILOT_VENV/bin/python" "$HERE/sandbox_probe.py" --install
}

record() {
    local out
    out=$PILOT_OUT/env/$(hostname)
    mkdir -p "$out"
    "$UV" pip freeze --python "$PILOT_VENV/bin/python" >"$out/train-serve.freeze.txt"
    "$UV" pip freeze --python "$PILOT_FRONTEND/bin/python" >"$out/frontend.freeze.txt"
    "$UV" pip freeze --python "$PILOT_AGENT_ENV/bin/python3.9" >"$out/agent.freeze.txt"
    cp "$PILOT_ROOT/sandbox_mode.json" "$out/sandbox_mode.json"
    {
        echo "host=$(hostname)"
        echo "recorded=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "repo_commit=$(git -C "$PILOT_REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "uv=$("$UV" --version)"
        echo "train_python=$("$PILOT_VENV/bin/python" -V 2>&1)"
        echo "agent_python=$("$PILOT_AGENT_ENV/bin/python" -V 2>&1)"
        echo "agent_installer=$(cat "$PILOT_AGENT_ENV/.pilot_installer" 2>/dev/null || echo unknown)"
        echo "train_lock_sha256=$(digest "$REQ/train-serve.lock.txt")"
        echo "frontend_lock_sha256=$(digest "$REQ/frontend.lock.txt")"
        echo "agent_lock_sha256=$(digest "$REQ/agent.lock.txt")"
        echo "train_freeze_sha256=$(digest "$out/train-serve.freeze.txt")"
        echo "frontend_freeze_sha256=$(digest "$out/frontend.freeze.txt")"
        echo "agent_freeze_sha256=$(digest "$out/agent.freeze.txt")"
        echo "sandbox_mode=$(sed -n 's/.*"mode": "\([a-z-]*\)".*/\1/p' "$PILOT_ROOT/sandbox_mode.json" | head -1)"
        echo "cc=$(cc --version 2>/dev/null | head -1 || echo missing)"
        echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')"
    } >"$out/versions.txt"
    cat "$out/versions.txt"
    say "recorded in $out"
}

UV=$(find_uv)
say "uv: $("$UV" --version); root: $PILOT_ROOT; free: $(df -h "$PILOT_ROOT" | tail -1 | awk '{print $4}')"
parts=("$@")
((${#parts[@]})) || parts=(venv frontend agent bwrap record)
for part in "${parts[@]}"; do
    case "$part" in
        venv) build_venv ;;
        frontend) build_frontend ;;
        agent) build_agent ;;
        bwrap) build_bwrap ;;
        record) record ;;
        *)
            echo "Unknown part: $part (venv, frontend, agent, bwrap, record)" >&2
            exit 2
            ;;
    esac
done
say "PILOT_SETUP_OK ${parts[*]}"
