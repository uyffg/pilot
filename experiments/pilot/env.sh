#!/usr/bin/env bash
# Sourced by every pilot entry point. It sets the directory layout and the process environment
# that the original runs used (the exports of their Slurm job script), so that the unchanged
# runner, trainer and server behave as they did there.
#
#   source experiments/pilot/env.sh
#
# Layout (all overridable before sourcing):
#   PILOT_ROOT       node-local working area: environments, model, scratch   (default /arl/pilot)
#   PILOT_OUT        results. Node-local by default; it must be on storage that every node of the
#                    experiment sees (and that supports flock) when the slots span several nodes.
#   PILOT_MODEL      Qwen3.5-35B-A3B at the pinned revision, with bcb_verified.json
#   PILOT_VENV       training and serving environment (torch, transformers, vLLM)
#   PILOT_FRONTEND   Mini-SWE environment (mini-swe-agent, litellm); never mixed into PILOT_VENV
#   PILOT_AGENT_ENV  self-contained Python 3.9 with the BigCodeBench packages, mounted in the sandbox
#   PILOT_SCRATCH    merged models and task workspaces (large, disposable)
# shellcheck shell=bash

if [[ -z ${BASH_SOURCE[0]:-} ]]; then
    echo "env.sh: source this file from bash" >&2
    return 2 2>/dev/null || exit 2
fi
PILOT_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PILOT_REPO

if [[ -z ${PILOT_ROOT:-} ]]; then
    if [[ -d /arl ]]; then
        PILOT_ROOT=/arl/pilot
    else
        echo "env.sh: set PILOT_ROOT to a node-local directory with about 500 GB free" >&2
        return 2 2>/dev/null || exit 2
    fi
fi
export PILOT_ROOT
export PILOT_OUT=${PILOT_OUT:-$PILOT_ROOT/out}
export PILOT_MODEL=${PILOT_MODEL:-$PILOT_ROOT/models/Qwen3.5-35B-A3B}
export PILOT_VENV=${PILOT_VENV:-$PILOT_ROOT/venv}
export PILOT_FRONTEND=${PILOT_FRONTEND:-$PILOT_ROOT/frontend}
export PILOT_AGENT_ENV=${PILOT_AGENT_ENV:-$PILOT_ROOT/agent_env}
export PILOT_TOOLS=${PILOT_TOOLS:-$PILOT_ROOT/tools}
export PILOT_SCRATCH=${PILOT_SCRATCH:-$PILOT_ROOT/scratch}
mkdir -p "$PILOT_ROOT" "$PILOT_OUT" "$PILOT_TOOLS" "$PILOT_SCRATCH" "$PILOT_ROOT/tmp" "$PILOT_ROOT/hf" \
    "$PILOT_ROOT/vllm_cache"

# ---- process environment of the original jobs ----
export MODEL_PATH=$PILOT_MODEL
export OPD_FRONTEND_PYTHON=$PILOT_FRONTEND/bin/python
# The repository is imported from this checkout, not installed: installing it would pull
# mini-swe-agent and its dependencies into the training environment.
export PYTHONPATH=$PILOT_REPO
export PATH=$PILOT_VENV/bin:$PATH
export HF_HOME=$PILOT_ROOT/hf TMPDIR=$PILOT_ROOT/tmp
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
if [[ ${PILOT_ONLINE:-0} != 1 ]]; then
    # Runs load the model from PILOT_MODEL only. setup_env.sh and fetch_model.py set PILOT_ONLINE=1.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    # The worker and the reviewer talk to a server on 127.0.0.1; a proxy would break that.
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export ACCELERATE_USE_FSDP=true FSDP_VERSION=2 FSDP_CPU_RAM_EFFICIENT_LOADING=true
export FSDP_SYNC_MODULE_STATES=true FSDP_AUTO_WRAP_POLICY=TRANSFORMER_BASED_WRAP
export FSDP_OFFLOAD_PARAMS=true FSDP_RESHARD_AFTER_FORWARD=true TRAIN_CPU_OFFLOAD=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_CACHE_ROOT=$PILOT_ROOT/vllm_cache
export LITELLM_LOCAL_MODEL_COST_MAP=True MSWEA_COST_TRACKING=ignore_errors

# Many sandboxes and server workers per node: use the hard file-descriptor limit.
ulimit -Sn "$(ulimit -Hn)" 2>/dev/null || true

# ---- CUDA user-space driver ----
# libcuda must not be older than the kernel module it talks to. A CUDA forward-compatibility
# directory (an image's /usr/local/cuda/compat) holds a newer libcuda for hosts whose driver is
# older than the toolkit; on a host whose driver is newer than that copy, CUDA cannot start
# ("Unexpected error from cudaGetDeviceCount()"). So every library-path entry whose libcuda is
# older than the kernel module is dropped, and the libraries matching the host driver are used.
# Nothing changes where no libcuda is on the path or the kernel version cannot be read.
# PILOT_KERNEL_DRIVER_VERSION overrides the detected version (tests).
pilot_version_lt() {
    [[ $1 != "$2" && $(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1) == "$1" ]]
}
pilot_cuda_library_path() {
    local kernel=${PILOT_KERNEL_DRIVER_VERSION:-} dir lib version entries=() kept=() dropped=()
    if [[ -z $kernel && -r /proc/driver/nvidia/version ]]; then
        kernel=$(awk '/NVRM/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+(\.[0-9]+)?$/){print $i; exit}}' /proc/driver/nvidia/version)
    fi
    if [[ -z $kernel && -r /sys/module/nvidia/version ]]; then
        kernel=$(cat /sys/module/nvidia/version)
    fi
    [[ -n $kernel && -n ${LD_LIBRARY_PATH:-} ]] || return 0
    IFS=: read -r -a entries <<<"$LD_LIBRARY_PATH"
    for dir in "${entries[@]}"; do
        [[ -n $dir ]] || continue
        lib=$(find -H "$dir" -maxdepth 1 -name 'libcuda.so.[0-9]*.*' 2>/dev/null | sort -V | tail -n 1)
        version=${lib##*/libcuda.so.}
        if [[ -n $lib ]] && pilot_version_lt "$version" "$kernel"; then
            dropped+=("$dir (libcuda $version)")
            continue
        fi
        kept+=("$dir")
    done
    LD_LIBRARY_PATH=$(IFS=:; echo "${kept[*]}")
    export LD_LIBRARY_PATH
    export PILOT_CUDA_KERNEL_DRIVER=$kernel
    export PILOT_CUDA_DROPPED_LIBRARY_DIRS=$(IFS=';'; echo "${dropped[*]}")
}
pilot_cuda_library_path
