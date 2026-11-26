#!/usr/bin/env bash
# Run a command of this repository inside the Merlin training image.
#
#   bash experiments/pilot/merlin/enter.sh bash experiments/pilot/setup_env.sh
#   nodekit run <node> pilot -- bash <checkout>/experiments/pilot/merlin/enter.sh bash experiments/pilot/launch.sh
#
# torch 2.11 is built for CUDA 13. On a node with driver 535 it needs the forward-compatibility
# libraries of the image; on a node with driver 580 or newer it must not use them, because they
# are older than the driver. This calls the `uniagent.sh` wrapper of the agentic_rl checkout
# (user namespace, pivot_root into the image rootfs at $ARL/rootfs, the compat libraries first on
# the library path, then the host's) with this checkout as the working directory, and env.sh then
# drops every library directory whose libcuda is older than the node's kernel module. Inside,
# /arl is the node's $ARL (default /tmp/arl), and the scripts keep everything under /arl/pilot.
#
# This checkout must be on the devbox share (/mlx_devbox/...): that path is bound into the image,
# a path on the node's own disk is not.
#
# The node needs the image rootfs and the host driver libraries, as `ray_node.sh` with
# PREP_ONLY=1 prepares them. It does not need the 30B model or the R2E data.
#
# PILOT_UNIAGENT   path of uniagent.sh (default: the shared agentic_rl checkout)
# Variables named PILOT_*, ARL_*, WANDB_ENTITY/PROJECT/MODE and DRY_RUN are passed into the image.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
UNIAGENT=${PILOT_UNIAGENT:-/mlx_devbox/users/yufa.zhou/playground/yuli/agentic_rl/slurm_scripts/merlin/uniagent.sh}
if [[ ! -f $UNIAGENT ]]; then
    echo "enter.sh: $UNIAGENT not found; set PILOT_UNIAGENT" >&2
    exit 2
fi
if (($# == 0)); then
    echo "usage: enter.sh <command...>" >&2
    exit 2
fi
# uniagent.sh changes into $REPO and puts it on PYTHONPATH; experiments/pilot/env.sh then sets its own.
REPO=$(cd -- "$HERE/../../.." && pwd)
export REPO
export ARL_PASS_RE='^(PILOT_[A-Z0-9_]*|ARL_[A-Z0-9_]*|WANDB_(ENTITY|PROJECT|MODE)|DRY_RUN)'
exec bash "$UNIAGENT" "$@"
