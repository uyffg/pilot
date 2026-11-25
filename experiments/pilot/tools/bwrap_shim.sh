#!/usr/bin/env bash
# Installed by sandbox_probe.py as <agent env>/bin/bwrap, only on hosts where the sandbox command
# of worker_reviewer_bench/execution.py cannot run as written.
#
# Why it sits in the agent environment: execution.py starts `bwrap` with PATH set to
# "<agent env>/bin:/usr/bin:/bin", so that is the only place a wrapper can be found without
# changing /usr/bin on the host.
#
# What it changes: nothing in mode "native" (it only locates a bwrap binary that is not in
# /usr/bin). In the modes "bind-proc", "dev-bind-proc" and "ro-bind-proc" it replaces the
# option pair `--proc /proc` with `--bind /proc /proc`, `--dev-bind /proc /proc` or
# `--ro-bind /proc /proc`. A container with a masked /proc refuses to mount a new procfs, so
# the sandbox gets the host's /proc bound instead. Every other option, and the command, are
# passed through unchanged.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
conf=$here/../pilot_bwrap.conf
real=$(sed -n 's/^real=//p' "$conf")
mode=$(sed -n 's/^mode=//p' "$conf")
if [[ -z $real || ! -x $real ]]; then
    echo "bwrap shim: no usable bwrap binary recorded in $conf" >&2
    exit 127
fi

case "$mode" in
    bind-proc) bind=--bind ;;
    dev-bind-proc) bind=--dev-bind ;;
    ro-bind-proc) bind=--ro-bind ;;
    *) exec "$real" "$@" ;;
esac

args=()
rewritten=0
while (($#)); do
    # The first `--proc /proc` is always the sandbox option: it precedes the command.
    if ((!rewritten)) && [[ $1 == --proc && ${2:-} == /proc ]]; then
        args+=("$bind" /proc /proc)
        rewritten=1
        shift 2
        continue
    fi
    args+=("$1")
    shift
done
exec "$real" "${args[@]}"
