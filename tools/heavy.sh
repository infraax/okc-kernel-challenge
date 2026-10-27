#!/bin/sh
# Run a heavy command (torch tests, local_check on big shapes) in its own systemd user scope, outside the
# claude-remote service cgroup (4 GB cap shared by the whole Claude session): an OOM then kills only this command.
# Limits per AGENTS.md: 4 cores, 4 GB. Usage: tools/heavy.sh <command> [args...]
exec env XDG_RUNTIME_DIR="/run/user/$(id -u)" systemd-run --user --scope -q -p MemoryMax=4G -p CPUQuota=400% -- nice -n 10 "$@"
