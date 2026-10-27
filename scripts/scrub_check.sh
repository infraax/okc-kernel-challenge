#!/bin/sh
# Publication gate: prints CLEAN only if no secret patterns, private paths, internal hosts, emails or internal ids remain.
cd "$(dirname "$0")/.." || exit 2
hits=$(grep -rIlE 'ghp_[A-Za-z0-9]{20}|gho_[A-Za-z0-9]{20}|github_pat_|sk-[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16}|Bearer [A-Za-z0-9._-]{20}|BEGIN [A-Z ]*PRIVATE KEY|xox[bp]-|AIza[0-9A-Za-z_-]{30}|/srv/fbt|tailscale|tuf-gaming|@gmail\.com|(ses|msg|tsk|agt)_01[0-9A-Z]{20}|trig_01[0-9A-Za-z]{10}|claude-(opus|sonnet|haiku|fable)-[0-9]' --exclude=scrub_check.sh . )
if [ -n "$hits" ]; then echo "HITS:"; echo "$hits"; exit 1; fi
echo CLEAN
