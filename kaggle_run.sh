#!/usr/bin/env bash
# =============================================================================
# Kaggle P100 single-session driver (12h cap):
#   - auto-pushes progress to GitHub every 20 min AND after every stop
#     (Kaggle wipes the VM afterwards -> GitHub is the persistent store)
#   - hard-stops the sweep at SESSION_HOURS (default 10) so the final push
#     always completes inside the 12h window
#   - fully resumable: next session, same command continues where this stopped
#
# Requires env: GITHUB_TOKEN, GITHUB_REPO (set by the notebook cell).
# P100 realistic coverage per session (EPOCHS_FAST=30):
#   session 1: tests + V1-V7 + ALL synthetic + cifar10 A1/A2/A5 (+S1/S6 if lucky)
#   session 2: rest of cifar10 + modelnet40 A1/A2/A5
#   session 3: remaining sweeps
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
SESSION_HOURS=${SESSION_HOURS:-10}
export EPOCHS_FAST=${EPOCHS_FAST:-30}
export PARALLEL=${PARALLEL:-1}          # 4 weak vCPUs: parallelism hurts on Kaggle
export WORKERS=${WORKERS:-3}

# background pusher: progress survives even a hard session kill
( while true; do sleep 1200; bash save_progress.sh > /dev/null 2>&1; done ) &
PUSHER=$!
trap 'kill $PUSHER 2>/dev/null; bash save_progress.sh' EXIT

echo "== Kaggle session: budget ${SESSION_HOURS}h, then push and stop =="
timeout "${SESSION_HOURS}h" bash run_deadline.sh
echo "== session budget reached or sweep finished; final push via trap =="
