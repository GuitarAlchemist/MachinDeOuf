#!/usr/bin/env bash
# ── Scheduled monitoring loop ─────────────────────────────────────────────
#
# Runs the sensor-anomaly pipeline every INTERVAL seconds for MAX_RUNS
# iterations. After each run, updates the `sensor_health` belief with
# a hexavalent verdict based on the governance gate's exit code, and
# captures a snapshot when confidence drops or the verdict flips.
#
# Real-world use: a watchdog that runs in the background, tracking
# sensor health as an evolving belief state for audit replay.
#
# Usage:
#   bash examples/showcase/advanced/monitoring-loop.sh
#   INTERVAL=3 MAX_RUNS=4 bash examples/showcase/advanced/monitoring-loop.sh

set -euo pipefail

IX_BIN="./target/release/ix.exe"
[[ -f "$IX_BIN" ]] || IX_BIN="./target/release/ix"
[[ -f "$IX_BIN" ]] || IX_BIN="./target/debug/ix.exe"
[[ -f "$IX_BIN" ]] || IX_BIN="./target/debug/ix"
[[ -f "$IX_BIN" ]] || { echo "error: ix binary not found" >&2; exit 1; }
IX_BIN="$(cd "$(dirname "$IX_BIN")" && pwd)/$(basename "$IX_BIN")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export IX_ROOT="$WORKSPACE_ROOT"
export IX_GOVERNANCE_DIR="$WORKSPACE_ROOT/governance/demerzel"

INTERVAL="${INTERVAL:-2}"
MAX_RUNS="${MAX_RUNS:-5}"

# Dedicated temp workspace for beliefs + pipeline state.
RUN_DIR="$(mktemp -d -t ix-monitor-XXXXXX)"
trap 'rm -rf "$RUN_DIR"' EXIT
cp "$SCRIPT_DIR/sensor-anomaly.yaml" "$RUN_DIR/ix.yaml"
cd "$RUN_DIR"

if [[ -t 1 ]]; then
    HEAD="\033[1;36m" OK="\033[1;32m" WARN="\033[1;33m" ERR="\033[1;31m" DIM="\033[2m" RESET="\033[0m"
else
    HEAD="" OK="" WARN="" ERR="" DIM="" RESET=""
fi

printf "${HEAD}━━━ Sensor monitoring loop ━━━${RESET}\n"
printf "${DIM}interval=${INTERVAL}s  max_runs=${MAX_RUNS}  workdir=${RUN_DIR}${RESET}\n\n"

previous_verdict=""
for ((i = 1; i <= MAX_RUNS; i++)); do
    printf "${HEAD}── Run %d/%d ──${RESET}\n" "$i" "$MAX_RUNS"

    # Execute the sensor-anomaly pipeline.
    if run_json=$("$IX_BIN" --format json pipeline run 2>&1); then
        # Extract the governance verdict from the compliance_audit stage.
        verdict=$(echo "$run_json" \
            | grep -oE '"compliant":[[:space:]]*(true|false)' \
            | head -1 \
            | sed 's/.*:[[:space:]]*//')
        if [[ "$verdict" == "true" ]]; then
            hex="T"
            conf="0.85"
            color="$OK"
        else
            hex="D"
            conf="0.55"
            color="$WARN"
        fi
    else
        hex="F"
        conf="0.95"
        color="$ERR"
    fi

    printf "  pipeline: ${color}verdict=%s${RESET} confidence=%s\n" "$hex" "$conf"

    # Update the sensor_health belief.
    "$IX_BIN" beliefs set sensor_health \
        "sensor monitoring run #$i — constitutional audit" \
        --truth "$hex" --confidence "$conf" \
        --format json >/dev/null

    # If the verdict flipped from the previous run, snapshot.
    if [[ -n "$previous_verdict" && "$previous_verdict" != "$hex" ]]; then
        snap_desc="verdict-flip-$previous_verdict-to-$hex"
        printf "  ${WARN}▶ verdict flipped${RESET} %s→%s — capturing snapshot\n" "$previous_verdict" "$hex"
        "$IX_BIN" beliefs snapshot "$snap_desc" --format json >/dev/null
    fi
    previous_verdict="$hex"

    if ((i < MAX_RUNS)); then
        sleep "$INTERVAL"
    fi
    echo
done

printf "${HEAD}━━━ Loop complete ━━━${RESET}\n\n"
printf "Final belief state:\n"
"$IX_BIN" --format table beliefs get sensor_health

printf "\nSnapshots captured:\n"
ls -1 state/snapshots/ 2>/dev/null || echo "  (none)"

cat <<EOF

${DIM}The loop tracked sensor health over ${MAX_RUNS} runs, wrote one
belief per run (overwriting state/beliefs/sensor_health.belief.json),
and captured a snapshot every time the hexavalent verdict flipped.
Real monitoring setups would persist this to durable storage and
wire the snapshots into an alerting channel.${RESET}

EOF
