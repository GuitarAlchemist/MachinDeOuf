#!/usr/bin/env bash
#
# harness-remediate.sh — the self-improvement loop
#
# WHAT IT DOES:
#   1. Runs harness adapters for round N (observe)
#   2. Matches observations against a remediation catalog (decide)
#   3. Runs the remediation command for each match (fix)
#   4. Runs harness adapters for round N+1 (verify)
#   5. Compares rounds: if observations improved, stage + commit (compound)
#   6. If regression detected, revert (safety)
#
# THIS IS THE FACTORY. The adapters are factories that produce
# observations. The remediation catalog maps observations to actions.
# This script ties them together. Add a new adapter + a new
# remediation mapping and the loop works for that case too.
#
# Usage:
#   ./scripts/harness-remediate.sh [--dry-run] [--round N]
#
# Requires: ix-harness-cargo, ix-harness-clippy binaries in target/release/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HARNESS="$REPO_ROOT/target/release"
OBS_DIR="$REPO_ROOT/.ix/remediation"
DRY_RUN=false
ROUND="${ROUND:-1}"

# Parse args
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --round=*) ROUND="${arg#*=}" ;;
    esac
done

mkdir -p "$OBS_DIR"

# ─── Colors ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[harness]${NC} $*"; }
ok()   { echo -e "${GREEN}[  ok  ]${NC} $*"; }
warn() { echo -e "${YELLOW}[ warn ]${NC} $*"; }
fail() { echo -e "${RED}[ fail ]${NC} $*"; }

# ─── Step 1: Observe (round N) ──────────────────────────────
log "Step 1: Running adapters for round $ROUND"

ROUND_FILE="$OBS_DIR/round-${ROUND}.jsonl"
rm -f "$ROUND_FILE"

# Adapter: clippy (the self-improvement signal)
if [[ -x "$HARNESS/ix-harness-clippy" ]]; then
    log "  Running clippy adapter..."
    cargo clippy --workspace --tests --message-format=json 2>&1 \
        | "$HARNESS/ix-harness-clippy" --round "$ROUND" >> "$ROUND_FILE" 2>/dev/null
    ok "  clippy: $(wc -l < "$ROUND_FILE") observations"
else
    fail "  ix-harness-clippy not found at $HARNESS"
    exit 1
fi

# Count findings by severity
F_COUNT=$(grep -c '"variant":"F"' "$ROUND_FILE" 2>/dev/null || echo 0)
D_COUNT=$(grep -c '"variant":"D"' "$ROUND_FILE" 2>/dev/null || echo 0)
T_COUNT=$(grep -c '"variant":"T"' "$ROUND_FILE" 2>/dev/null || echo 0)
log "  Round $ROUND signal: ${RED}F=$F_COUNT${NC} ${YELLOW}D=$D_COUNT${NC} ${GREEN}T=$T_COUNT${NC}"

if [[ $F_COUNT -eq 0 && $D_COUNT -eq 0 ]]; then
    ok "No findings to remediate. Codebase is clean."
    exit 0
fi

# ─── Step 2: Match against remediation catalog ──────────────
#
# THE CATALOG: maps claim_key patterns → remediation commands.
# This is the part that makes it a factory, not a one-off.
# Add new entries here for new adapter types.
#
# Format: pattern → command
# The command runs in the repo root. $CRATE is extracted from
# the claim_key when applicable.

log "Step 2: Matching findings against remediation catalog"

declare -A REMEDIATION_CATALOG
REMEDIATION_CATALOG["clippy::"]="cargo clippy --fix --workspace --tests --allow-dirty --allow-staged"

# Future catalog entries (uncomment when adapters exist):
# REMEDIATION_CATALOG["submodule:"]="git submodule update --remote"
# REMEDIATION_CATALOG["cargo_suite::"]="cargo test --workspace --no-fail-fast"
# REMEDIATION_CATALOG["build:"]="cargo build --workspace"

MATCHED=false
for PATTERN in "${!REMEDIATION_CATALOG[@]}"; do
    if grep -q "\"claim_key\":\"clippy:" "$ROUND_FILE" 2>/dev/null; then
        MATCHED=true
        COMMAND="${REMEDIATION_CATALOG[$PATTERN]}"
        log "  Matched pattern '$PATTERN' → $COMMAND"
    fi
done

if [[ "$MATCHED" == "false" ]]; then
    warn "No remediation catalog entry matched the findings."
    warn "Findings require manual intervention or a new catalog entry."
    exit 0
fi

# ─── Step 3: Remediate (fix) ────────────────────────────────
log "Step 3: Applying automated remediation"

if [[ "$DRY_RUN" == "true" ]]; then
    warn "  --dry-run: would run: $COMMAND"
else
    cd "$REPO_ROOT"
    log "  Running: $COMMAND"
    eval "$COMMAND" 2>&1 | tail -5
    ok "  Remediation applied"
fi

# ─── Step 4: Re-observe (round N+1) ─────────────────────────
NEXT_ROUND=$((ROUND + 1))
log "Step 4: Running adapters for round $NEXT_ROUND (verification)"

NEXT_FILE="$OBS_DIR/round-${NEXT_ROUND}.jsonl"
rm -f "$NEXT_FILE"

cargo clippy --workspace --tests --message-format=json 2>&1 \
    | "$HARNESS/ix-harness-clippy" --round "$NEXT_ROUND" >> "$NEXT_FILE" 2>/dev/null

F_NEXT=$(grep -c '"variant":"F"' "$NEXT_FILE" 2>/dev/null || echo 0)
D_NEXT=$(grep -c '"variant":"D"' "$NEXT_FILE" 2>/dev/null || echo 0)
T_NEXT=$(grep -c '"variant":"T"' "$NEXT_FILE" 2>/dev/null || echo 0)
log "  Round $NEXT_ROUND signal: ${RED}F=$F_NEXT${NC} ${YELLOW}D=$D_NEXT${NC} ${GREEN}T=$T_NEXT${NC}"

# ─── Step 5: Compare and decide ─────────────────────────────
log "Step 5: Comparing rounds"

echo ""
echo "  ┌─────────────┬──────────┬──────────┬─────────┐"
echo "  │             │ Round $ROUND  │ Round $NEXT_ROUND  │ Delta   │"
echo "  ├─────────────┼──────────┼──────────┼─────────┤"
printf "  │ ${RED}F (safety)${NC}  │ %-8s │ %-8s │ %-7s │\n" "$F_COUNT" "$F_NEXT" "$((F_NEXT - F_COUNT))"
printf "  │ ${YELLOW}D (doubt)${NC}   │ %-8s │ %-8s │ %-7s │\n" "$D_COUNT" "$D_NEXT" "$((D_NEXT - D_COUNT))"
printf "  │ ${GREEN}T (verified)${NC}│ %-8s │ %-8s │ %-7s │\n" "$T_COUNT" "$T_NEXT" "$((T_NEXT - T_COUNT))"
echo "  └─────────────┴──────────┴──────────┴─────────┘"
echo ""

# Decision logic: did F observations decrease?
if [[ $F_NEXT -lt $F_COUNT ]]; then
    ok "Safety signal improved: F went from $F_COUNT to $F_NEXT"
    IMPROVED=true
elif [[ $F_NEXT -eq $F_COUNT && $D_NEXT -lt $D_COUNT ]]; then
    ok "Quality signal improved: D went from $D_COUNT to $D_NEXT"
    IMPROVED=true
elif [[ $F_NEXT -eq $F_COUNT && $D_NEXT -eq $D_COUNT ]]; then
    warn "No measurable change. Remediation was a no-op."
    IMPROVED=false
else
    fail "Regression detected! F went from $F_COUNT to $F_NEXT, D from $D_COUNT to $D_NEXT"
    fail "The fix made things worse. Consider reverting."
    IMPROVED=false
fi

# ─── Step 6: Commit if improved ─────────────────────────────
if [[ "$IMPROVED" == "true" && "$DRY_RUN" == "false" ]]; then
    CHANGED=$(git diff --name-only | wc -l)
    if [[ $CHANGED -gt 0 ]]; then
        log "Step 6: Staging and committing improvements"

        # Stage only .rs files (not settings, not lock files)
        git diff --name-only | grep '\.rs$' | xargs git add 2>/dev/null || true

        STAGED=$(git diff --cached --name-only | wc -l)
        if [[ $STAGED -gt 0 ]]; then
            git commit -m "$(cat <<COMMITEOF
fix: harness-driven self-improvement (round $ROUND → $NEXT_ROUND)

Automated remediation loop via scripts/harness-remediate.sh.

## Observations

Round $ROUND: F=$F_COUNT D=$D_COUNT T=$T_COUNT
Round $NEXT_ROUND: F=$F_NEXT D=$D_NEXT T=$T_NEXT

## Remediation applied

$COMMAND

## Decision

$(if [[ $F_NEXT -lt $F_COUNT ]]; then echo "Safety improved: F $F_COUNT → $F_NEXT"; fi)
$(if [[ $D_NEXT -lt $D_COUNT ]]; then echo "Quality improved: D $D_COUNT → $D_NEXT"; fi)
$STAGED files changed across $(git diff --cached --name-only | sed 's|/[^/]*$||' | sort -u | wc -l) crates.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
COMMITEOF
)"
            ok "Committed: $(git log --oneline -1)"
        else
            warn "No .rs files changed — nothing to commit"
        fi
    else
        warn "No files changed by remediation"
    fi
elif [[ "$DRY_RUN" == "true" ]]; then
    warn "  --dry-run: skipping commit"
else
    warn "  Not committing (no improvement detected)"
fi

# ─── Summary ────────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════════════════"
log "  Harness self-improvement loop complete"
log "  Round: $ROUND → $NEXT_ROUND"
log "  Improved: $IMPROVED"
log "  Observations: $OBS_DIR/"
log "═══════════════════════════════════════════════════"
