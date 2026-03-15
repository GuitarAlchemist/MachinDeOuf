#!/bin/bash
# Governance check hook — warns on potentially irreversible or disproportionate actions
# References Demerzel Constitution Articles 3 (Reversibility) and 4 (Proportionality)

TOOL_NAME="${CLAUDE_TOOL_NAME:-}"
COMMAND="${CLAUDE_BASH_COMMAND:-}"

# Article 3: Reversibility — warn on destructive commands
if [[ "$TOOL_NAME" == "Bash" ]]; then
    if [[ "$COMMAND" =~ (rm\ -rf|git\ reset\ --hard|git\ push\ --force|drop\ database|DROP\ TABLE|git\ clean\ -fd) ]]; then
        echo "[demerzel] WARN: Article 3 (Reversibility) — destructive command detected. Confirm before proceeding." >&2
    fi

    # Article 9: Bounded Autonomy — warn on permission escalation
    if [[ "$COMMAND" =~ (chmod\ 777|sudo|chown\ root|--no-verify) ]]; then
        echo "[demerzel] WARN: Article 9 (Bounded Autonomy) — permission escalation detected." >&2
    fi
fi

# Article 4: Proportionality — warn on overly broad file operations
if [[ "$TOOL_NAME" == "Write" ]]; then
    FILE_PATH="${CLAUDE_FILE_PATH:-}"
    # Warn if overwriting config files that affect the whole system
    if [[ "$FILE_PATH" =~ (/etc/|\.env$|credentials|secret) ]]; then
        echo "[demerzel] WARN: Article 4 (Proportionality) — writing to sensitive system file." >&2
    fi
fi
