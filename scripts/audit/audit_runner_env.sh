#!/usr/bin/env bash
# ==============================================================================
# Environment Initialization Vector for Distributed Audit Tasks
# Usage: source scripts/audit/audit_runner_env.sh
# ==============================================================================

# Guard Execution Context: Enforce sourcing to preserve environment mutations
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[-] Error: This initialization script must be sourced, not executed directly." >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# 1. Path Topology & Workspace Resolution
# ------------------------------------------------------------------------------
_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${_ENV_SCRIPT_DIR}/../.." && pwd)}"

# Standardize host-specific execution roots dynamically
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/morg/students/rashkovits}"
export CONDA_HOME="${CONDA_HOME:-${WORKSPACE_ROOT}/miniconda3}"
export TARGET_CONDA_ENV="${TARGET_CONDA_ENV:-${WORKSPACE_ROOT}/envs/snmf_env}"

# ------------------------------------------------------------------------------
# 2. Conda Core Bootstrap & Environment Activation
# ------------------------------------------------------------------------------
_CONDA_EXEC="${CONDA_HOME}/bin/conda"

if [[ -x "$_CONDA_EXEC" ]]; then
    # Evaluate shell hook natively to eliminate subshell isolation failure modes
    if _CONDA_HOOK="$("$_CONDA_EXEC" shell.bash hook 2>/dev/null)"; then
        eval "$_CONDA_HOOK"
    elif [[ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]]; then
        source "${CONDA_HOME}/etc/profile.d/conda.sh"
    else
        export PATH="${CONDA_HOME}/bin:${PATH}"
    fi
    unset _CONDA_HOOK

    # Attempt deterministic environment activation with downstream fallback
    if ! conda activate "$TARGET_CONDA_ENV" 2>/dev/null; then
        if ! conda activate snmf_env 2>/dev/null; then
            echo "[-] Error: Could not activate required Conda environment (tried: ${TARGET_CONDA_ENV}, snmf_env)." >&2
            return 1
        fi
    fi
else
    echo "[-] Warning: Conda binary unresolved at $_CONDA_EXEC" >&2
fi

# ------------------------------------------------------------------------------
# 3. Cache Topologies & Runtime Variables
# ------------------------------------------------------------------------------
export CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_ROOT}/hf_cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "logs"

# ------------------------------------------------------------------------------
# 4. Python Environment Invariants
# ------------------------------------------------------------------------------
cd "$REPO_ROOT" || return 1

if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$(pwd)"
else
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
fi

echo "[+] Execution context fully initialized."
echo "    -> Workspace : $REPO_ROOT"
echo "    -> Active Env: ${CONDA_DEFAULT_ENV:-UNRESOLVED}"