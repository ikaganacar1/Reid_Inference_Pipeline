#!/usr/bin/env bash
# Shared .env and Python-runtime helpers. Source this file; do not execute it.

REID_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

load_runtime_env() {
    local required="${1:-false}"
    local env_file="${REID_ENV_FILE:-${REID_REPO_ROOT}/.env}"
    local legacy_file="${JETSON_ENV_FILE:-${HOME}/.config/reid-pipeline/jetson.env}"

    if [ ! -r "${env_file}" ] && [ -r "${legacy_file}" ]; then
        env_file="${legacy_file}"
        echo "WARNING: Loading legacy environment file ${legacy_file}." >&2
        echo "Move these settings to ${REID_REPO_ROOT}/.env." >&2
    fi

    if [ ! -r "${env_file}" ]; then
        if runtime_bool_true "${required}"; then
            echo "ERROR: Runtime environment file not found: ${env_file}" >&2
            echo "Run scripts/reidctl.sh init first." >&2
            return 1
        fi
        return 0
    fi

    # Explicit command environment variables take precedence over .env. Save
    # variables already present, source the shell-compatible file, then restore
    # those original values.
    local line key
    local -a preserved_names=()
    local -a preserved_values=()
    while IFS= read -r line || [ -n "${line}" ]; do
        if [[ "${line}" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)= ]]; then
            key="${BASH_REMATCH[2]}"
            if [[ -v "${key}" ]]; then
                preserved_names+=("${key}")
                preserved_values+=("${!key}")
            fi
        fi
    done < "${env_file}"

    local restore_allexport=false
    if [[ "$-" != *a* ]]; then
        set -a
        restore_allexport=true
    fi
    # shellcheck disable=SC1090
    source "${env_file}"
    if runtime_bool_true "${restore_allexport}"; then
        set +a
    fi

    local index
    for index in "${!preserved_names[@]}"; do
        printf -v "${preserved_names[$index]}" '%s' "${preserved_values[$index]}"
        export "${preserved_names[$index]}"
    done
    export REID_ENV_FILE="${env_file}"
}

runtime_bool_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

runtime_select_python() {
    local requested="${PYTHON_BIN:-}"
    local venv="${JETSON_VENV:-${REID_REPO_ROOT}/.venv-jetson}"

    if [ -n "${requested}" ]; then
        if [[ "${requested}" == */* ]] && [ ! -x "${requested}" ]; then
            requested="${REID_REPO_ROOT}/${requested}"
        fi
        if ! command -v "${requested}" >/dev/null 2>&1 && [ ! -x "${requested}" ]; then
            echo "ERROR: PYTHON_BIN is not executable: ${requested}" >&2
            return 1
        fi
        RUNTIME_PYTHON_BIN="${requested}"
    elif [ -x "${venv}/bin/python" ]; then
        RUNTIME_PYTHON_BIN="${venv}/bin/python"
    elif [ -n "${CONDA_ENV:-}" ]; then
        if ! command -v conda >/dev/null 2>&1; then
            echo "ERROR: CONDA_ENV is set but conda is unavailable." >&2
            return 1
        fi
        local conda_prefix
        conda_prefix="$(conda run --no-capture-output -n "${CONDA_ENV}" \
            python -c 'import sys; from pathlib import Path; print(Path(sys.executable).resolve().parents[1])')"
        RUNTIME_PYTHON_BIN="${conda_prefix}/bin/python"
    else
        RUNTIME_PYTHON_BIN="$(command -v python3)"
    fi

    export RUNTIME_PYTHON_BIN
}
