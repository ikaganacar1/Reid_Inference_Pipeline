#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROLE="${1:-}"
ENABLE="${2:-}"
if [ "${ROLE}" != "prime" ] && [ "${ROLE}" != "worker" ]; then
    echo "Usage: $0 prime|worker [--enable]"
    exit 2
fi
if [ -n "${ENABLE}" ] && [ "${ENABLE}" != "--enable" ]; then
    echo "ERROR: Unknown option '${ENABLE}'."
    exit 2
fi

REPO_DIR="$(pwd -P)"
JETSON_VENV="${JETSON_VENV:-${REPO_DIR}/.venv-jetson}"
PYTHON_BIN="${PYTHON_BIN:-${JETSON_VENV}/bin/python}"
UNIT_DIR="${UNIT_DIR:-${HOME}/.config/systemd/user}"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Missing runtime Python: ${PYTHON_BIN}"
    echo "Run scripts/install_jetson_runtime.sh ${ROLE} first."
    exit 1
fi

mkdir -p "${UNIT_DIR}" "$(dirname "${ENV_FILE}")"
if [ ! -f "${ENV_FILE}" ]; then
    cp .env.example "${ENV_FILE}"
    if [ "${ROLE}" = "worker" ]; then
        sed -i \
            -e 's/^PIPELINE_ROLE=.*/PIPELINE_ROLE=worker/' \
            -e 's/^CAMERA_IDS=.*/CAMERA_IDS=cam2/' \
            "${ENV_FILE}"
    fi
    chmod 600 "${ENV_FILE}"
    echo "Created environment file: ${ENV_FILE}"
fi

export REID_ENV_FILE="${ENV_FILE}"
source scripts/runtime_env.sh
load_runtime_env true
if [ "${PIPELINE_ROLE:-${ROLE}}" != "${ROLE}" ]; then
    echo "ERROR: ${ENV_FILE} has PIPELINE_ROLE=${PIPELINE_ROLE}, but installer role=${ROLE}."
    exit 1
fi

escape_sed() {
    printf '%s' "$1" | sed 's/[&|]/\\&/g'
}

render_unit() {
    local template="$1"
    local destination="$2"
    sed \
        -e "s|@REPO_DIR@|$(escape_sed "${REPO_DIR}")|g" \
        -e "s|@PYTHON_BIN@|$(escape_sed "${PYTHON_BIN}")|g" \
        -e "s|@ENV_FILE@|$(escape_sed "${ENV_FILE}")|g" \
        "${template}" > "${destination}"
}

units=()
if [ "${ROLE}" = "prime" ]; then
    render_unit deploy/systemd/reid-prime.service.in "${UNIT_DIR}/reid-prime.service"
    units+=(reid-prime.service)
    if runtime_bool_true "${LOCAL_CAMERA_ENABLED:-true}"; then
        render_unit deploy/systemd/reid-camera.service.in "${UNIT_DIR}/reid-camera.service"
        units+=(reid-camera.service)
    fi
else
    render_unit deploy/systemd/reid-camera.service.in "${UNIT_DIR}/reid-camera.service"
    units+=(reid-camera.service)
fi

if [ "${SKIP_SYSTEMCTL:-0}" != "1" ]; then
    systemctl --user daemon-reload
    if [ "${ENABLE}" = "--enable" ]; then
        systemctl --user enable --now "${units[@]}"
    fi
fi

echo "Installed role=${ROLE} units: ${units[*]}"
echo "Per-device overrides: ${ENV_FILE}"
if [ "${ENABLE}" != "--enable" ]; then
    echo "Units were prepared but not enabled. Rerun with --enable after preflight passes."
fi
echo "For boot before login, run once on the Jetson: sudo loginctl enable-linger ${USER}"
