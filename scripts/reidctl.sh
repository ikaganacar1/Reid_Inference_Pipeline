#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/runtime_env.sh

usage() {
    cat <<'EOF'
Usage: scripts/reidctl.sh COMMAND [options]

Commands:
  init [prime|worker]  Create the private .env file
  smoke [--load-models]
                       Validate .env, configs, protocol, and optional models
  preflight            Run role-specific Jetson GPU/model/camera checks
  start                Smoke-test and start this device's configured role
  stop                 Stop this device's configured role
  restart              Stop and start this device's configured role
  status               Show local process and HTTP status
  logs                 Tail the role's local logs
EOF
}

init_environment() {
    local role="${1:-prime}"
    local env_file="${REID_ENV_FILE:-$(pwd -P)/.env}"
    if [ "${role}" != "prime" ] && [ "${role}" != "worker" ]; then
        echo "ERROR: init role must be prime or worker." >&2
        exit 2
    fi
    if [ -e "${env_file}" ]; then
        echo "Environment already exists: ${env_file}"
        return 0
    fi

    cp .env.example "${env_file}"
    chmod 600 "${env_file}"
    if [ "${role}" = "worker" ]; then
        sed -i \
            -e 's/^PIPELINE_ROLE=.*/PIPELINE_ROLE=worker/' \
            -e 's/^CAMERA_IDS=.*/CAMERA_IDS=cam2/' \
            "${env_file}"
    fi
    echo "Created ${env_file} for role=${role}"
    echo "Edit PRIME_URL, CAMERA_IDS, and model paths before deployment."
}

run_smoke() {
    runtime_select_python
    local -a arguments=(scripts/smoke_test.py --env-file "${REID_ENV_FILE}")
    if runtime_bool_true "${SMOKE_LOAD_MODELS:-false}"; then
        arguments+=(--load-models)
    fi
    if [ "${1:-}" = "--load-models" ]; then
        arguments+=(--load-models)
    elif [ -n "${1:-}" ]; then
        echo "ERROR: Unknown smoke option: ${1}" >&2
        exit 2
    fi
    scripts/with_onnxruntime_cuda_env.sh "${RUNTIME_PYTHON_BIN}" "${arguments[@]}"
}

run_preflight() {
    runtime_select_python
    local role="${PIPELINE_ROLE}"
    local -a common=(scripts/jetson_preflight.py --role "${role}" --load-models)
    if [ "${role}" = "worker" ]; then
        common+=(--check-camera)
    fi
    scripts/with_onnxruntime_cuda_env.sh "${RUNTIME_PYTHON_BIN}" "${common[@]}"

    if [ "${role}" = "prime" ] && runtime_bool_true "${LOCAL_CAMERA_ENABLED:-true}"; then
        "${RUNTIME_PYTHON_BIN}" scripts/jetson_preflight.py \
            --role worker \
            --realtime-config "${REALTIME_CONFIG:-configs/realtime_config.yaml}" \
            --yolo-config "${YOLO_CONFIG:-configs/yolo_config.yaml}" \
            --load-models \
            --check-camera
    fi
}

start_runtime() {
    if runtime_bool_true "${STARTUP_SMOKE_TEST:-true}"; then
        run_smoke
    fi

    case "${PIPELINE_ROLE}" in
        prime)
            scripts/start_prime_dashboard.sh
            if runtime_bool_true "${LOCAL_CAMERA_ENABLED:-true}"; then
                scripts/start_worker_control.sh prime
            fi
            ;;
        worker)
            scripts/start_worker_control.sh worker
            ;;
    esac
}

stop_runtime() {
    case "${PIPELINE_ROLE}" in
        prime)
            if runtime_bool_true "${LOCAL_CAMERA_ENABLED:-true}"; then
                scripts/stop_worker_control.sh
            fi
            scripts/stop_prime_dashboard.sh
            ;;
        worker)
            scripts/stop_worker_control.sh
            ;;
    esac
}

show_status() {
    local prime_port="${PRIME_PORT:-8765}"
    local worker_port="${WORKER_CONTROL_PORT:-8787}"
    echo "role=${PIPELINE_ROLE} env=${REID_ENV_FILE}"
    printf 'prime_pids='
    pgrep -d, -f '[r]ealtime_prime.py' || echo
    printf 'control_pids='
    pgrep -d, -f '[r]ealtime_worker_control.py' || echo
    printf 'worker_pids='
    pgrep -d, -f '[r]ealtime_worker.py' || echo
    if [ "${PIPELINE_ROLE}" = "prime" ]; then
        if ! curl -fsS "http://127.0.0.1:${prime_port}/status" 2>/dev/null; then
            echo "prime_http=offline"
        else
            echo
        fi
    fi
    if ! curl -fsS "http://127.0.0.1:${worker_port}/status" 2>/dev/null; then
        echo "worker_control_http=offline"
    else
        echo
    fi
}

show_logs() {
    local log_root="${RUNTIME_LOG_DIR:-outputs}"
    if command -v systemctl >/dev/null 2>&1 \
        && systemctl --user is-active --quiet reid-prime.service 2>/dev/null; then
        journalctl --user -u reid-prime.service -n 80 --no-pager
    fi
    if command -v systemctl >/dev/null 2>&1 \
        && systemctl --user is-active --quiet reid-camera.service 2>/dev/null; then
        journalctl --user -u reid-camera.service -n 80 --no-pager
    fi
    if [ "${PIPELINE_ROLE}" = "prime" ]; then
        tail -n 80 "${log_root}/realtime/prime.log" 2>/dev/null || true
    fi
    tail -n 80 "${log_root}/realtime_worker_logs/control.log" 2>/dev/null || true
}

command="${1:-}"
shift || true

case "${command}" in
    init)
        init_environment "${1:-prime}"
        ;;
    smoke|preflight|start|stop|restart|status|logs)
        load_runtime_env true
        PIPELINE_ROLE="${PIPELINE_ROLE:-prime}"
        export PIPELINE_ROLE
        case "${command}" in
            smoke) run_smoke "${1:-}" ;;
            preflight) run_preflight ;;
            start) start_runtime ;;
            stop) stop_runtime ;;
            restart)
                stop_runtime
                sleep 1
                start_runtime
                ;;
            status) show_status ;;
            logs) show_logs ;;
        esac
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
