#!/usr/bin/env bash
# Helpers for launching ONNX Runtime GPU sessions from conda environments.

conda_env_prefix() {
    local env_name="$1"
    conda run --no-capture-output -n "${env_name}" python -c \
        'import sys; from pathlib import Path; print(Path(sys.executable).resolve().parents[1])'
}

conda_env_site_packages() {
    local env_name="$1"
    conda run --no-capture-output -n "${env_name}" python -c \
        'import site; print(site.getsitepackages()[0])'
}

prepend_onnxruntime_cuda_ld_library_path() {
    local prefix="$1"
    local site_packages="${2:-}"
    local additions=()
    local lib_dir

    if [ -z "${site_packages}" ]; then
        site_packages="$("${prefix}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
    fi

    if [ -d "${site_packages}/onnxruntime/capi" ]; then
        additions+=("${site_packages}/onnxruntime/capi")
    fi

    # ONNX Runtime's CUDA provider directly depends on cuBLAS and cuDNN. Avoid
    # adding every nvidia wheel directory: mixed CUDA-major packages can be
    # installed in one environment and loading both can crash the provider.
    for lib_dir in \
        "${site_packages}/nvidia/cublas/lib" \
        "${site_packages}/nvidia/cudnn/lib"; do
        if [ -d "${lib_dir}" ]; then
            additions+=("${lib_dir}")
        fi
    done

    if [ -d "/usr/local/cuda/lib64" ]; then
        additions+=("/usr/local/cuda/lib64")
    fi

    if [ "${#additions[@]}" -eq 0 ]; then
        return 0
    fi

    local joined
    joined="$(IFS=:; echo "${additions[*]}")"
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="${joined}:${LD_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="${joined}"
    fi
}
