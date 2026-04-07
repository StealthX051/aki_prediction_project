#!/bin/bash

AKI_DEFAULT_ARTIFACT_ROOT="/media/volume/catch22/data/aki_prediction_project"
AKI_ALLOWED_STORAGE_ROOT="/media/volume/catch22"

aki_realpath() {
    realpath -m "$1"
}

aki_path_is_within_allowed_root() {
    local candidate
    local allowed
    candidate="$(aki_realpath "$1")"
    allowed="$(aki_realpath "${2:-$AKI_ALLOWED_STORAGE_ROOT}")"
    case "$candidate" in
        "$allowed"|"$allowed"/*) return 0 ;;
        *) return 1 ;;
    esac
}

aki_configure_catch22_env() {
    local script_dir="$1"
    local default_log_name="$2"

    AKI_ARTIFACT_ROOT="${AKI_ARTIFACT_ROOT:-$AKI_DEFAULT_ARTIFACT_ROOT}"
    AKI_STORAGE_POLICY="${AKI_STORAGE_POLICY:-enforce}"
    DATA_DIR="${DATA_DIR:-${AKI_ARTIFACT_ROOT}/data}"
    RAW_DIR="${RAW_DIR:-${script_dir}/data/raw}"
    PROCESSED_DIR="${PROCESSED_DIR:-${DATA_DIR}/processed}"
    RESULTS_DIR="${RESULTS_DIR:-${AKI_ARTIFACT_ROOT}/results/catch22/experiments}"
    PAPER_DIR="${PAPER_DIR:-${AKI_ARTIFACT_ROOT}/results/catch22/paper}"
    SMOKE_ROOT="${SMOKE_ROOT:-${AKI_ARTIFACT_ROOT}/smoke_test_outputs}"
    LOG_FILE="${LOG_FILE:-${AKI_ARTIFACT_ROOT}/logs/${default_log_name}}"

    export AKI_ARTIFACT_ROOT AKI_STORAGE_POLICY DATA_DIR RAW_DIR PROCESSED_DIR RESULTS_DIR PAPER_DIR SMOKE_ROOT LOG_FILE
}

aki_enforce_generated_paths() {
    local policy="${AKI_STORAGE_POLICY:-enforce}"
    local entry
    local label
    local path

    if [[ "$policy" == "off" ]]; then
        return 0
    fi

    for entry in "$@"; do
        label="${entry%%::*}"
        path="${entry#*::}"
        if aki_path_is_within_allowed_root "$path"; then
            continue
        fi

        local message
        message="Generated artifact path must live under ${AKI_ALLOWED_STORAGE_ROOT} when AKI_STORAGE_POLICY=${policy}: ${label}=${path}"
        if [[ "$policy" == "warn" ]]; then
            echo "Warning: ${message}" >&2
            continue
        fi
        echo "Error: ${message}" >&2
        echo "Set AKI_STORAGE_POLICY=warn or AKI_STORAGE_POLICY=off to bypass this check." >&2
        return 1
    done
}

aki_ensure_repo_symlink() {
    local link_path="$1"
    local target_path="$2"

    mkdir -p "$(dirname "$target_path")"
    mkdir -p "$(dirname "$link_path")"

    if [[ -L "$link_path" ]]; then
        rm -f "$link_path"
        ln -s "$target_path" "$link_path"
        return 0
    fi

    if [[ -e "$link_path" ]]; then
        if [[ -d "$link_path" ]] && [[ -z "$(find "$link_path" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            rmdir "$link_path"
            ln -s "$target_path" "$link_path"
            return 0
        fi
        echo "Warning: repo convenience path ${link_path} already exists as a real path; leaving it in place." >&2
        return 0
    fi

    ln -s "$target_path" "$link_path"
}

aki_refresh_repo_symlinks() {
    local script_dir="$1"
    aki_ensure_repo_symlink "${script_dir}/data/processed" "${PROCESSED_DIR}"
    aki_ensure_repo_symlink "${script_dir}/results/catch22/experiments" "${RESULTS_DIR}"
    aki_ensure_repo_symlink "${script_dir}/results/catch22/paper" "${PAPER_DIR}"
}
