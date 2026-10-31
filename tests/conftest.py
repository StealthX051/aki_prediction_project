"""Keep ordinary pytest runs offline and isolated before project imports."""

from pathlib import Path
import tempfile

import pytest


def pytest_configure(config):
    root = Path(tempfile.mkdtemp(prefix="aki-pytest-"))
    environment = pytest.MonkeyPatch()
    config.add_cleanup(environment.undo)
    config._aki_artifact_root = root
    paths = {
        "AKI_ARTIFACT_ROOT": root,
        "DATA_DIR": root / "data",
        "RAW_DIR": root / "no-research-data",
        "INPUT_FILE": root / "no-research-data/clinical_data.csv",
        "LAB_DATA_FILE": root / "no-research-data/lab_data.csv",
        "PROCESSED_DIR": root / "processed",
        "RESULTS_DIR": root / "results",
        "PAPER_DIR": root / "paper",
        "FIGURES_DIR": root / "paper/figures",
        "TABLES_DIR": root / "paper/tables",
        "REPORTS_DIR": root / "paper/reports",
        "XAI_DIR": root / "xai",
        "SMOKE_ROOT": root / "smoke",
        "AEON_OUT_DIR": root / "aeon",
        "AKI_SPLIT_CACHE_DIR": root / "split-cache",
        "LOG_FILE": root / "imports.log",
        "MPLCONFIGDIR": root / "matplotlib",
        "NUMBA_CACHE_DIR": root / "numba-cache",
    }
    for name, path in paths.items():
        environment.setenv(name, str(path))
    for name in (
        "COHORT_FILE", "CATCH_22_FILE", "CATCH_22_ERROR_FILE", "PREOP_PROCESSED_FILE",
        "CATCH_22_WINDOWED_FILE", "INTRAOP_WIDE_FILE", "INTRAOP_WIDE_WINDOWED_FILE",
        "WIDE_FEATURES_FILE", "WIDE_FEATURES_WINDOWED_FILE", "MERGED_DATASET_PATH",
        "RAW_SOURCE_DIR", "COHORT_CSV", "MERGED_DATASET", "COUNTS_FILE", "PROCESSED_PREOP",
        "DISPLAY_DICT", "DISPLAY_DICTIONARY_PATH", "COHORT_SCOPE", "VALIDATION_SCHEME", "BRANCH_ROSTER",
        "OUTER_FOLDS", "INNER_FOLDS", "CALIBRATION_FOLDS", "REPEATS", "MAX_WORKERS",
        "THREADS_PER_MODEL", "N_TRIALS", "BLAS_THREADS", "PARALLEL_BACKEND",
        "CASE_LIMIT", "HPO_TRIALS", "SMOKE_OUTCOMES",
    ):
        environment.delenv(name, raising=False)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        environment.setenv(name, "2")
    environment.setenv("AKI_STORAGE_POLICY", "off")
    environment.setenv("MPLBACKEND", "Agg")
    environment.setenv("GENERATE_FULL_FEATURES", "True")
    environment.setenv("GENERATE_WINDOWED_FEATURES", "False")


def pytest_report_header(config):
    return f"Isolated study artifacts: {config._aki_artifact_root}"


@pytest.fixture(autouse=True)
def no_live_vitaldb_requests(monkeypatch):
    import vitaldb

    def unexpected_request(*args, **kwargs):
        pytest.fail("Offline tests must replace VitalDB data requests with a synthetic fixture.")

    for name in ("load_case", "find_cases", "load_clinical_data", "load_lab_data"):
        if hasattr(vitaldb, name):
            monkeypatch.setattr(vitaldb, name, unexpected_request)
