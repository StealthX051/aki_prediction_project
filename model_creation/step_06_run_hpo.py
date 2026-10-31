import argparse
import json
import logging
import os
from pathlib import Path
import sys

# Add project root to path to import utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_creation import utils
from artifact_paths import get_scoped_results_dir
from data_preparation.cohort_scopes import COHORT_SCOPES
from model_creation.validation import select_modeling_dataset, tune_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_hpo(
    outcome,
    branch,
    feature_set,
    n_trials=None,
    smoke_test=False,
    model_type="xgboost",
    inner_folds: int = 3,
    threads_per_model: int = 8,
    legacy_imputation: bool = False,
    cohort_scope: str = "main_shared",
):
    logger.info(
        "Starting HPO for Outcome: %s, Branch: %s, Feature Set: %s, Model: %s",
        outcome,
        branch,
        feature_set,
        model_type,
    )

    n_trials = (20 if model_type == "logistic_regression" else 50) if n_trials is None else n_trials
    if smoke_test:
        logger.info("SMOKE TEST MODE: running with reduced trials.")
        n_trials = min(n_trials, 2)

    preserve_nan = not legacy_imputation

    try:
        df = utils.load_data(branch)
        working_df, X, y, _, groups = select_modeling_dataset(
            df,
            outcome,
            feature_set,
            require_holdout_split=True,
            cohort_scope=cohort_scope,
        )
        train_mask = working_df["split_group"] == "train"
        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        train_groups = groups.loc[train_mask] if groups is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to prepare data: %s", exc)
        raise SystemExit(1) from exc

    best_params, best_value, actual_inner_folds = tune_model(
        X_train,
        y_train,
        train_groups,
        model_type=model_type,
        n_trials=n_trials,
        requested_splits=inner_folds,
        random_state=42,
        preserve_nan=preserve_nan,
        threads_per_model=threads_per_model,
        diagnostic=smoke_test,
    )

    logger.info("HPO complete. Best AUPRC: %.4f", best_value)
    logger.info("Best Params: %s", best_params)

    if model_type == "xgboost":
        best_params["scale_pos_weight"] = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    best_params["actual_inner_folds"] = actual_inner_folds
    best_params["validation_scheme"] = "holdout"
    best_params["threads_per_model"] = threads_per_model
    best_params["cohort_scope"] = cohort_scope
    best_params["diagnostic"] = bool(smoke_test)
    best_params["compute_profile"] = (
        "diagnostic_ebm_200_rounds_2_outer_bags_no_inner_bags"
        if model_type == "ebm" and smoke_test else "manuscript"
    )
    best_params["n_trials"] = n_trials

    output_dir = get_scoped_results_dir(PROJECT_ROOT, cohort_scope, "holdout") / "params" / model_type / outcome / branch
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{feature_set}.json"

    with output_file.open("w") as fp:
        json.dump(best_params, fp, indent=4)

    logger.info("Best parameters saved to %s", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HPO for AKI Prediction Models")
    parser.add_argument("--outcome", type=str, required=True, help="Target outcome name")
    parser.add_argument(
        "--branch",
        type=str,
        required=True,
        choices=["windowed", "non_windowed"],
        help="Data branch",
    )
    parser.add_argument("--feature_set", type=str, required=True, help="Feature set name")
    parser.add_argument("--n_trials", type=int, default=None, help="Trials (default: LR20, other families50)")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["xgboost", "ebm", "logistic_regression"],
        default="xgboost",
        help="Model type to optimize",
    )
    parser.add_argument("--inner-folds", type=int, default=3, help="Grouped inner folds for HPO.")
    parser.add_argument(
        "--threads-per-model",
        type=int,
        default=8,
        help="Threads allocated to each fitted model.",
    )
    parser.add_argument(
        "--legacy_imputation",
        action="store_true",
        help="Apply fold-local imputation instead of preserving NaNs.",
    )

    parser.add_argument("--cohort-scope", choices=COHORT_SCOPES, default="main_shared")
    args = parser.parse_args()
    os.environ.setdefault("AKI_SPLIT_CACHE_DIR", str(
        get_scoped_results_dir(PROJECT_ROOT, args.cohort_scope, "holdout") / "metadata" / "split_cache"))

    run_hpo(
        args.outcome,
        args.branch,
        args.feature_set,
        args.n_trials,
        args.smoke_test,
        args.model_type,
        args.inner_folds,
        args.threads_per_model,
        args.legacy_imputation,
        args.cohort_scope,
    )
