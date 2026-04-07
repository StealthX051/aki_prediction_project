import argparse
import json
import logging
from pathlib import Path
import sys

# Add project root to path to import utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_creation import utils
from model_creation.validation import select_modeling_dataset, tune_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_hpo(
    outcome,
    branch,
    feature_set,
    n_trials=100,
    smoke_test=False,
    model_type="xgboost",
    inner_folds: int = 5,
    threads_per_model: int = 8,
    legacy_imputation: bool = False,
):
    logger.info(
        "Starting HPO for Outcome: %s, Branch: %s, Feature Set: %s, Model: %s",
        outcome,
        branch,
        feature_set,
        model_type,
    )

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
    )

    logger.info("HPO complete. Best AUPRC: %.4f", best_value)
    logger.info("Best Params: %s", best_params)

    best_params["scale_pos_weight"] = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    best_params["actual_inner_folds"] = actual_inner_folds
    best_params["validation_scheme"] = "holdout"
    best_params["threads_per_model"] = threads_per_model

    output_dir = utils.RESULTS_DIR / "params" / model_type / outcome / branch
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
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["xgboost", "ebm"],
        default="xgboost",
        help="Model type to optimize",
    )
    parser.add_argument("--inner-folds", type=int, default=5, help="Grouped inner folds for HPO.")
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

    args = parser.parse_args()

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
    )
