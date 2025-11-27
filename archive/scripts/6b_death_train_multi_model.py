# =============================================================================
# 7_train_model_flexible.py
#
# This is a fully refactored and flexible training script that incorporates
# the latest state-of-the-art training recipe.
#
# Key Improvements:
# 1.  Flexible Configuration: Easily pivot to new outcomes (like ICU
#     admission) by changing variables in the config block.
# 2.  Domain-Specific Augmentation: Replaced RandAugment with a custom
#     SpecAugment class designed for spectrograms.
# 3.  Improved Loss Function: Switched from Focal Loss to weighted
#     BCEWithLogitsLoss, as recommended.
# 4.  Weighted Data Sampling: Implemented a sampler to guarantee the model
#     sees rare positive cases in every batch.
# 5.  Conservative Regularization: Adopted a lighter touch for MixUp.
# 6.  Better Evaluation Metric: Using Average Precision (AUPRC), which is
#     more sensitive for highly imbalanced datasets.
# 7.  TensorBoard Launch: Automatically starts TensorBoard for live monitoring.
# =============================================================================
# =============================================================================
# 7_train_model_flexible.py  (UPDATED 2025-06-26)
#
# Key Improvements vs. previous version
# -------------------------------------
# • Loss: switched to Focal-Loss with γ = 0 and α = neg/pos   ← valid in AutoMM
# • Keeps WeightedRandomSampler + light MixUp for stability
# • All other behaviour unchanged
# =============================================================================

# --- Standard Library Imports ---
import os, logging, shutil, subprocess, traceback
from collections import Counter

import pandas as pd
import numpy as np

# --- Third-Party Imports ---
import torch
from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import classification_report, f1_score

# --- Local Imports ---
try:
    from spec_aug import SpecAugmentRGB
except ImportError:
    print("CRITICAL ERROR: spec_aug.py not found. Please ensure it is in the same directory.")
    exit()

# =============================================================================
# 1. EXPERIMENT CONFIGURATION
# =============================================================================
TARGET_LABEL_COLUMN = "death_label"
POSITIVE_CLASS_NAME = "Died"
NEGATIVE_CLASS_NAME = "Survived"
MODEL_DIR_SUFFIX     = "death_specaugment_focal"
EVAL_METRIC          = "average_precision"          # AUPRC

# =============================================================================
# 2. PATHS  (relative to repo root)
# =============================================================================
PROJECT_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR            = os.path.join(PROJECT_ROOT, "cache")
TEMP_DIR             = os.path.join(PROJECT_ROOT, "temp")
PROCESSED_DATA_DIR   = os.path.join(PROJECT_ROOT, "data", "processed")

MODEL_OUTPUT_PATH    = os.path.join(PROJECT_ROOT, "models",
                                    f"autogluon_multimodel_{MODEL_DIR_SUFFIX}")
RESULTS_FILE         = os.path.join(MODEL_OUTPUT_PATH,
                                    f"evaluation_results_{MODEL_DIR_SUFFIX}.txt")

MANIFEST_PREFIX      = TARGET_LABEL_COLUMN.replace("_label", "")
TRAIN_MANIFEST       = os.path.join(PROCESSED_DATA_DIR,
                                    f"train_data_{MANIFEST_PREFIX}.csv")
TEST_MANIFEST        = os.path.join(PROCESSED_DATA_DIR,
                                    f"test_data_{MANIFEST_PREFIX}.csv")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR,  exist_ok=True)
os.environ.update({
    "HF_HOME":   os.path.join(CACHE_DIR, "huggingface"),
    "TORCH_HOME":os.path.join(CACHE_DIR, "torch"),
    "TEMP":      TEMP_DIR,
    "TMP":       TEMP_DIR,
})

# =============================================================================
# 3. CONSTANTS
# =============================================================================
TIME_LIMIT_SECONDS = 72 * 3600
PRESET_QUALITY     = "medium_quality"
BACKBONE           = "mobilenetv3_large_100"

# =============================================================================
# 4. MAIN
# =============================================================================
def main() -> None:
    torch.set_float32_matmul_precision("medium")

    # Fresh run
    if os.path.exists(MODEL_OUTPUT_PATH):
        logging.warning(f"Removing previous model directory: {MODEL_OUTPUT_PATH}")
        shutil.rmtree(MODEL_OUTPUT_PATH)

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info(f"=== Training '{POSITIVE_CLASS_NAME}' predictor ({BACKBONE}) ===")

    # Optional TensorBoard launch
    try:
        subprocess.Popen(["tensorboard", "--logdir", MODEL_OUTPUT_PATH])
        logging.info("TensorBoard running at http://localhost:6006/")
    except FileNotFoundError:
        logging.info("TensorBoard not installed; skipping live dashboard.")

    # -------------------------------------------------------------------------
    # 4-A. Data loading & class-weight calculation
    # -------------------------------------------------------------------------
    train_df = pd.read_csv(TRAIN_MANIFEST).rename(columns={TARGET_LABEL_COLUMN: "label"})
    test_df  = pd.read_csv(TEST_MANIFEST ).rename(columns={TARGET_LABEL_COLUMN: "label"})

    cls_counts = Counter(train_df["label"].values)         # {0:neg, 1:pos}
    if cls_counts[1] == 0:
        logging.error("No positive samples in training data — aborting.")
        return

    # >>> TWO weights, one per class  [w_neg, w_pos]
    w_pos  = cls_counts[0] / cls_counts[1]                 # ≈ 145.3
    alpha  = [1.0, float(w_pos)]                           # len == n_classes
    logging.info(f"Focal-loss α set to [neg, pos] = {alpha}")
    # -------------------------------------------------------------------------
    # 4-B. Augmentation
    # -------------------------------------------------------------------------
    spec_aug = SpecAugmentRGB(freq_mask_param=24, time_mask_param=48, num_masks=2)
    logging.info(f"Using SpecAugmentRGB: {spec_aug}")

    # -------------------------------------------------------------------------
    # 4-C. Train
    # -------------------------------------------------------------------------
    predictor = MultiModalPredictor(
        label        = "label",
        problem_type = "binary",
        eval_metric  = EVAL_METRIC,
        presets      = PRESET_QUALITY,
    )

    logging.info(f"Starting fit (time limit: {TIME_LIMIT_SECONDS}s)")
    predictor.fit(
        train_data = train_df,
        save_path  = MODEL_OUTPUT_PATH,
        hyperparameters = {
            # Backbone & I/O
            "model.timm_image.checkpoint_name": BACKBONE,
            "model.timm_image.train_transforms": [spec_aug],
            "model.timm_image.val_transforms"  : ["resize_shorter_side",
                                                  "center_crop"],

            "env.per_gpu_batch_size": 128,
            "env.precision"         : "16-mixed",
            "env.num_workers"       : 4,

            # ---------- Loss: focal ≡ weighted BCE when γ = 0 ----------
            "optim.loss_func"         : "focal_loss",
            "optim.focal_loss.alpha"  : alpha,
            "optim.focal_loss.gamma"  : 0.0,

            # ---------- MixUp (light) ----------
            "data.mixup.turn_on"    : True,
            "data.mixup.mixup_alpha": 0.2,
            "data.mixup.cutmix_alpha": 1.0,   # 1.0 + mixup.prob → CutMix off
            "data.mixup.prob"       : 0.15,

            # ---------- Checkpoints ----------
            "optim.top_k": 3,
        },
        time_limit = TIME_LIMIT_SECONDS,
    )

    # -------------------------------------------------------------------------
    # 4-D. Evaluation
    # -------------------------------------------------------------------------
    logging.info("=== Evaluation ===")
    predictor = MultiModalPredictor.load(MODEL_OUTPUT_PATH)

    scores = predictor.evaluate(
        data   = test_df,
        metrics= [EVAL_METRIC, "accuracy", "f1", "roc_auc"],
    )
    logging.info(f"Initial scores (threshold = 0.5): {scores}")

    # Threshold search on a 50 k OOF sample
    oof_sample        = train_df.sample(n=min(50_000, len(train_df)), random_state=42)
    y_true_oof        = oof_sample["label"]
    proba_oof         = predictor.predict_proba(oof_sample, as_pandas=True)
    pos_col           = proba_oof.columns[1]

    thresholds        = np.arange(0.01, 1.0, 0.01)
    f1_scores         = [f1_score(y_true_oof,
                                  (proba_oof[pos_col] >= t).astype(int))
                         for t in thresholds]
    best_t            = thresholds[int(np.argmax(f1_scores))]
    logging.info(f"Optimal threshold for F1: {best_t:.3f}")

    proba_test        = predictor.predict_proba(test_df, as_pandas=True)
    y_pred_optimal    = (proba_test[pos_col] >= best_t).astype(int)
    report            = classification_report(
                            test_df["label"],
                            y_pred_optimal,
                            target_names=[NEGATIVE_CLASS_NAME,
                                          POSITIVE_CLASS_NAME])
    logging.info("\n" + report)

    # Save report
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    with open(RESULTS_FILE, "w") as fh:
        fh.write(f"=== {POSITIVE_CLASS_NAME} model ({MODEL_DIR_SUFFIX}) ===\n\n")
        fh.write(f"Backbone          : {BACKBONE}\n")
        fh.write(f"Evaluation metric : {EVAL_METRIC}\n\n")
        fh.write(f"Scores @0.5       : {scores}\n\n")
        fh.write(f"Optimal threshold : {best_t:.3f}\n\n")
        fh.write(report)
    logging.info(f"Results saved to {RESULTS_FILE}")
    logging.info("=== Done ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
