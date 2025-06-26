# =============================================================================
# 6_train_death_model.py
#
# This script is specifically adapted to train a model for the mortality
# prediction task.
#
# Key Adaptations:
# 1.  Updated Paths: Points to the 'train_data_death.csv' and
#     'test_data_death.csv' manifest files.
# 2.  Correct Label: Renames 'death_label' to 'label' for AutoGluon.
# 3.  Dynamic Class Weights: Correctly calculates the Focal Loss alpha
#     weights based on the actual, severe class imbalance of the
#     mortality outcome.
# 4.  Updated Reporting: Changes the final classification report labels to
#     'Survived' and 'Died' for clarity.
# 5.  New Output Paths: Saves the model and results to new directories to
#     avoid overwriting previous experiments.
# 6.  Advanced Regularization: Adds MixUp and CutMix with CORRECTED
#     hyperparameter keys to combat overfitting.
# 7.  REMOVED `optimization.swa` due to KeyError.
# =============================================================================

# --- Standard Library Imports ---
import os
import pandas as pd
import numpy as np
import logging
import shutil
import traceback

# =============================================================================
# START: DYNAMIC & ROBUST PATH CONFIGURATION
# =============================================================================

# 1. Dynamically determine the project's root directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Define all key directories using absolute paths based on the project root.
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'models', 'autogluon_multimodel_death') 
RESULTS_FILE = os.path.join(MODEL_OUTPUT_PATH, 'evaluation_results_death.txt')
TRAIN_MANIFEST = os.path.join(PROCESSED_DATA_DIR, 'train_data_death.csv')
TEST_MANIFEST = os.path.join(PROCESSED_DATA_DIR, 'test_data_death.csv')

# 3. Create these directories if they don't exist to prevent errors.
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# 4. Set environment variables to control where libraries store files.
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['TORCH_HOME'] = os.path.join(CACHE_DIR, 'torch')
os.environ['TEMP'] = TEMP_DIR
os.environ['TMP'] = TEMP_DIR

# =============================================================================
# END: PATH CONFIGURATION
# =============================================================================

# --- Third-Party Imports ---
import torch
from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration Block ---

TIME_LIMIT_SECONDS = 3600 * 72  # 72 hours
EVAL_METRIC = 'roc_auc' 
PRESET_QUALITY = 'medium_quality'
MODEL_TO_TUNE = 'mobilenetv3_large_100'

# --- Main Execution Block ---

def main():
    """Main function to run the model training and evaluation pipeline."""
    torch.set_float32_matmul_precision('medium')

    if os.path.exists(MODEL_OUTPUT_PATH):
        logging.warning(f"Removing previous model directory to start a fresh run: {MODEL_OUTPUT_PATH}")
        shutil.rmtree(MODEL_OUTPUT_PATH)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
    logging.info("--- Starting Model Training for Mortality Prediction ---")
    
    try:
        logging.info(f"Loading training data from: {TRAIN_MANIFEST}")
        train_df = pd.read_csv(TRAIN_MANIFEST)
        test_df = pd.read_csv(TEST_MANIFEST)
        
        train_df = train_df.rename(columns={'death_label': 'label'})
        test_df = test_df.rename(columns={'death_label': 'label'})

        class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
        focal_alpha_list = class_weights.tolist()
        logging.info(f"Dynamically calculated Focal Loss alpha list for mortality: {focal_alpha_list}")

        predictor = MultiModalPredictor(label='label', problem_type='binary', eval_metric=EVAL_METRIC, presets=PRESET_QUALITY)

        logging.info(f"Starting model training with time limit: {TIME_LIMIT_SECONDS} seconds.")
        try:
            predictor.fit(
                train_data=train_df,
                save_path=MODEL_OUTPUT_PATH,
                hyperparameters={
                    # --- Model & Efficiency ---
                    'model.timm_image.checkpoint_name': MODEL_TO_TUNE,
                    'env.per_gpu_batch_size': 128,
                    'env.precision': '16-mixed',
                    'env.num_workers': 4,

                    # --- Core Training Objective & Imbalance Handling ---
                    'optim.loss_func': 'focal_loss',
                    'optim.focal_loss.alpha': focal_alpha_list,
                    'optim.focal_loss.gamma': 2.0,

                    # --- Existing Regularization ---
                    'model.timm_image.train_transforms': ['randaug'],
                    'optim.weight_decay': 1e-4, # Keep L2 regularization
                    
                    # --- Advanced Regularization with proper keys ---
                    'data.mixup.turn_on': True,
                    'data.mixup.mixup_alpha': 0.8,
                    'data.mixup.cutmix_alpha': 1.0, # Not active since alpha=1.0
                    'data.mixup.prob': 1.0,
                    'data.mixup.switch_prob': 0.5,
                    'data.mixup.mode': 'batch',
                    'data.mixup.label_smoothing': 0.1,

                    # --- REMOVED: This key caused a configuration error ---
                    # 'optimization.swa': True,

                    # --- Checkpointing ---
                    'optim.top_k': 3,
                },
                time_limit=TIME_LIMIT_SECONDS
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error("!!! CUDA OUT OF MEMORY ERROR CAUGHT !!!")
            else:
                raise e

    except (Exception, KeyboardInterrupt) as e:
        logging.error(f"An exception occurred during training: {e}")
        logging.error(traceback.format_exc())
    
    finally:
        logging.info("--- Entering Final Evaluation Block ---")
        
        try:
            if not os.path.exists(os.path.join(MODEL_OUTPUT_PATH, "assets.json")):
                logging.error("No model found to evaluate. Exiting.")
                return

            predictor = MultiModalPredictor.load(MODEL_OUTPUT_PATH)
            
            test_df = pd.read_csv(TEST_MANIFEST).rename(columns={'death_label': 'label'})
            
            scores = predictor.evaluate(data=test_df, metrics=[EVAL_METRIC, 'accuracy', 'f1'])
            logging.info(f"Initial evaluation scores (0.5 threshold): {scores}")

            logging.info("Finding optimal prediction threshold...")
            oof_sample = train_df.sample(n=min(50000, len(train_df)), random_state=42)
            y_true_oof = oof_sample['label']
            y_pred_proba_oof = predictor.predict_proba(oof_sample, as_pandas=True)
            positive_class_col = y_pred_proba_oof.columns[1]

            thresholds = np.arange(0.01, 1.0, 0.01)
            f1_scores = [f1_score(y_true_oof, (y_pred_proba_oof[positive_class_col] >= t).astype(int)) for t in thresholds]
            
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            logging.info(f"Optimal threshold found: {optimal_threshold:.4f}")
            
            y_true_test = test_df['label']
            y_pred_proba_test = predictor.predict_proba(test_df, as_pandas=True)
            y_pred_optimal = (y_pred_proba_test[positive_class_col] >= optimal_threshold).astype(int)
            
            report = classification_report(y_true_test, y_pred_optimal, target_names=['Survived', 'Died'])
            logging.info("Final Classification Report:\n" + report)

            logging.info(f"Saving final evaluation results to {RESULTS_FILE}")
            with open(RESULTS_FILE, 'w') as f:
                f.write("--- Mortality Model Evaluation Results (Advanced Regularization) ---\n\n")
                f.write(f"Model Tuned: {MODEL_TO_TUNE}\n")
                f.write(f"Evaluation Metric: {EVAL_METRIC}\n\n")
                f.write(f"Initial Scores (0.5 Threshold): {scores}\n\n")
                f.write(f"Optimal Threshold (for F1): {optimal_threshold:.4f}\n\n")
                f.write("Final Classification Report (Optimal Threshold):\n")
                f.write(report)

            logging.info("--- Model Training and Evaluation Complete ---")

        except Exception as final_e:
            logging.error(f"An error occurred during final evaluation: {final_e}")
            logging.error(traceback.format_exc())

if __name__ == '__main__':
    main()
