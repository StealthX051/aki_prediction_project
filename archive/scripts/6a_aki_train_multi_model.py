# =============================================================================
# 6_train_multi_model_v2.py
#
# This script is an updated version designed to address the failures of the
# initial training run by properly handling severe class imbalance.
#
# Key Fixes Implemented:
# 1.  Model Change: Switched to a MobileNetV3-Large model to resolve the
#     unknown model error.
# 2.  Corrected Loss Function: Implements `focal_loss` with a hardcoded
#     `alpha` parameter to bypass a persistent hyperparameter parsing error.
# 3.  Aligned Evaluation Metric: Changed the `eval_metric` to
#     `balanced_accuracy` to ensure the best-saved model is the one that
#     performs most equally on both classes.
# 4.  Optimal Thresholding: After training, the script now finds the optimal
#     prediction threshold on the training data to maximize the F1 score.
# 5.  Checkpointing: Saves the top 3 model checkpoints for later use.
# 6.  Enhanced Regularization: Added stronger data augmentation and weight
#     decay with corrected hyperparameter names to combat overfitting.
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
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'models', 'autogluon_multimodel_v2') # New output path
RESULTS_FILE = os.path.join(MODEL_OUTPUT_PATH, 'evaluation_results_v2.txt') # New results file
TRAIN_MANIFEST = os.path.join(PROCESSED_DATA_DIR, 'train_data.csv')
TEST_MANIFEST = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')

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

# --- Configuration Block ---

# 1. Model and Training Configuration
TIME_LIMIT_SECONDS = 3600 * 22  # 22 hours
EVAL_METRIC = 'balanced_accuracy'
PRESET_QUALITY = 'medium_quality'
MODEL_TO_TUNE = 'mobilenetv3_large_100'

# --- Main Execution Block ---

def main():
    """Main function to run the model training and evaluation pipeline."""
    # 1. Setup and Performance Optimization
    torch.set_float32_matmul_precision('medium')

    # Automatically clear previous model directory for a fresh run
    if os.path.exists(MODEL_OUTPUT_PATH):
        logging.warning(f"Removing previous model directory to start a fresh run: {MODEL_OUTPUT_PATH}")
        shutil.rmtree(MODEL_OUTPUT_PATH)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.info("--- Starting Full Model Training and Evaluation (v2) ---")
    logging.info(f"HuggingFace cache set to: {os.environ.get('HF_HOME')}")
    logging.info(f"PyTorch cache set to: {os.environ.get('TORCH_HOME')}")
    logging.info(f"Temporary file directory set to: {os.environ.get('TEMP')}")

    try:
        # 2. Load Data
        logging.info(f"Loading training data from: {TRAIN_MANIFEST}")
        train_df = pd.read_csv(TRAIN_MANIFEST)
        test_df = pd.read_csv(TEST_MANIFEST)
        
        logging.info(f"Training data shape: {train_df.shape}")
        logging.info(f"Testing data shape: {test_df.shape}")
        
        # 3. Rename label column for AutoGluon
        train_df = train_df.rename(columns={'aki_label': 'label'})
        test_df = test_df.rename(columns={'aki_label': 'label'})

        # 4. NEW: Hardcode weights for Focal Loss alpha to debug ValueError
        # The dynamic calculation was correct but caused a hyperparameter
        # parsing error. Using a simple, hardcoded list is a robust workaround.
        # These values correspond to a ~95% / 5% class split.
        focal_alpha_list = [0.5338, 7.8913]
        logging.info(f"Using hardcoded Focal Loss alpha list: {focal_alpha_list}")

        # 5. Initialize Predictor
        predictor = MultiModalPredictor(
            label='label',
            problem_type='binary',
            eval_metric=EVAL_METRIC,
            presets=PRESET_QUALITY
        )

        # 6. Fit Predictor with updated hyperparameters
        logging.info(f"Starting model training with time limit: {TIME_LIMIT_SECONDS} seconds.")
        logging.info(f"Model to be tuned: {MODEL_TO_TUNE}")
        logging.info(f"Evaluation metric: {EVAL_METRIC}")
        try:
            predictor.fit(
                train_data=train_df,
                save_path=MODEL_OUTPUT_PATH,
                hyperparameters={
                    # Model backbone
                    'model.timm_image.checkpoint_name': MODEL_TO_TUNE,
                    
                    # Data Augmentation
                    'model.timm_image.train_transforms': ['randaug'],

                    # Loss function for imbalance
                    'optim.loss_func': 'focal_loss',
                    'optim.focal_loss.alpha': focal_alpha_list, # Pass the list here
                    'optim.focal_loss.gamma': 2.0,
                    
                    # Checkpointing and Weight Decay
                    'optim.top_k': 3,
                    'optim.weight_decay': 1e-4, # Add L2 regularization

                    # Efficiency
                    'env.per_gpu_batch_size': 100, # 32 safe for convnextv2_nano 3.6 gb, probably can up to 56. 56 safe for mobilentv3_large_100, upping to 100
                    'env.batch_size': 256,
                    'env.precision': '16-mixed',
                    'env.num_workers': 4,
                },
                time_limit=TIME_LIMIT_SECONDS
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error("!!! CUDA OUT OF MEMORY ERROR CAUGHT !!!")
                logging.error("Proceeding to evaluate any models that finished training.")
            else:
                raise e

    except (Exception, KeyboardInterrupt) as e:
        logging.error(f"An exception or interrupt occurred during the training phase: {e}")
        logging.error(traceback.format_exc())
    
    finally:
        # --- This block ALWAYS runs, ensuring evaluation happens ---
        logging.info("--- Entering Final Evaluation Block ---")
        
        try:
            if not os.path.exists(os.path.join(MODEL_OUTPUT_PATH, "assets.json")):
                logging.error("Model training did not complete successfully. No model to evaluate. Exiting.")
                return

            logging.info(f"Loading predictor from path: {MODEL_OUTPUT_PATH}")
            predictor = MultiModalPredictor.load(MODEL_OUTPUT_PATH)
            
            # Re-load test data to be safe
            test_df = pd.read_csv(TEST_MANIFEST).rename(columns={'aki_label': 'label'})
            
            # Get initial scores
            scores = predictor.evaluate(data=test_df, metrics=[EVAL_METRIC, 'roc_auc', 'accuracy', 'f1'])
            logging.info(f"Initial evaluation scores (using 0.5 threshold): {scores}")

            # --- Find Optimal Prediction Threshold ---
            logging.info("Finding optimal prediction threshold to maximize F1-score...")
            oof_sample = train_df.sample(n=min(50000, len(train_df)), random_state=42)
            y_true_oof = oof_sample['label']
            y_pred_proba_oof = predictor.predict_proba(oof_sample, as_pandas=True)
            positive_class_col = y_pred_proba_oof.columns[1]

            thresholds = np.arange(0.01, 1.0, 0.01)
            f1_scores = [f1_score(y_true_oof, (y_pred_proba_oof[positive_class_col] >= t).astype(int)) for t in thresholds]
            
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            logging.info(f"Optimal threshold found: {optimal_threshold:.4f} (with F1 score: {f1_scores[optimal_idx]:.4f})")
            
            # --- Generate Final Report with Optimal Threshold ---
            logging.info(f"Generating final classification report using optimal threshold of {optimal_threshold:.4f}")
            y_true_test = test_df['label']
            y_pred_proba_test = predictor.predict_proba(test_df, as_pandas=True)
            y_pred_optimal = (y_pred_proba_test[positive_class_col] >= optimal_threshold).astype(int)
            
            report = classification_report(y_true_test, y_pred_optimal, target_names=['No AKI', 'AKI'])
            logging.info("Final Classification Report:\n" + report)

            logging.info(f"Saving final evaluation results to {RESULTS_FILE}")
            with open(RESULTS_FILE, 'w') as f:
                f.write("--- Model Evaluation Results (v2) ---\n\n")
                f.write(f"Model Tuned: {MODEL_TO_TUNE}\n")
                f.write(f"Evaluation Metric: {EVAL_METRIC}\n\n")
                f.write(f"Initial Scores (0.5 Threshold): {scores}\n\n")
                f.write(f"Optimal Threshold (for F1): {optimal_threshold:.4f}\n\n")
                f.write("Final Classification Report (using Optimal Threshold):\n")
                f.write(report)

            logging.info("--- Model Training and Evaluation Complete ---")

        except Exception as final_e:
            logging.error(f"An error occurred during the final evaluation phase: {final_e}")
            logging.error(traceback.format_exc())

if __name__ == '__main__':
    main()
