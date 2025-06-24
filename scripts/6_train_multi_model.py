# =============================================================================
# 6_train_multi_model.py
#
# This script executes the full, unattended model training phase (Phase 6)
# of the project, with robust error handling for unattended execution.
#
# Key Features:
# - Redirects cache AND temporary directories to the project folder.
# - Loads the complete train_data.csv and test_data.csv manifests.
# - Instantiates AutoGluon's MultiModalPredictor with performance optimizations.
# - Fine-tunes a powerful pre-trained model.
# - Implements robust error handling and guarantees final evaluation.
# =============================================================================

# --- Standard Library Imports ---
import os
import pandas as pd
import logging
import shutil
import traceback

# =============================================================================
# START: DYNAMIC & ROBUST PATH CONFIGURATION
# This block automatically determines the project's root directory and sets
# all necessary paths on the D: drive, including the critical TEMP folder.
# This MUST be done before importing torch, autogluon, etc.
# =============================================================================

# 1. Dynamically determine the project's root directory.
#    This makes the script portable and independent of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Define all key directories using absolute paths based on the project root.
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp') # For hidden temporary files
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'models', 'autogluon_multimodel')
RESULTS_FILE = os.path.join(MODEL_OUTPUT_PATH, 'evaluation_results.txt')
TRAIN_MANIFEST = os.path.join(PROCESSED_DATA_DIR, 'train_data.csv')
TEST_MANIFEST = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')

# 3. Create these directories if they don't exist to prevent errors.
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
# The model output path will be created/cleared in main()

# 4. Set environment variables to control where libraries store files.
#    This is the most critical step to prevent filling up the C: drive.
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['TORCH_HOME'] = os.path.join(CACHE_DIR, 'torch')
os.environ['TEMP'] = TEMP_DIR
os.environ['TMP'] = TEMP_DIR

# =============================================================================
# END: PATH CONFIGURATION
# It is now safe to import machine learning libraries.
# =============================================================================

# --- Third-Party Imports ---
import torch
from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import classification_report

# --- Configuration Block ---

# NOTE: All path configuration is now handled above.

# 2. Model and Training Configuration
TIME_LIMIT_SECONDS = 3600 * 17  # 9 hours
EVAL_METRIC = 'roc_auc'
PRESET_QUALITY = 'medium_quality'

# We will train one model per run for robust comparison.
MODEL_TO_TUNE = 'convnextv2_nano'

# --- Main Execution Block ---

def main():
    """Main function to run the model training and evaluation pipeline."""
    # 1. Setup and Performance Optimization
    torch.set_float32_matmul_precision('medium')

    # Automatically clear previous model directory for a fresh run
    if os.path.exists(MODEL_OUTPUT_PATH):
        logging.warning(f"Removing previous model directory to start a fresh run: {MODEL_OUTPUT_PATH}")
        shutil.rmtree(MODEL_OUTPUT_PATH)
    
    # Configure logging for initial steps. AutoGluon will set up its own file logger.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.info("--- Starting Full Model Training and Evaluation ---")
    # Log all critical environment paths to confirm they are set correctly.
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
        
        # 3. Rename label column
        train_df = train_df.rename(columns={'aki_label': 'label'})
        test_df = test_df.rename(columns={'aki_label': 'label'})

        # 4. Initialize Predictor without a path
        predictor = MultiModalPredictor(
            label='label',
            problem_type='binary',
            eval_metric=EVAL_METRIC,
            presets=PRESET_QUALITY
        )

        # 5. Fit Predictor with OOM handling
        logging.info(f"Starting model training with time limit: {TIME_LIMIT_SECONDS} seconds.")
        logging.info(f"Model to be tuned: {MODEL_TO_TUNE}")
        try:
            # Pass the `save_path` here. AutoGluon will create this directory.
            predictor.fit(
                train_data=train_df,
                save_path=MODEL_OUTPUT_PATH,
                hyperparameters={
                    # backbone
                    'model.timm_image.checkpoint_name': MODEL_TO_TUNE,

                    # loss
                    'optim.loss_func': 'focal_loss',

                    # efficiency
                    'env.per_gpu_batch_size': 32,     # fits with fp16 on 6 GB
                    'env.batch_size': 256,            # grad-accum keeps effective batch large
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
        logging.info("--- Entering Final Evaluation Block (runs on success, error, or interrupt) ---")
        
        try:
            # Check if the model directory was created and has content before loading
            if not os.path.exists(os.path.join(MODEL_OUTPUT_PATH, "assets.json")):
                logging.error("Model training did not complete successfully. No model to evaluate. Exiting.")
                return

            logging.info(f"Loading predictor from path: {MODEL_OUTPUT_PATH}")
            predictor = MultiModalPredictor.load(MODEL_OUTPUT_PATH)
            
            test_df = pd.read_csv(TEST_MANIFEST).rename(columns={'aki_label': 'label'})

            scores = predictor.evaluate(data=test_df, metrics=[EVAL_METRIC, 'accuracy', 'f1'])
            logging.info(f"Final evaluation scores on the test set: {scores}")

            y_true = test_df['label']
            y_pred_proba = predictor.predict_proba(test_df, as_pandas=True)
            positive_class_col = y_pred_proba.columns[1]
            y_pred = (y_pred_proba[positive_class_col] >= 0.5).astype(int)
            report = classification_report(y_true, y_pred, target_names=['No AKI', 'AKI'])
            logging.info("Classification Report:\n" + report)

            logging.info(f"Saving final evaluation results to {RESULTS_FILE}")
            with open(RESULTS_FILE, 'w') as f:
                f.write("--- Model Evaluation Results ---\n\n")
                f.write(f"Final Scores: {scores}\n\n")
                f.write("Classification Report:\n")
                f.write(report)

            logging.info("--- Model Training and Evaluation Complete ---")

        except Exception as final_e:
            logging.error(f"An error occurred during the final evaluation phase: {final_e}")
            logging.error(traceback.format_exc())

if __name__ == '__main__':
    main()
