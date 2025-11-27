# =============================================================================
# 7_continue_training.py
#
# ### MODIFICATION ###
# This script continues an existing training run from a SPECIFIC checkpoint.
# It loads a user-specified .ckpt file from a previous run,
# continues the .fit() process, and saves all new artifacts to a
# separate 'autogluon_multimodel_continued' directory.
#
# It is now configured to save only the top 3 best-performing epoch
# checkpoints to conserve disk space.
# =============================================================================

# --- Standard Library Imports ---
import os
import pandas as pd
import logging
import shutil
import traceback

# =============================================================================
# START: DYNAMIC & ROBUST PATH CONFIGURATION
# This block automatically determines the project's root directory.
# =============================================================================

# 1. Dynamically determine the project's root directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Define all key directories.
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# ### MODIFICATION ###: Define paths for BOTH the original and new model directories.
ORIGINAL_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'autogluon_multimodel')
CONTINUED_MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'models', 'autogluon_multimodel_continued')
RESULTS_FILE = os.path.join(CONTINUED_MODEL_OUTPUT_PATH, 'evaluation_results_continued.txt') # Save results in new dir

TRAIN_MANIFEST = os.path.join(PROCESSED_DATA_DIR, 'train_data.csv')
TEST_MANIFEST = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')

# 3. Create cache and temp directories if they don't exist.
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CONTINUED_MODEL_OUTPUT_PATH, exist_ok=True) # Ensure the new output dir exists

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
from sklearn.metrics import classification_report

# --- Configuration Block ---

TIME_LIMIT_SECONDS = 3600 * 24  # 24 more hours
EVAL_METRIC = 'roc_auc'

# ### MODIFICATION ###: Define the path to the BEST unsouped checkpoint.
# YOU MUST UPDATE THIS FILENAME. This is the checkpoint you copied.
# It should be placed inside the original model directory to be found.
BEST_INDIVIDUAL_CHECKPOINT_FILENAME = "epoch=17-step=8500.ckpt" # <-- CHANGE THIS
BEST_INDIVIDUAL_CKPT_PATH = os.path.join(ORIGINAL_MODEL_PATH, BEST_INDIVIDUAL_CHECKPOINT_FILENAME)

# --- Main Execution Block ---

def main():
    """Main function to continue model training from a specific checkpoint."""
    # 1. Setup and Performance Optimization
    torch.set_float32_matmul_precision('medium')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.info("--- Starting Continued Model Training from Specific Checkpoint ---")
    logging.info(f"New model and artifacts will be saved to: {CONTINUED_MODEL_OUTPUT_PATH}")

    try:
        # 2. Load Data (Still needed for the .fit call)
        logging.info(f"Loading training data from: {TRAIN_MANIFEST}")
        train_df = pd.read_csv(TRAIN_MANIFEST)
        logging.info(f"Training data shape: {train_df.shape}")

        # 3. Rename label column
        train_df = train_df.rename(columns={'aki_label': 'label'})

        # 4. ### MODIFICATION ###: Load the predictor from a SPECIFIC checkpoint file.
        logging.info(f"Loading specific predictor state from: {BEST_INDIVIDUAL_CKPT_PATH}")

        if not os.path.exists(BEST_INDIVIDUAL_CKPT_PATH):
            logging.error(f"FATAL: The specified checkpoint could not be found.")
            logging.error(f"Path checked: {BEST_INDIVIDUAL_CKPT_PATH}")
            logging.error("Please ensure the filename is correct and the file is in the original model directory.")
            return

        predictor = MultiModalPredictor.load(BEST_INDIVIDUAL_CKPT_PATH)
        logging.info("Predictor from specific checkpoint loaded successfully.")

        # 5. Fit Predictor
        logging.info(f"Continuing model training with new time limit: {TIME_LIMIT_SECONDS} seconds.")
        try:
            # ### MODIFICATION ###: Configure checkpointing to save only the top 3 models.
            predictor.fit(
                train_data=train_df,
                save_path=CONTINUED_MODEL_OUTPUT_PATH,
                hyperparameters={
                    # This tells AutoGluon to only keep the 3 best checkpoints based on validation roc_auc.
                    'optimization.checkpoint.save_top_k': 3,

                    # The `clean_ckpts` parameter is no longer needed as `save_top_k` handles it.
                    # Other efficiency settings remain:
                    'env.precision': '16-mixed',
                    'env.per_gpu_batch_size': 32,
                    'env.num_workers': 4,
                },
                time_limit=TIME_LIMIT_SECONDS
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error("!!! CUDA OUT OF MEMORY ERROR CAUGHT !!!")
                logging.error("Proceeding to evaluate the best model from the run.")
            else:
                raise e

    except (Exception, KeyboardInterrupt) as e:
        logging.error(f"An exception or interrupt occurred during the training phase: {e}")
        logging.error(traceback.format_exc())

    finally:
        # --- This block ALWAYS runs, ensuring evaluation happens ---
        logging.info("--- Entering Final Evaluation Block ---")
        try:
            # ### MODIFICATION ###: Check for and load the model from the NEW output path.
            if not os.path.exists(os.path.join(CONTINUED_MODEL_OUTPUT_PATH, "model.ckpt")):
                logging.error("Continued model training did not complete successfully. No new model to evaluate. Exiting.")
                return

            logging.info(f"Loading final predictor from continued run path: {CONTINUED_MODEL_OUTPUT_PATH}")
            final_predictor = MultiModalPredictor.load(CONTINUED_MODEL_OUTPUT_PATH)

            test_df = pd.read_csv(TEST_MANIFEST).rename(columns={'aki_label': 'label'})

            scores = final_predictor.evaluate(data=test_df, metrics=[EVAL_METRIC, 'accuracy', 'f1'])
            logging.info(f"Final evaluation scores on the test set: {scores}")

            y_true = test_df['label']
            y_pred_proba = final_predictor.predict_proba(test_df, as_pandas=True)
            positive_class_col = y_pred_proba.columns[1]
            y_pred = (y_pred_proba[positive_class_col] >= 0.5).astype(int)
            report = classification_report(y_true, y_pred, target_names=['No AKI', 'AKI'])
            logging.info("Final Classification Report:\n" + report)

            logging.info(f"Saving final evaluation results to {RESULTS_FILE}")
            with open(RESULTS_FILE, 'w') as f:
                f.write("--- Model Evaluation Results (Continued Run) ---\n\n")
                f.write(f"Final Scores: {scores}\n\n")
                f.write("Classification Report:\n")
                f.write(report)

            logging.info("--- Model Training and Evaluation Complete ---")

        except Exception as final_e:
            logging.error(f"An error occurred during the final evaluation phase: {final_e}")
            logging.error(traceback.format_exc())

if __name__ == '__main__':
    main()
