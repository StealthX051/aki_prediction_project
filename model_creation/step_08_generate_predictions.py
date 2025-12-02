import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd
import xgboost as xgb
import numpy as np

# Add project root to path to import utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_creation import utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_predictions_for_model(model_path: Path):
    """
    Generates predictions for a single model and saves them to predictions.csv.
    """
    try:
        # Infer configuration from path
        # Expected path: .../results/models/{outcome}/{branch}/{feature_set}/model.json
        feature_set = model_path.parent.name
        branch = model_path.parent.parent.name
        outcome = model_path.parent.parent.parent.name
        
        logger.info(f"Processing Model: Outcome={outcome}, Branch={branch}, FeatureSet={feature_set}")
        
        # Load Model
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        
        # Load Data
        df = utils.load_data(branch)
        
        # --- MANUAL DATA PREP (Critical for preserving caseid) ---
        # 1. Filter for target outcome existence
        target_col = utils.OUTCOMES.get(outcome)
        if not target_col:
            logger.error(f"Invalid outcome: {outcome}")
            return
        df = df.dropna(subset=[target_col])
        
        # 2. Filter for Test Set
        if 'split_group' not in df.columns:
            logger.error("split_group column missing. Cannot perform safe evaluation.")
            return
            
        test_mask = df['split_group'] == 'test'
        df_test = df[test_mask].copy()
        
        if df_test.empty:
            logger.warning(f"No test data found for {outcome} in {branch} branch.")
            return

        # 3. Handle Missing Values (Same logic as utils.prepare_data)
        # We need to apply this to the features ONLY, but keep the dataframe intact
        feature_sets = utils.get_feature_sets(df)
        if feature_set not in feature_sets:
            logger.error(f"Invalid feature set: {feature_set}")
            return
            
        selected_features = feature_sets[feature_set]
        
        # Create X_test (features only)
        X_test = df_test[selected_features].copy()
        
        # Impute missing values
        for col in X_test.columns:
            if col in utils.CONTINUOUS_PREOP_COLS:
                X_test[col] = X_test[col].fillna(-99)
            else:
                X_test[col] = X_test[col].fillna(0)
                
        # 4. Generate Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 5. Construct Tidy Output
        output_df = pd.DataFrame({
            'caseid': df_test['caseid'],
            'y_true': df_test[target_col],
            'y_pred_proba': y_pred_proba,
            'split_group': 'test', # Explicitly labeling this
            'model_name': 'XGBoost',
            'feature_set': feature_set,
            'outcome': outcome,
            'branch': branch
        })
        
        # 6. Save
        output_path = model_path.parent / "predictions.csv"
        output_df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to process model at {model_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Generate predictions from trained models.")
    parser.add_argument("--outcome", type=str, help="Target outcome name (optional)")
    parser.add_argument("--branch", type=str, choices=['windowed', 'non_windowed'], help="Data branch (optional)")
    parser.add_argument("--feature_set", type=str, help="Feature set name (optional)")
    
    args = parser.parse_args()
    
    models_dir = utils.RESULTS_DIR / 'models'
    
    if args.outcome and args.branch and args.feature_set:
        # Single Model Mode
        model_path = models_dir / args.outcome / args.branch / args.feature_set / "model.json"
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return
        generate_predictions_for_model(model_path)
    else:
        # Batch Mode - Crawl directory
        logger.info(f"Crawling {models_dir} for models...")
        model_files = list(models_dir.glob('**/*/model.json'))
        
        if not model_files:
            logger.warning("No models found.")
            return
            
        logger.info(f"Found {len(model_files)} models. Processing...")
        
        for model_path in model_files:
            generate_predictions_for_model(model_path)

if __name__ == "__main__":
    main()
