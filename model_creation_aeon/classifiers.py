import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import logging

try:
    # Aeon v0.26.0+ / v1.0.0+ unified classes
    from aeon.transformations.collection.convolution_based import MultiRocket, MiniRocket
    from aeon.transformations.collection.feature_based import TSFresh
    from aeon.classification.sklearn import RotationForestClassifier
except ImportError as e:
    logging.warning(f"Aeon/TSFresh import failed: {e}. Models will not work.")

class FusedClassifier(BaseEstimator, ClassifierMixin):
    """
    Base class for Early Fusion models (Waveform + Preop).
    
    API:
        fit(X_wave, X_preop, y)
        predict(X_wave, X_preop)
        predict_proba(X_wave, X_preop)
    
    If X_preop is None, it acts as a pure Time Series Classifier.
    """
    def __init__(self):
        pass

    def _validate_inputs(self, X_wave, X_preop):
        if X_preop is not None:
            if X_wave.shape[0] != X_preop.shape[0]:
                raise ValueError(f"Mismatch in n_samples: Wave={X_wave.shape[0]}, Preop={X_preop.shape[0]}")
        return X_wave, X_preop

class RocketFused(FusedClassifier):
    def __init__(self, variant='multi', n_kernels=10000, random_state=42, n_jobs=-1, estimator=None):
        self.variant = variant
        self.n_kernels = n_kernels # Aeon 1.1 uses n_kernels
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimator = estimator 
        
        self.transformer = None
        self.scaler = None
        self.classifier = None


    def fit(self, X_wave, X_preop, y):
        X_wave, X_preop = self._validate_inputs(X_wave, X_preop)
        
        # 1. Transform Waveforms
        if self.variant == 'multi':
            self.transformer = MultiRocket(
                n_kernels=self.n_kernels, 
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif self.variant == 'mini':
            self.transformer = MiniRocket(
                n_kernels=self.n_kernels, 
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
            
        logging.info(f"[{self.variant}Rocket] Transforming waveforms shape {X_wave.shape}...")
        X_rocket = self.transformer.fit_transform(X_wave)
        logging.info(f"[{self.variant}Rocket] Output shape: {X_rocket.shape}")
        
        # 2. Concatenate
        if X_preop is not None:
            # Ensure preop is numeric (should be handled by step_04, but safety first)
            X_preop_np = np.array(X_preop, dtype=np.float32)
            X_combined = np.hstack([X_rocket, X_preop_np])
            logging.info(f"Fused with Preop. Final shape: {X_combined.shape}")
        else:
            X_combined = X_rocket
            logging.info("No Preop provided. Using purely Rocket features.")

        # 3. Scale (Global)
        self.scaler = StandardScaler()
        logging.info("Scaling features with StandardScaler...")
        X_scaled = self.scaler.fit_transform(X_combined)
        
        # 4. Fit Linear Head
        logging.info(f"[{self.variant}Rocket] Fitting linear head...")
        
        if self.estimator is None:
             # Default to RidgeClassifierCV if not provided
             self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight='balanced')
        else:
             # Clone would be better practice but direct assignment works for this script
             from sklearn.base import clone
             self.classifier = clone(self.estimator)

        self.classifier.fit(X_scaled, y)
        logging.info(f"[{self.variant}Rocket] Linear head fit complete.")
            
        return self

    # Old fit method removed.


    def predict_proba(self, X_wave, X_preop):
        # 1. Transform
        X_rocket = self.transformer.transform(X_wave)
        
        # 2. Concatenate
        if X_preop is not None:
            X_preop_np = np.array(X_preop, dtype=np.float32)
            X_combined = np.hstack([X_rocket, X_preop_np])
        else:
            X_combined = X_rocket
            
        # 3. Scale
        X_scaled = self.scaler.transform(X_combined)
        
        # 4. Predict
        if isinstance(self.classifier, RidgeClassifierCV):
            # RidgeClassifierCV lacks predict_proba, uses decision_function
            d = self.classifier.decision_function(X_scaled)
            # Sigmoid calibration (naive)
            # Or use CalibratedClassifierCV wrapper? 
            # Ideally step_07 handles calibration.
            # But the user asked for predict_proba output.
            # We will use the decision function -> sigmoid for raw 'probability-like' score
            # Note: Ridge scores are not probs. But standard practice in ROCKET papers is using Ridge Classifier.
            # If the user wants probabilities for AUPRC/Calibration, LogisticRegression is safer.
            # Or we sigmoid the decision function manually.
            try:
                probs = 1 / (1 + np.exp(-d))
            except RuntimeWarning:
                probs = d # fallback
            
            # Helper to return (N, 2)
            return np.column_stack([1-probs, probs])
            
        return self.classifier.predict_proba(X_scaled)

    def predict(self, X_wave, X_preop):
        probs = self.predict_proba(X_wave, X_preop)[:, 1]
        return (probs >= 0.5).astype(int)

class FreshPrinceFused(FusedClassifier):
    def __init__(self, n_estimators=200, random_state=42, n_jobs=-1, fc_parameters="comprehensive"):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.fc_parameters = fc_parameters
        
        self.transformer = None
        self.imputer = None # TSFresh can produce NaNs
        self.scaler = None
        self.classifier = None
        
    def fit(self, X_wave, X_preop, y):
        X_wave, X_preop = self._validate_inputs(X_wave, X_preop)
        
        # 1. Transform Waveforms (TSFresh)
        # Note: TSFresh expects specific formats, but Aeon wrapper handles numpy 3D?
        # Aeon TSFresh wrapper takes (n_cases, n_channels, n_timepoints)
        self.transformer = TSFresh(
            default_fc_parameters=self.fc_parameters,
            n_jobs=self.n_jobs,
            show_warnings=False,
            # chunksize might not be in v1.1 TSFresh signature, let's remove it if unsafe or check docs. 
            # safely removing chunksize as it's often auto-handled or not exposed in newer wrapper
        )
        
        logging.info(f"[FreshPrince] Extracting features ({self.fc_parameters})... This may take a while.")
        # TSFresh is slow.
        X_tsfresh = self.transformer.fit_transform(X_wave)
        logging.info(f"[FreshPrince] Output shape: {X_tsfresh.shape}")
        
        # TSFresh often creates NaNs (e.g. if length is small)
        self.imputer = SimpleImputer(strategy='median')
        X_tsfresh = self.imputer.fit_transform(X_tsfresh)
        
        # 2. Concatenate
        if X_preop is not None:
             X_preop_np = np.array(X_preop, dtype=np.float32)
             X_combined = np.hstack([X_tsfresh, X_preop_np])
        else:
             X_combined = X_tsfresh

        # 3. Scale (Global)
        # Important for PCA in RotationForest, especially with mixed feature types (Wave+Preop)
        self.scaler = StandardScaler()
        logging.info("[FreshPrince] Scaling features...")
        X_scaled = self.scaler.fit_transform(X_combined)

        # 4. Fit Rotation Forest
        # RotationForest is an ensemble of trees on PCA-transformed subsets
        self.classifier = RotationForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.classifier.fit(X_scaled, y)
        
        return self

    def predict_proba(self, X_wave, X_preop):
        X_tsfresh = self.transformer.transform(X_wave)
        X_tsfresh = self.imputer.transform(X_tsfresh)
        
        if X_preop is not None:
            X_preop_np = np.array(X_preop, dtype=np.float32)
            X_combined = np.hstack([X_tsfresh, X_preop_np])
        else:
            X_combined = X_tsfresh
            
        X_scaled = self.scaler.transform(X_combined)
        return self.classifier.predict_proba(X_scaled)
