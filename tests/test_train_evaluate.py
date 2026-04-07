import json
import logging
import shutil
import sys
import types
import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


class _DummyTreeExplainer:
    def __init__(self, model):
        self.model = model

    def shap_values(self, X):
        return np.zeros_like(X, dtype=float)


dummy_shap = types.SimpleNamespace(TreeExplainer=_DummyTreeExplainer, summary_plot=lambda *args, **kwargs: None)
sys.modules.setdefault("shap", dummy_shap)

import model_creation.utils as utils
import model_creation.step_06_run_hpo as hpo_module
import model_creation.step_07_train_evaluate as train_module
from model_creation.postprocessing import LogisticRecalibrationModel
from model_creation.preprocessing import FoldPreprocessor
from model_creation.validation import OuterFoldRunResult


class DummyModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        positive = np.full(len(X), 0.6)
        return np.column_stack([1 - positive, positive])

    def save_model(self, path):
        Path(path).write_text("dummy")


class DummyEbm(DummyModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.term_names_ = ["feature1", "feature2"]
        self.term_importances_ = [0.6, 0.4]
        self.term_features_ = [[0], [1]]

    def decision_function(self, X):
        return np.full(len(X), 0.5)

    def predict_proba(self, X):
        logits = self.decision_function(X)
        probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])

    def explain_global(self, name=None):
        return DummyExplanation({"names": self.term_names_, "scores": self.term_importances_})

    def explain_local(self, X, y):
        scores = [[0.2, 0.3] for _ in range(len(X))]
        return DummyExplanation({"names": self.term_names_, "scores": scores})

    def save(self, path):
        Path(path).write_text("dummy")


class DummyExplanation:
    def __init__(self, payload):
        self._payload = payload

    def data(self, *args, **kwargs):
        return self._payload


def _install_dummy_interpret(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_explain = types.SimpleNamespace(EBMExplanation=DummyExplanation)
    glassbox = types.SimpleNamespace(ExplainableBoostingClassifier=DummyEbm, _ebm=types.SimpleNamespace(_explain=dummy_explain))
    monkeypatch.setitem(sys.modules, "interpret", types.SimpleNamespace(glassbox=glassbox))
    monkeypatch.setitem(sys.modules, "interpret.glassbox", glassbox)
    monkeypatch.setitem(sys.modules, "interpret.glassbox._ebm", glassbox._ebm)
    monkeypatch.setitem(sys.modules, "interpret.glassbox._ebm._explain", dummy_explain)


def _mock_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "caseid": [1, 2, 3, 4, 5, 6],
            "subjectid": [101, 102, 103, 104, 105, 106],
            "feature1": [0.1, np.nan, 0.3, 0.4, 0.5, 0.6],
            "feature2": [1.0, 1.1, np.nan, 1.3, 1.4, 1.5],
            "aki_label": [0, 1, 0, 1, 0, 1],
        }
    )


def _prepare_data_stub(df: pd.DataFrame, preserve_nan: bool = True):
    X = df[["feature1", "feature2"]]
    X_train = X.iloc[:4]
    X_test = X.iloc[4:]
    y_train = df["aki_label"].iloc[:4]
    y_test = df["aki_label"].iloc[4:]
    scale_pos_weight = 1.0
    return X_train, X_test, y_train, y_test, scale_pos_weight


def _fake_generate_oof_predictions(X, y):
    preds = np.linspace(0.2, 0.8, len(y))
    folds = np.zeros(len(y), dtype=int)
    return preds, folds


def _fake_outer_result(df: pd.DataFrame, model_type: str, validation_scheme: str = "holdout") -> OuterFoldRunResult:
    X_train, X_test, y_train, y_test, _ = _prepare_data_stub(df)
    preprocessor = FoldPreprocessor(impute_missing=False).fit(X_train)
    transformed_test = preprocessor.transform(X_test)
    threshold = 0.5
    train_predictions = pd.DataFrame(
        {
            "caseid": df["caseid"].iloc[:4].values,
            "subjectid": df["subjectid"].iloc[:4].values,
            "y_true": y_train.values,
            "y_prob_raw": [0.2, 0.7, 0.3, 0.8],
            "y_prob_calibrated": [0.25, 0.75, 0.35, 0.85],
            "threshold": threshold,
            "y_pred_label": [0, 1, 0, 1],
            "fold": [0, 1, 0, 1],
            "outer_fold_id": [0, 0, 0, 0],
            "repeat_id": [0, 0, 0, 0],
            "validation_scheme": [validation_scheme] * 4,
            "is_oof": [True] * 4,
            "outcome": ["any_aki"] * 4,
            "branch": ["non_windowed"] * 4,
            "feature_set": ["preop_only"] * 4,
            "model_name": ["XGBoost" if model_type == "xgboost" else "EBM"] * 4,
            "pipeline": [f"step_07_train_evaluate_{validation_scheme}"] * 4,
        }
    )
    test_predictions = pd.DataFrame(
        {
            "caseid": df["caseid"].iloc[4:].values,
            "subjectid": df["subjectid"].iloc[4:].values,
            "y_true": y_test.values,
            "y_prob_raw": [0.6, 0.6],
            "y_prob_calibrated": [0.65, 0.65],
            "threshold": threshold,
            "y_pred_label": [1, 1],
            "fold": [0, 0],
            "outer_fold_id": [0, 0],
            "repeat_id": [0, 0],
            "validation_scheme": [validation_scheme] * 2,
            "is_oof": [validation_scheme == "nested_cv"] * 2,
            "outcome": ["any_aki"] * 2,
            "branch": ["non_windowed"] * 2,
            "feature_set": ["preop_only"] * 2,
            "model_name": ["XGBoost" if model_type == "xgboost" else "EBM"] * 2,
            "pipeline": [f"step_07_train_evaluate_{validation_scheme}"] * 2,
        }
    )
    final_model = DummyModel() if model_type == "xgboost" else DummyEbm()
    return OuterFoldRunResult(
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        best_params={"learning_rate": 0.1},
        best_score=0.5,
        calibration_model=LogisticRecalibrationModel(intercept=0.0, slope=1.0, eps=1e-15),
        threshold_payload={
            "threshold": threshold,
            "youden_j": 0.1,
            "sensitivity": 1.0,
            "specificity": 0.5,
        },
        metadata_payload={
            "actual_inner_folds": 2,
            "counts": {
                "train": {"total": 4, "positive": 2, "negative": 2},
                "test": {"total": 2, "positive": 1, "negative": 1},
            },
        },
        final_model=final_model,
        final_preprocessor=preprocessor,
        transformed_test_features=transformed_test,
        raw_test_logits=np.full(len(transformed_test), 0.5),
    )


@pytest.mark.parametrize("model_type,model_filename", [("xgboost", "model.json"), ("ebm", "model.pkl")])
def test_train_evaluate_creates_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_type: str, model_filename: str):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(train_module.utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "dummy-hash")

    df = _mock_dataframe()
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: df)
    monkeypatch.setattr(
        train_module,
        "run_outer_split",
        lambda **kwargs: _fake_outer_result(df, model_type=model_type, validation_scheme="holdout"),
    )

    if model_type == "ebm":
        _install_dummy_interpret(monkeypatch)

    train_module.train_evaluate(
        outcome="any_aki",
        branch="non_windowed",
        feature_set="preop_only",
        smoke_test=True,
        model_type=model_type,
        legacy_imputation=False,
        validation_scheme="holdout",
        inner_folds=2,
        n_trials=1,
    )

    output_dir = results_dir / "models" / model_type / "any_aki" / "non_windowed" / "preop_only"
    predictions_dir = output_dir / "predictions"
    artifacts_dir = output_dir / "artifacts"

    assert (predictions_dir / "train_oof.csv").is_file()
    assert (predictions_dir / "test.csv").is_file()
    assert (artifacts_dir / "calibration.json").is_file()
    assert (artifacts_dir / "threshold.json").is_file()
    assert (output_dir / model_filename).is_file()

    written_test = pd.read_csv(predictions_dir / "test.csv")
    assert "subjectid" in written_test.columns

    shutil.rmtree(results_dir, ignore_errors=True)


def test_export_ebm_local_attributions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(train_module.utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "dummy-hash")

    df = _mock_dataframe()
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: df)
    monkeypatch.setattr(
        train_module,
        "run_outer_split",
        lambda **kwargs: _fake_outer_result(df, model_type="ebm", validation_scheme="holdout"),
    )
    _install_dummy_interpret(monkeypatch)

    train_module.train_evaluate(
        outcome="any_aki",
        branch="non_windowed",
        feature_set="preop_only",
        smoke_test=False,
        model_type="ebm",
        legacy_imputation=False,
        export_ebm_explanations_flag=True,
        validation_scheme="holdout",
        inner_folds=2,
        n_trials=1,
    )

    xai_dir = (
        results_dir
        / "models"
        / "ebm"
        / "any_aki"
        / "non_windowed"
        / "preop_only"
        / "artifacts"
        / "ebm_xai"
    )

    attribution_path = xai_dir / "local_attributions.csv"
    readme_path = xai_dir / "README.md"

    assert attribution_path.is_file()
    recon_df = pd.read_csv(attribution_path)
    for required in ["caseid", "raw_logit", "calibrated_probability", "predicted_label"]:
        assert required in recon_df.columns
    assert not recon_df.empty
    assert readme_path.is_file()

    shutil.rmtree(results_dir, ignore_errors=True)


def test_plotly_export_warns_without_kaleido(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    class DummyFigure:
        def __init__(self, data=None):
            self.data = data

        def update_layout(self, **kwargs):
            return self

        def update_yaxes(self, **kwargs):
            return self

        def update_xaxes(self, **kwargs):
            return self

        def write_html(self, path):
            Path(path).write_text("html")

        def write_image(self, path):
            raise ValueError("Image export requires kaleido")

    go_module = types.SimpleNamespace(Bar=lambda **kwargs: {"bar": kwargs}, Figure=DummyFigure)
    plotly_module = types.SimpleNamespace(graph_objects=go_module)
    monkeypatch.setitem(sys.modules, "plotly", plotly_module)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go_module)

    destination = tmp_path / "plotly_importances"

    with caplog.at_level(logging.WARNING):
        train_module._export_plotly_bar(
            ["feature1", "feature2"], [0.2, 0.8], destination, "EBM Term Importances"
        )

    assert (tmp_path / "plotly_importances.html").is_file()
    assert not (tmp_path / "plotly_importances.png").exists()
    assert "Kaleido is unavailable" in caplog.text


def test_plotly_export_skips_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "plotly.graph_objects" or name == "plotly":
            raise ImportError("plotly unavailable in test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    destination = tmp_path / "missing_plotly"

    with caplog.at_level(logging.WARNING):
        train_module._export_plotly_bar(["feature1"], [0.5], destination, "Missing Plotly")

    assert not (tmp_path / "missing_plotly.html").exists()
    assert "Plotly is not installed" in caplog.text


class FakeStudy:
    def __init__(self):
        self.best_params = {"learning_rate": 0.1}
        self.best_value = 0.5

    def optimize(self, func, n_trials, show_progress_bar, n_jobs=None):
        return None


@pytest.mark.parametrize("model_type", ["xgboost", "ebm"])
def test_run_hpo_smoke_saves_params(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_type: str):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_module.utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(hpo_module.utils, "RESULTS_DIR", results_dir)

    df = _mock_dataframe()
    monkeypatch.setattr(hpo_module.utils, "load_data", lambda branch: df)
    monkeypatch.setattr(
        hpo_module,
        "tune_model",
        lambda *args, **kwargs: ({"learning_rate": 0.1}, 0.5, 2),
    )

    hpo_module.run_hpo(
        outcome="any_aki",
        branch="non_windowed",
        feature_set="preop_only",
        smoke_test=True,
        model_type=model_type,
    )

    params_path = results_dir / "params" / model_type / "any_aki" / "non_windowed" / "preop_only.json"
    assert params_path.is_file()

    with params_path.open() as fp:
        saved_params = json.load(fp)
    assert "learning_rate" in saved_params
    assert "scale_pos_weight" in saved_params

    shutil.rmtree(results_dir, ignore_errors=True)


def test_run_hpo_invalid_feature_set_exits_nonzero(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(hpo_module.utils, "load_data", lambda branch: _mock_dataframe())

    with pytest.raises(SystemExit) as excinfo:
        hpo_module.run_hpo(
            outcome="any_aki",
            branch="non_windowed",
            feature_set="does_not_exist",
            smoke_test=True,
            model_type="xgboost",
        )

    assert excinfo.value.code == 1


def test_train_evaluate_holdout_save_final_refit_writes_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(train_module.utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "dummy-hash")

    df = _mock_dataframe()
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: df)
    monkeypatch.setattr(
        train_module,
        "run_outer_split",
        lambda **kwargs: _fake_outer_result(df, model_type="xgboost", validation_scheme="holdout"),
    )
    monkeypatch.setattr(
        train_module,
        "tune_model",
        lambda *args, **kwargs: ({"learning_rate": 0.1, "max_bins": 64}, 0.5, 2),
    )
    monkeypatch.setattr(
        train_module,
        "generate_cross_fitted_predictions",
        lambda X, y, groups, **kwargs: (np.linspace(0.2, 0.8, len(y)), np.zeros(len(y), dtype=int), 2),
    )

    def _fake_fit_final_model(X_train, y_train, **kwargs):
        preprocessor = FoldPreprocessor(impute_missing=False).fit(X_train)
        transformed = preprocessor.transform(X_train)
        return preprocessor, DummyModel(), transformed

    monkeypatch.setattr(train_module, "fit_final_model", _fake_fit_final_model)

    train_module.train_evaluate(
        outcome="any_aki",
        branch="non_windowed",
        feature_set="preop_only",
        smoke_test=True,
        model_type="xgboost",
        validation_scheme="holdout",
        inner_folds=2,
        n_trials=1,
        save_final_refit=True,
    )

    output_dir = results_dir / "models" / "xgboost" / "any_aki" / "non_windowed" / "preop_only"
    final_refit_dir = output_dir / "artifacts" / "final_refit"

    assert (output_dir / "predictions" / "test.csv").is_file()
    assert (final_refit_dir / "train_oof.csv").is_file()
    assert (final_refit_dir / "calibration.json").is_file()
    assert (final_refit_dir / "threshold.json").is_file()
    assert (final_refit_dir / "metadata.json").is_file()
    assert (final_refit_dir / "model.json").is_file()

    validation_manifest = json.loads((output_dir / "artifacts" / "validation.json").read_text())
    assert validation_manifest["final_refit_saved"] is True

    shutil.rmtree(results_dir, ignore_errors=True)
