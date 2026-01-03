import json
import logging
import shutil
import sys
import types
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

    def data(self):
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
            "caseid": [1, 2, 3, 4],
            "feature1": [0.1, np.nan, 0.3, 0.4],
            "feature2": [1.0, 1.1, np.nan, 1.3],
            "aki_label": [0, 1, 0, 1],
        }
    )


def _prepare_data_stub(df: pd.DataFrame, preserve_nan: bool = True):
    X = df[["feature1", "feature2"]]
    X_train = X.iloc[:3]
    X_test = X.iloc[3:]
    y_train = df["aki_label"].iloc[:3]
    y_test = df["aki_label"].iloc[3:]
    scale_pos_weight = 1.0
    return X_train, X_test, y_train, y_test, scale_pos_weight


def _fake_generate_oof_predictions(X, y):
    preds = np.linspace(0.2, 0.8, len(y))
    folds = np.zeros(len(y), dtype=int)
    return preds, folds


@pytest.mark.parametrize("model_type,model_filename", [("xgboost", "model.json"), ("ebm", "model.ebm")])
def test_train_evaluate_creates_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_type: str, model_filename: str):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(train_module.utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "dummy-hash")

    df = _mock_dataframe()
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: df)

    def _prepare(df_arg, outcome, feature_set, random_state=42, preserve_nan=True):
        assert preserve_nan is True
        return _prepare_data_stub(df_arg, preserve_nan)

    monkeypatch.setattr(train_module.utils, "prepare_data", _prepare)
    monkeypatch.setattr(
        train_module,
        "generate_stratified_oof_predictions",
        lambda *args, **kwargs: _fake_generate_oof_predictions(args[1], args[2]),
    )

    monkeypatch.setattr(train_module.xgb, "XGBClassifier", DummyModel)

    if model_type == "ebm":
        _install_dummy_interpret(monkeypatch)

    params_dir = results_dir / "params" / model_type / "any_aki" / "non_windowed"
    params_dir.mkdir(parents=True)
    params_path = params_dir / "preop_only.json"
    params_path.write_text(json.dumps({"random_state": 0, "n_estimators": 5}))

    train_module.train_evaluate(
        outcome="any_aki",
        branch="non_windowed",
        feature_set="preop_only",
        smoke_test=True,
        model_type=model_type,
        legacy_imputation=False,
    )

    output_dir = results_dir / "models" / model_type / "any_aki" / "non_windowed" / "preop_only"
    predictions_dir = output_dir / "predictions"
    artifacts_dir = output_dir / "artifacts"

    assert (predictions_dir / "train_oof.csv").is_file()
    assert (predictions_dir / "test.csv").is_file()
    assert (artifacts_dir / "calibration.json").is_file()
    assert (artifacts_dir / "threshold.json").is_file()
    assert (output_dir / model_filename).is_file()

    shutil.rmtree(results_dir, ignore_errors=True)


def test_export_ebm_local_attributions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(train_module.utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "dummy-hash")

    df = _mock_dataframe()
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: df)
    monkeypatch.setattr(train_module.utils, "prepare_data", lambda *args, **kwargs: _prepare_data_stub(df))
    monkeypatch.setattr(
        train_module,
        "generate_stratified_oof_predictions",
        lambda *args, **kwargs: _fake_generate_oof_predictions(args[1], args[2]),
    )
    _install_dummy_interpret(monkeypatch)

    params_dir = results_dir / "params" / "ebm" / "any_aki" / "non_windowed"
    params_dir.mkdir(parents=True)
    params_path = params_dir / "preop_only.json"
    params_path.write_text(json.dumps({"random_state": 0, "n_estimators": 5}))

    train_module.train_evaluate(
        outcome="any_aki",
        branch="non_windowed",
        feature_set="preop_only",
        smoke_test=False,
        model_type="ebm",
        legacy_imputation=False,
        export_ebm_explanations_flag=True,
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
    monkeypatch.delitem(sys.modules, "plotly", raising=False)
    monkeypatch.delitem(sys.modules, "plotly.graph_objects", raising=False)

    destination = tmp_path / "missing_plotly"

    with caplog.at_level(logging.WARNING):
        train_module._export_plotly_bar(["feature1"], [0.5], destination, "Missing Plotly")

    assert not (tmp_path / "missing_plotly.html").exists()
    assert "Plotly is not installed" in caplog.text


class FakeStudy:
    def __init__(self):
        self.best_params = {"learning_rate": 0.1}
        self.best_value = 0.5

    def optimize(self, func, n_trials, show_progress_bar):
        return None


@pytest.mark.parametrize("model_type", ["xgboost", "ebm"])
def test_run_hpo_smoke_saves_params(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_type: str):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_module.utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(hpo_module.utils, "RESULTS_DIR", results_dir)

    df = _mock_dataframe()
    monkeypatch.setattr(utils, "load_data", lambda branch: df)
    monkeypatch.setattr(utils, "prepare_data", lambda *args, **kwargs: _prepare_data_stub(df))
    monkeypatch.setattr(train_module.xgb, "XGBClassifier", DummyModel)
    if model_type == "ebm":
        _install_dummy_interpret(monkeypatch)

    monkeypatch.setattr(hpo_module.optuna, "create_study", lambda direction: FakeStudy())

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
