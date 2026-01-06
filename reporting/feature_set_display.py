"""Helpers for rendering feature set inputs as checkbox grids or labels.

This module centralizes:
* Loading canonical feature components (in a fixed order) and mappings from
  feature set keys to those components.
* Rendering helpers for HTML (checkbox grid) and plain text (ASCII checkboxes).
* Label resolution that honors the display dictionary when present.

The checkbox rendering is the default mode for tables; switching back to text
labels is controlled by the FEATURE_SET_DISPLAY environment variable.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - fallback when PyYAML is unavailable
    yaml = None

from reporting.display_dictionary import DisplayDictionary, load_display_dictionary

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPONENTS_PATH = PROJECT_ROOT / "metadata" / "feature_components.yml"


@dataclass(frozen=True)
class FeatureComponent:
    """Single feature component in the fixed display order."""

    key: str
    label: str


@dataclass
class FeatureComponentsConfig:
    """Configuration container for components and mappings."""

    components: List[FeatureComponent]
    feature_set_map: Dict[str, List[str]]

    @property
    def component_keys(self) -> List[str]:
        return [c.key for c in self.components]


DEFAULT_COMPONENTS: List[FeatureComponent] = [
    FeatureComponent(key="preop", label="Preop"),
    FeatureComponent(key="awp", label="AWP"),
    FeatureComponent(key="co2", label="CO2"),
    FeatureComponent(key="ecg", label="ECG"),
    FeatureComponent(key="pleth", label="Pleth"),
]

DEFAULT_FEATURE_SET_MAP: Dict[str, List[str]] = {
    # Baseline
    "preop_only": ["preop"],
    "awp_only": ["awp"],
    "co2_only": ["co2"],
    "ecg_only": ["ecg"],
    "pleth_only": ["pleth"],
    # Combined sets
    "all_waveforms": ["awp", "co2", "ecg", "pleth"],
    "preop_and_all_waveforms": ["preop", "awp", "co2", "ecg", "pleth"],
    "preop_and_awp": ["preop", "awp"],
    "preop_and_co2": ["preop", "co2"],
    "preop_and_ecg": ["preop", "ecg"],
    "preop_and_pleth": ["preop", "pleth"],
    "preop_and_all_minus_awp": ["preop", "co2", "ecg", "pleth"],
    "preop_and_all_minus_co2": ["preop", "awp", "ecg", "pleth"],
    "preop_and_all_minus_ecg": ["preop", "awp", "co2", "pleth"],
    "preop_and_all_minus_pleth": ["preop", "awp", "co2", "ecg"],
    # Aeon / fused variants
    "all_fused": ["awp", "co2", "ecg", "pleth"],
    "all_waveonly": ["awp", "co2", "ecg", "pleth"],
}


class FeatureSetDisplayMode:
    """Allowed display modes for feature sets."""

    CHECKBOX = "checkbox"
    LABEL = "label"
    BOTH = "both"

    @classmethod
    def from_env(cls) -> str:
        raw = os.getenv("FEATURE_SET_DISPLAY", cls.CHECKBOX).strip().lower()
        if raw in {cls.CHECKBOX, cls.LABEL, cls.BOTH}:
            return raw
        logger.warning("Unknown FEATURE_SET_DISPLAY=%s; defaulting to '%s'", raw, cls.CHECKBOX)
        return cls.CHECKBOX


def _load_components_config(path: Path) -> FeatureComponentsConfig:
    """Load components + mapping from YAML, falling back to defaults."""

    if yaml is None:
        logger.warning("PyYAML not installed; using default feature component mappings.")
        return FeatureComponentsConfig(components=DEFAULT_COMPONENTS, feature_set_map=DEFAULT_FEATURE_SET_MAP)

    if not path.exists():
        logger.info("Feature components file not found at %s; using defaults.", path)
        return FeatureComponentsConfig(components=DEFAULT_COMPONENTS, feature_set_map=DEFAULT_FEATURE_SET_MAP)

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    components_raw: Sequence[Dict[str, str]] = loaded.get("components", [])
    feature_sets_raw: Dict[str, List[str]] = loaded.get("feature_sets", {})

    components = (
        [FeatureComponent(**c) for c in components_raw]
        if components_raw
        else DEFAULT_COMPONENTS
    )
    feature_map = feature_sets_raw or DEFAULT_FEATURE_SET_MAP

    return FeatureComponentsConfig(components=components, feature_set_map=feature_map)


class FeatureSetDisplay:
    """Renderer that converts feature set keys into checkbox grids or labels."""

    def __init__(
        self,
        config: FeatureComponentsConfig,
        *,
        display_dictionary: Optional[DisplayDictionary] = None,
        default_label_map: Optional[Dict[str, str]] = None,
        mode: Optional[str] = None,
    ):
        self.config = config
        self.display_dictionary = display_dictionary
        self.default_label_map = default_label_map or {}
        self.mode = mode or FeatureSetDisplayMode.from_env()

        # Precompute convenience sets
        self._component_keys = self.config.component_keys
        self._waveform_components = [k for k in self._component_keys if k != "preop"]
        self.label_map = self._build_label_map()
        self.component_labels = [comp.label for comp in self.config.components]

    @classmethod
    def from_defaults(
        cls,
        *,
        display_dictionary: Optional[DisplayDictionary] = None,
        default_label_map: Optional[Dict[str, str]] = None,
    ) -> "FeatureSetDisplay":
        """Factory using default paths and env-driven mode."""

        dd = display_dictionary
        if dd is None:
            try:
                dd = load_display_dictionary()
            except FileNotFoundError:
                dd = None

        config = _load_components_config(DEFAULT_COMPONENTS_PATH)
        return cls(config=config, display_dictionary=dd, default_label_map=default_label_map, mode=None)

    def _build_label_map(self) -> Dict[str, str]:
        if self.display_dictionary:
            labels = self.display_dictionary.feature_set_labels(self.default_label_map)
            return labels
        if self.default_label_map:
            return dict(self.default_label_map)
        return {k: k for k in self.config.feature_set_map}

    def as_label(self, feature_set: str) -> str:
        """Return a human-readable label for a feature set key."""

        return self.label_map.get(feature_set, feature_set)

    def _resolve_from_map(self, feature_set: str) -> Optional[List[str]]:
        return self.config.feature_set_map.get(feature_set)

    def _resolve_from_patterns(self, feature_set: str) -> Optional[List[str]]:
        """Infer component membership from naming patterns."""

        fs = feature_set
        if fs in {"all_waveforms", "all_fused", "all_waveonly"}:
            return list(self._waveform_components)

        if fs == "preop_and_all_waveforms":
            return ["preop", *self._waveform_components]

        minus_prefix = "preop_and_all_minus_"
        if fs.startswith(minus_prefix):
            missing = fs[len(minus_prefix) :]
            comps = ["preop", *self._waveform_components]
            if missing in comps:
                return [c for c in comps if c != missing]

        preop_prefix = "preop_and_"
        if fs.startswith(preop_prefix):
            remainder = fs[len(preop_prefix) :]
            if remainder in self._component_keys:
                return ["preop", remainder]

        if fs.endswith("_only"):
            base = fs[: -len("_only")]
            if base in self._component_keys:
                return [base]

        return None

    def resolve_components(self, feature_set: str) -> Tuple[Dict[str, bool], bool]:
        """Return (component_flags, is_known)."""

        flags = {key: False for key in self._component_keys}

        comp_list = self._resolve_from_map(feature_set) or self._resolve_from_patterns(feature_set)
        if comp_list is None:
            return flags, False

        for comp in comp_list:
            if comp in flags:
                flags[comp] = True
        return flags, True

    def component_flags(self, feature_set: str) -> Tuple[Dict[str, bool], bool]:
        """Public helper to retrieve component membership."""

        return self.resolve_components(feature_set)

    def as_checkbox_html(self, feature_set: str) -> str:
        """Render a compact HTML grid with checkmarks."""

        flags, known = self.resolve_components(feature_set)
        if not known:
            return f"<div class=\"inputs-grid-unknown\">{self.as_label(feature_set)}</div>"

        header_cells = "".join(f"<th>{comp.label}</th>" for comp in self.config.components)
        body_cells = ""
        for comp in self.config.components:
            checked = flags.get(comp.key, False)
            mark = "&#10003;" if checked else ""
            cell_class = "checked" if checked else "unchecked"
            body_cells += f"<td class=\"{cell_class}\">{mark}</td>"

        return (
            "<table class=\"inputs-grid\">"
            f"<thead><tr>{header_cells}</tr></thead>"
            f"<tbody><tr>{body_cells}</tr></tbody>"
            "</table>"
        )

    def as_checkbox_text(self, feature_set: str) -> str:
        """Plain-text checkbox rendering suitable for DOCX/PDF."""

        flags, known = self.resolve_components(feature_set)
        if not known:
            return self.as_label(feature_set)

        parts = []
        for comp in self.config.components:
            tick = "[x]" if flags.get(comp.key, False) else "[ ]"
            parts.append(f"{comp.label} {tick}")
        return " | ".join(parts)

    @property
    def show_checkbox(self) -> bool:
        return self.mode in {FeatureSetDisplayMode.CHECKBOX, FeatureSetDisplayMode.BOTH}

    @property
    def show_label(self) -> bool:
        return self.mode in {FeatureSetDisplayMode.LABEL, FeatureSetDisplayMode.BOTH}
