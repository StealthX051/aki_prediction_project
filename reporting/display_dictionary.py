"""Utilities for mapping internal identifiers to publication-ready labels.

The display dictionary is a JSON document (``metadata/display_dictionary.json``
by default) that defines human-readable labels for outcomes, branches, model
families, feature sets, and feature names. This module centralizes loading,
lookups, and formatting so downstream scripts can render consistent text across
figures, tables, and interpretability outputs.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_DICTIONARY_PATH = Path(__file__).resolve().parent.parent / "metadata" / "display_dictionary.json"


@dataclass(frozen=True)
class DisplayEntry:
    """Container for a single display mapping."""

    label: str
    short_label: Optional[str] = None
    unit: Optional[str] = None
    description: Optional[str] = None

    def format(self, *, use_short: bool = False, include_unit: bool = False) -> str:
        """Return a formatted label optionally using a short name and units."""

        text = self.short_label if use_short and self.short_label else self.label
        if include_unit and self.unit:
            return f"{text} ({self.unit})"
        return text


class DisplayDictionary:
    """Load and resolve human-readable labels for identifiers."""

    def __init__(self, raw: Dict[str, Any], source: Path):
        self.raw = raw
        self.source = source
        self.schema_version = raw.get("schema_version", 1)

        self.outcomes = raw.get("outcomes", {})
        self.branches = raw.get("branches", {})
        self.model_types = raw.get("model_types", {})
        self.feature_sets = raw.get("feature_sets", {})
        self.waveforms = raw.get("waveforms", {})
        self.catch22_statistics = raw.get("catch22_statistics", {})
        self.catch22_aggregates = raw.get(
            "catch22_aggregates",
            {"mean": "Mean", "std": "Std Dev", "min": "Min", "max": "Max"},
        )
        self.features = raw.get("features", {})

        self._waveform_aliases = self._build_waveform_aliases(self.waveforms)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DisplayDictionary":
        """Load the display dictionary JSON file.

        The path can be provided explicitly or via the ``DISPLAY_DICTIONARY_PATH``
        environment variable. If neither is set, the default
        ``metadata/display_dictionary.json`` path is used.
        """

        env_override = os.getenv("DISPLAY_DICTIONARY_PATH")
        resolved_path = Path(path or env_override or DEFAULT_DICTIONARY_PATH)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Display dictionary not found at {resolved_path}")

        with resolved_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        return cls(raw=raw, source=resolved_path)

    def _to_entry(self, section: Dict[str, Any], key: str) -> Optional[DisplayEntry]:
        value = section.get(key)
        if value is None:
            return None

        if isinstance(value, str):
            return DisplayEntry(label=value)

        if isinstance(value, dict):
            return DisplayEntry(
                label=value.get("label", key),
                short_label=value.get("short_label"),
                unit=value.get("unit"),
                description=value.get("description"),
            )

        logger.debug("Unhandled display entry type for %s: %s", key, type(value))
        return None

    @staticmethod
    def _build_waveform_aliases(waveforms: Dict[str, Any]) -> Dict[str, Any]:
        aliases: Dict[str, Any] = {}
        for key, value in waveforms.items():
            aliases[key] = value
            aliases[key.replace("/", "_")] = value
        return aliases

    def _parse_catch22_feature(self, feature_name: str) -> Optional[tuple[str, str, Optional[str]]]:
        aggregate = None
        base = feature_name

        for agg_key in self.catch22_aggregates:
            suffix = f"_{agg_key}"
            if feature_name.endswith(suffix):
                aggregate = agg_key
                base = feature_name[: -len(suffix)]
                break

        for stat_key in self.catch22_statistics:
            suffix = f"_{stat_key}"
            if base.endswith(suffix):
                waveform_key = base[: -len(suffix)]
                if waveform_key:
                    return waveform_key, stat_key, aggregate
        return None

    def _lookup_waveform(self, waveform_key: str) -> Optional[DisplayEntry]:
        entry = self._waveform_aliases.get(waveform_key)
        if entry:
            if isinstance(entry, dict):
                return self._to_entry(self._waveform_aliases, waveform_key)
            return DisplayEntry(label=str(entry))
        return None

    def feature_entry(self, feature_name: str) -> DisplayEntry:
        """Return a :class:`DisplayEntry` for a feature name.

        Resolution order:
          1. Direct match in the ``features`` section.
          2. Catch22 waveform features that follow ``<waveform>_<stat>[_aggregate]``.
          3. Fallback to the raw feature name.
        """

        direct_entry = self._to_entry(self.features, feature_name)
        if direct_entry:
            return direct_entry

        parsed = self._parse_catch22_feature(feature_name)
        if parsed:
            waveform_key, stat_key, aggregate = parsed
            waveform_entry = self._lookup_waveform(waveform_key)
            stat_entry = self._to_entry(self.catch22_statistics, stat_key)

            waveform_label = waveform_entry.format(use_short=True) if waveform_entry else waveform_key
            stat_label = stat_entry.label if stat_entry else stat_key

            label = f"{waveform_label} — {stat_label}"
            if aggregate:
                aggregate_label = self.catch22_aggregates.get(aggregate, aggregate)
                label = f"{label} ({aggregate_label})"

            unit = waveform_entry.unit if waveform_entry else None
            description = stat_entry.description if stat_entry else None
            return DisplayEntry(label=label, unit=unit, description=description)

        return DisplayEntry(label=feature_name)

    def feature_label(self, feature_name: str, *, use_short: bool = False, include_unit: bool = False) -> str:
        """Return the formatted feature label."""

        return self.feature_entry(feature_name).format(use_short=use_short, include_unit=include_unit)

    def _section_label(self, section: Dict[str, Any], key: str, *, use_short: bool = False) -> str:
        entry = self._to_entry(section, key)
        if entry:
            return entry.format(use_short=use_short)
        return key

    def outcome_label(self, outcome_key: str, *, use_short: bool = False) -> str:
        return self._section_label(self.outcomes, outcome_key, use_short=use_short)

    def branch_label(self, branch_key: str, *, use_short: bool = False) -> str:
        return self._section_label(self.branches, branch_key, use_short=use_short)

    def model_type_label(self, model_key: str, *, use_short: bool = False) -> str:
        return self._section_label(self.model_types, model_key, use_short=use_short)

    def feature_set_labels(self, fallback: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Return mapping of feature set key to label, with optional fallbacks."""

        labels = {k: self._section_label(self.feature_sets, k) for k in self.feature_sets}
        if fallback:
            for key, value in fallback.items():
                labels.setdefault(key, value)
        return labels


def load_display_dictionary(path: Optional[Path] = None) -> DisplayDictionary:
    """Convenience loader for the project display dictionary."""

    return DisplayDictionary.load(path=path)
