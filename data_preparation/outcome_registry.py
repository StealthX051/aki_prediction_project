from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutcomeSpec:
    name: str
    target_col: str
    eligibility_col: str
    split_col: str | None
    cohort_profile: str
    trainable: bool = True


LEGACY_DEFAULT_OUTCOME_NAME = "any_aki"
LEGACY_SPLIT_ALIAS = "split_group"

OUTCOME_SPECS = {
    "any_aki": OutcomeSpec(
        name="any_aki",
        target_col="aki_label",
        eligibility_col="eligible_any_aki",
        split_col="split_group_any_aki",
        cohort_profile="aki",
        trainable=True,
    ),
    "severe_aki": OutcomeSpec(
        name="severe_aki",
        target_col="y_severe_aki",
        eligibility_col="eligible_severe_aki",
        split_col="split_group_severe_aki",
        cohort_profile="aki",
        trainable=True,
    ),
    "mortality": OutcomeSpec(
        name="mortality",
        target_col="y_inhosp_mortality",
        eligibility_col="eligible_mortality",
        split_col="split_group_mortality",
        cohort_profile="non_aki_default",
        trainable=True,
    ),
    "icu_admission": OutcomeSpec(
        name="icu_admission",
        target_col="y_icu_admit",
        eligibility_col="eligible_icu_admission",
        split_col="split_group_icu_admission",
        cohort_profile="non_aki_default",
        trainable=True,
    ),
    "extended_los": OutcomeSpec(
        name="extended_los",
        target_col="y_prolonged_los_postop",
        eligibility_col="eligible_extended_los",
        split_col=None,
        cohort_profile="archival",
        trainable=False,
    ),
}

TRAINABLE_OUTCOME_SPECS = tuple(
    spec for spec in OUTCOME_SPECS.values() if spec.trainable
)
TRAINABLE_OUTCOME_NAMES = tuple(spec.name for spec in TRAINABLE_OUTCOME_SPECS)
ALL_OUTCOME_COLUMNS = tuple(dict.fromkeys(spec.target_col for spec in OUTCOME_SPECS.values()))
ALL_ELIGIBILITY_COLUMNS = tuple(
    dict.fromkeys(spec.eligibility_col for spec in OUTCOME_SPECS.values())
)
ALL_SPLIT_COLUMNS = tuple(
    dict.fromkeys(spec.split_col for spec in OUTCOME_SPECS.values() if spec.split_col)
)
DEFAULT_OUTCOME_COLUMN = OUTCOME_SPECS[LEGACY_DEFAULT_OUTCOME_NAME].target_col
DEFAULT_OUTCOME_SPEC = OUTCOME_SPECS[LEGACY_DEFAULT_OUTCOME_NAME]


def get_outcome_spec(outcome_name: str) -> OutcomeSpec:
    try:
        return OUTCOME_SPECS[outcome_name]
    except KeyError as exc:
        raise ValueError(
            f"Invalid outcome: {outcome_name}. Available: {list(OUTCOME_SPECS.keys())}"
        ) from exc


def get_trainable_outcome_spec(outcome_name: str) -> OutcomeSpec:
    spec = get_outcome_spec(outcome_name)
    if not spec.trainable:
        raise ValueError(
            f"Outcome '{outcome_name}' is archival only and is not supported for training."
        )
    return spec
