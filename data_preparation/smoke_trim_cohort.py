from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_preparation.outcome_registry import get_trainable_outcome_spec

DEFAULT_SMOKE_OUTCOMES = ("any_aki", "icu_admission")
MIN_PER_CLASS_BY_OUTCOME = {
    "any_aki": 4,
}


def _unique_subject_sample(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if "subjectid" not in df.columns:
        return df.head(limit).copy()

    unique_subject_rows = df.drop_duplicates(subset=["subjectid"])
    return unique_subject_rows.head(limit).copy()


def trim_cohort_for_smoke(
    cohort_path: Path,
    *,
    limit: int,
    outcome_names: list[str],
    random_state: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(cohort_path)
    if limit <= 0:
        raise ValueError("Smoke cohort limit must be positive.")

    ordered_outcomes = sorted(
        outcome_names,
        key=lambda name: (0 if name == "any_aki" else 1, outcome_names.index(name)),
    )
    base_per_class = max(2, limit // max(2, len(ordered_outcomes) * 2))
    effective_limit = max(
        limit,
        max((MIN_PER_CLASS_BY_OUTCOME.get(name, 0) * 2 for name in ordered_outcomes), default=0),
    )
    selected_frames: list[pd.DataFrame] = []

    for outcome_name in ordered_outcomes:
        spec = get_trainable_outcome_spec(outcome_name)
        if spec.target_col not in df.columns:
            continue

        per_class = max(base_per_class, MIN_PER_CLASS_BY_OUTCOME.get(outcome_name, 0))
        labeled = df[df[spec.target_col].isin([0, 1])].copy()
        if labeled.empty:
            continue

        pos = _unique_subject_sample(labeled[labeled[spec.target_col] == 1], per_class)
        neg = _unique_subject_sample(labeled[labeled[spec.target_col] == 0], per_class)
        selected_frames.extend([pos, neg])

    if selected_frames:
        df_smoke = pd.concat(selected_frames, ignore_index=True).drop_duplicates(subset=["caseid"])
    else:
        df_smoke = _unique_subject_sample(df, effective_limit)

    if len(df_smoke) < effective_limit:
        additional = df.loc[~df["caseid"].isin(df_smoke["caseid"])].head(effective_limit - len(df_smoke))
        df_smoke = pd.concat([df_smoke, additional], ignore_index=True)

    df_smoke = df_smoke.head(effective_limit).sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_smoke.to_csv(cohort_path, index=False)
    return df_smoke


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim a cohort CSV for the smoke pipeline.")
    parser.add_argument("--cohort-path", type=Path, required=True, help="Path to the cohort CSV to trim in place.")
    parser.add_argument("--limit", type=int, required=True, help="Target number of cases to retain.")
    parser.add_argument(
        "--outcomes",
        default=",".join(DEFAULT_SMOKE_OUTCOMES),
        help="Comma-separated trainable outcomes that the smoke cohort must preserve class support for.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Shuffle seed for the trimmed cohort.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outcome_names = [item.strip() for item in args.outcomes.split(",") if item.strip()]
    df_smoke = trim_cohort_for_smoke(
        args.cohort_path,
        limit=args.limit,
        outcome_names=outcome_names or list(DEFAULT_SMOKE_OUTCOMES),
        random_state=args.random_state,
    )

    for outcome_name in outcome_names:
        spec = get_trainable_outcome_spec(outcome_name)
        if spec.target_col in df_smoke.columns:
            counts = df_smoke[spec.target_col].value_counts(dropna=False).to_dict()
            print(f"{outcome_name} label counts in smoke cohort: {counts}")

    print(f"Cohort trimmed to {len(df_smoke)} rows at {args.cohort_path}")


if __name__ == "__main__":
    main()
