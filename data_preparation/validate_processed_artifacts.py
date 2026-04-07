from __future__ import annotations

import argparse
import sys

from data_preparation.artifact_metadata import (
    STEP_03_PREOP_ARTIFACT,
    STEP_05_MERGED_ARTIFACT,
    ArtifactCompatibilityError,
    read_versioned_csv,
)
from data_preparation.inputs import (
    PREOP_PROCESSED_FILE,
    WIDE_FEATURES_FILE,
    WIDE_FEATURES_WINDOWED_FILE,
)
from data_preparation.outcome_registry import (
    ALL_ELIGIBILITY_COLUMNS,
    ALL_OUTCOME_COLUMNS,
    ALL_SPLIT_COLUMNS,
)

PREOP_REQUIRED_COLUMNS = [
    "caseid",
    "subjectid",
    *ALL_OUTCOME_COLUMNS,
    *ALL_ELIGIBILITY_COLUMNS,
    *ALL_SPLIT_COLUMNS,
]

MERGED_REQUIRED_COLUMNS = [
    "caseid",
    "subjectid",
    *ALL_OUTCOME_COLUMNS,
    *ALL_ELIGIBILITY_COLUMNS,
    *ALL_SPLIT_COLUMNS,
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed Catch22 artifacts before reuse.")
    parser.add_argument(
        "--skip-preop",
        action="store_true",
        help="Skip validation of the Step 03 preoperative artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        if not args.skip_preop:
            read_versioned_csv(
                PREOP_PROCESSED_FILE,
                artifact_role=STEP_03_PREOP_ARTIFACT,
                required_columns=PREOP_REQUIRED_COLUMNS,
            )

        read_versioned_csv(
            WIDE_FEATURES_FILE,
            artifact_role=STEP_05_MERGED_ARTIFACT,
            required_columns=MERGED_REQUIRED_COLUMNS,
        )
        read_versioned_csv(
            WIDE_FEATURES_WINDOWED_FILE,
            artifact_role=STEP_05_MERGED_ARTIFACT,
            required_columns=MERGED_REQUIRED_COLUMNS,
        )
    except (ArtifactCompatibilityError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    print("Processed artifacts passed schema validation.")


if __name__ == "__main__":
    main()
