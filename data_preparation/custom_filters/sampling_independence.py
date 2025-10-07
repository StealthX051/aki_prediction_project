import pandas as pd

# Ensure sample independence by selecting only one surgery per patient first
# This is critical for statistical validity
def ensure_sample_independence(cohort_df: pd.DataFrame) -> pd.DataFrame:
    independent_cohort_df = cohort_df.groupby('subjectid').sample(n=1, random_state=42)
    print(f"Cohort size after ensuring sample independence: {len(independent_cohort_df)}")
    return independent_cohort_df
