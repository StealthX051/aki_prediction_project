import pandas as pd

# Exclude pre-existing severe kidney disease (preop_cr > 4.0)
def filter_preop_cr(cohort_df: pd.DataFrame) -> pd.DataFrame:
    cohort_df = cohort_df[cohort_df['preop_cr'] <= 4.0].copy()
    print(f"Cases after excluding severe kidney disease: {len(cohort_df)}")
    return cohort_df
