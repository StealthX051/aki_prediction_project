import pandas as pd


# Exclude patients without outcome data (postoperative labs)
def filter_postop_cr(independent_cohort_df: pd.DataFrame) -> pd.DataFrame:
    # Load the lab results data first, as we'll need it for filtering
    lab_df = pd.read_csv('./data/raw/lab_data.csv')
    cr_labs = lab_df[lab_df['name'] == 'cr']

    print("\nFiltering cohort to include only patients with postoperative creatinine labs...")

    # To do this efficiently, we first find which caseids have the required labs
    # Merge cohort opend times with creatinine labs
    merged_labs = pd.merge(independent_cohort_df[['caseid', 'opend']], cr_labs, on='caseid')

    # Filter for labs taken between the end of the operation and 7 days after
    postop_labs_7d = merged_labs[
        (merged_labs['dt'] > merged_labs['opend']) &
        (merged_labs['dt'] <= merged_labs['opend'] + (7 * 24 * 3600))
    ]

    # Get the unique set of caseids that have at least one valid post-op lab
    valid_caseids_with_labs = set(postop_labs_7d['caseid'])

    # Filter the main cohort to keep only the cases in our valid set
    final_cohort_df = independent_cohort_df[independent_cohort_df['caseid'].isin(valid_caseids_with_labs)].copy()

    print(f"Final study cohort size after excluding patients without post-op labs: {len(final_cohort_df)}")
    return final_cohort_df