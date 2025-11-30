---
description: Check for potential data leakage in VitalDB Catch22 Project
---

# Check for potential data leakage

- Inspect the data-preparation and modeling code touched in this change.
- Verify that:
  - the train/test split is performed once and reused consistently,
  - all statistics (outliers, imputations, encodings) are fitted on training data only,
  - the test set is not used in any model selection or preprocessing decisions.
- Suggest fixes if any leakage risks are detected.
