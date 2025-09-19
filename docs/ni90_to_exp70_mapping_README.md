NI90 → EXP70 Mapping (Template)
================================

This CSV documents how 90-class NIFlagsV2 labels (verification bundle) map to the curated 70-class experimental taxonomy used in the paper.

- File: `docs/ni90_to_exp70_mapping.csv`
- Columns:
  - `ni90_label`: label string from the 90-class verification taxonomy
  - `exp70_label`: corresponding label in the 70-class experimental taxonomy (empty if excluded)
  - `action`: one of `normalize` (context folded), `merge` (synonym/alias merged), or `exclude` (ultra-rare/low-confidence)
  - `rationale`: short reason for the action
  - `confidence`: curator confidence (e.g., high/medium/low)
  - `notes`: any additional curator notes

Guidance
--------

- Normalization: remove mounting/display-context variants when they do not alter the primary semantic class (enforce single-label evaluation).
- Merge: combine synonym/alias labels that would otherwise split samples across near-identical classes.
- Exclude: only for ultra-rare or low-confidence categories that cannot support meaningful evaluation.

Status
------

This is a starter template with a few example mappings. Curators can extend/confirm entries as needed for audit.

Verification
------------

- Rows with an empty `exp70_label` are interpreted as “excluded” in the experimental taxonomy.
- All other rows should point to a single experimental label.

