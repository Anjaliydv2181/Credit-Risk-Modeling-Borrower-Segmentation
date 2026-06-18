"""Load, clean, and merge the trade-line and bureau datasets.

Cleaning recipe (reproduces the 42,064-borrower analysis dataset):
  1. Drop rows in case_study1 where Age_Oldest_TL is the missing sentinel.
  2. Drop the 8 case_study2 columns that are >20% sentinel-missing.
  3. Drop remaining case_study2 rows that still contain any sentinel.
  4. Inner-merge the two tables on PROSPECTID.
"""
from __future__ import annotations

import pandas as pd

from . import config


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two source Excel files (trade-line + bureau)."""
    df1 = pd.read_excel(config.CASE_STUDY_1)   # trade-line / account-level
    df2 = pd.read_excel(config.CASE_STUDY_2)   # bureau (CIBIL) + demographic
    return df1, df2


def clean_and_merge(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented cleaning recipe and inner-merge on PROSPECTID."""
    sentinel = config.MISSING_SENTINEL

    # 1. trade-line: a sentinel in Age_Oldest_TL marks an unusable record.
    df1 = df1.loc[df1["Age_Oldest_TL"] != sentinel].copy()

    # 2. bureau: drop columns that are mostly missing.
    df2 = df2.drop(columns=config.HIGH_MISSING_COLUMNS, errors="ignore").copy()

    # 3. bureau: drop rows that still carry a sentinel in any remaining column.
    for col in df2.columns:
        df2 = df2.loc[df2[col] != sentinel]

    # 4. combine the lender's trade-line view with the bureau view.
    df = pd.merge(df1, df2, how="inner", on=config.JOIN_KEY)
    return df


def get_clean_data(verbose: bool = True) -> pd.DataFrame:
    """End-to-end: load -> clean -> merge, with a reproducibility self-check."""
    df1, df2 = load_raw()
    df = clean_and_merge(df1, df2)

    n_null = int(df.isna().sum().sum())
    if verbose:
        print(f"case_study1 (trade-line): {df1.shape}")
        print(f"case_study2 (bureau)    : {df2.shape}")
        print(f"merged analysis dataset : {df.shape}  | nulls: {n_null}")
        print(f"target distribution:\n{df[config.TARGET].value_counts()}")

    assert n_null == 0, f"Expected 0 nulls after cleaning, found {n_null}"
    assert df.shape[0] == config.EXPECTED_ROWS, (
        f"Expected {config.EXPECTED_ROWS} rows, got {df.shape[0]}"
    )
    return df


if __name__ == "__main__":
    get_clean_data()
