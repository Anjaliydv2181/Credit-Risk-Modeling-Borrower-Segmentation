"""Statistical feature selection: Chi-Square, VIF, and ANOVA.

The goal is to keep only indicators that (a) are statistically associated with the
P1-P4 target and (b) are not redundant with one another (multicollinearity).

  * Categorical features -> Chi-Square test of independence vs the target.
  * Numerical features   -> VIF filter (drop collinear), then one-way ANOVA vs target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config


def select_categorical(df: pd.DataFrame, verbose: bool = True) -> list[str]:
    """Keep categorical features whose Chi-Square p-value <= alpha (vs target)."""
    kept = []
    for col in config.CATEGORICAL_FEATURES:
        _, pval, _, _ = chi2_contingency(pd.crosstab(df[col], df[config.TARGET]))
        keep = pval <= config.CHI2_ALPHA
        if keep:
            kept.append(col)
        if verbose:
            print(f"  chi2  {col:<18} p={pval:.3e}  {'KEEP' if keep else 'drop'}")
    return kept


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns eligible for modeling (exclude id, age, and target)."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    for col in (config.JOIN_KEY, "AGE"):
        if col in numeric:
            numeric.remove(col)
    return numeric


def filter_by_vif(df: pd.DataFrame, numeric_cols: list[str], verbose: bool = True) -> list[str]:
    """Iteratively drop the numeric features whose VIF exceeds the threshold."""
    vif_data = df[numeric_cols].copy()
    kept: list[str] = []
    col_index = 0

    for col in numeric_cols:
        # A perfectly collinear column gives VIF = inf (1/0); treat it as "drop".
        with np.errstate(divide="ignore", invalid="ignore"):
            vif_value = variance_inflation_factor(vif_data.values, col_index)
        if vif_value <= config.VIF_THRESHOLD:
            kept.append(col)
            col_index += 1
        else:
            vif_data = vif_data.drop(columns=[col])
    if verbose:
        print(f"  VIF<= {config.VIF_THRESHOLD}: kept {len(kept)} / {len(numeric_cols)} numeric features")
    return kept


def filter_by_anova(df: pd.DataFrame, numeric_cols: list[str], verbose: bool = True) -> list[str]:
    """Keep numeric features that differ significantly across P1-P4 (ANOVA p<=alpha)."""
    target = df[config.TARGET]
    groups_idx = {tier: (target == tier).values for tier in config.TIER_ORDER}

    kept = []
    for col in numeric_cols:
        values = df[col].values
        groups = [values[mask] for mask in groups_idx.values()]
        _, pval = f_oneway(*groups)
        if pval <= config.ANOVA_ALPHA:
            kept.append(col)
    if verbose:
        print(f"  ANOVA p<={config.ANOVA_ALPHA}: kept {len(kept)} / {len(numeric_cols)} numeric features")
    return kept


def select_features(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Run the full selection pipeline; return kept categorical + numeric features."""
    if verbose:
        print("Categorical selection (Chi-Square):")
    categorical = select_categorical(df, verbose=verbose)

    if verbose:
        print("Numerical selection (VIF -> ANOVA):")
    numeric_all = _numeric_feature_columns(df)
    numeric_vif = filter_by_vif(df, numeric_all, verbose=verbose)
    numeric_final = filter_by_anova(df, numeric_vif, verbose=verbose)

    return {
        "categorical": categorical,
        "numerical": numeric_final,
        "all": categorical + numeric_final,
    }


if __name__ == "__main__":
    from .data_prep import get_clean_data

    data = get_clean_data(verbose=False)
    selected = select_features(data)
    print(f"\nFinal feature count: {len(selected['all'])} "
          f"({len(selected['categorical'])} categorical + {len(selected['numerical'])} numerical)")
