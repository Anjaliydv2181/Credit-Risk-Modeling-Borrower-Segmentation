"""Central configuration: paths, constants, and the P1-P4 segmentation policy.

Keeping every "magic number" and business rule in one place makes the pipeline
reproducible and easy to audit.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CASE_STUDY_1 = DATA_DIR / "case_study1.xlsx"   # trade-line data (lender's own records)
CASE_STUDY_2 = DATA_DIR / "case_study2.xlsx"   # bureau (CIBIL) + demographic data

MODEL_PATH = MODELS_DIR / "xgb_credit_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"

# --------------------------------------------------------------------------- #
# Data cleaning
# --------------------------------------------------------------------------- #
# Sentinel value used in the source data to mark "missing".
MISSING_SENTINEL = -99999

# In case_study2 these columns carry too many sentinels (>10k of ~51k rows) to be
# usable, so they are dropped wholesale rather than dropping the rows.
HIGH_MISSING_COLUMNS = [
    "time_since_first_deliquency",
    "time_since_recent_deliquency",
    "max_delinquency_level",
    "max_deliq_6mts",
    "max_deliq_12mts",
    "CC_utilization",
    "PL_utilization",
    "max_unsec_exposure_inPct",
]

JOIN_KEY = "PROSPECTID"
TARGET = "Approved_Flag"

# Expected size after the documented cleaning recipe — asserted for reproducibility.
EXPECTED_ROWS = 42064

# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
VIF_THRESHOLD = 6.0       # drop numerical features whose VIF exceeds this
ANOVA_ALPHA = 0.05        # keep numerical features with ANOVA p-value <= alpha
CHI2_ALPHA = 0.05         # categorical features with chi-square p-value <= alpha

CATEGORICAL_FEATURES = [
    "MARITALSTATUS",
    "EDUCATION",
    "GENDER",
    "last_prod_enq2",
    "first_prod_enq2",
]

# EDUCATION is ordinal — encode by rough years of education / qualification level.
EDUCATION_MAP = {
    "SSC": 1,
    "OTHERS": 1,
    "12TH": 2,
    "GRADUATE": 3,
    "UNDER GRADUATE": 3,
    "PROFESSIONAL": 3,
    "POST-GRADUATE": 4,
}

# Columns standard-scaled before modeling (large/heterogeneous numeric ranges).
COLUMNS_TO_SCALE = [
    "Age_Oldest_TL",
    "Age_Newest_TL",
    "time_since_recent_payment",
    "max_recent_level_of_deliq",
    "recent_level_of_deliq",
    "time_since_recent_enq",
    "NETMONTHLYINCOME",
    "Time_With_Curr_Empr",
]

# --------------------------------------------------------------------------- #
# Modeling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20

RF_PARAMS = {"n_estimators": 200, "random_state": RANDOM_STATE}

# Best params found via GridSearchCV in the original analysis.
XGB_BEST_PARAMS = {
    "objective": "multi:softmax",
    "num_class": 4,
    "learning_rate": 0.2,
    "max_depth": 3,
    "n_estimators": 200,
}

# --------------------------------------------------------------------------- #
# Borrower segmentation framework (P1 = best / lowest risk ... P4 = worst)
# Ordering validated against mean credit score: P1=716 > P2=683 > P3=667 > P4=646.
# --------------------------------------------------------------------------- #
RECOMMENDATIONS = {
    "P1": {
        "risk": "Lowest risk",
        "decision": "Auto-approve",
        "action": "Approve with best pricing and highest credit limit; minimal manual review.",
    },
    "P2": {
        "risk": "Low-to-moderate risk",
        "decision": "Approve",
        "action": "Approve on standard terms; routine verification.",
    },
    "P3": {
        "risk": "Elevated risk",
        "decision": "Manual review",
        "action": "Refer to underwriting; approve only with risk-based pricing / reduced limit / collateral.",
    },
    "P4": {
        "risk": "High risk",
        "decision": "Decline / escalate",
        "action": "Decline or refer to senior underwriting; approve only as a secured/guaranteed exception.",
    },
}

TIER_ORDER = ["P1", "P2", "P3", "P4"]
