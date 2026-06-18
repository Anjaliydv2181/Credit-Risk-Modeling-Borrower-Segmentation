"""Score borrowers with the trained model and emit a tier + recommendation.

Run as a CLI demo (`python -m src.predict`) to score a few held-out borrowers,
or import `score_borrowers` to integrate into another application.
"""
from __future__ import annotations

import joblib
import pandas as pd

from . import config, segmentation


def load_bundle(path=config.MODEL_PATH) -> dict:
    """Load the persisted {model, feature_names, label_encoder} bundle."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def score_borrowers(X: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Predict P1-P4 tier + recommendation for each row of a feature frame.

    `X` must contain the bundle's feature columns; any extras are ignored and any
    missing columns are filled with 0 (e.g. absent one-hot dummy categories).
    """
    bundle = bundle or load_bundle()
    model = bundle["model"]
    feat = bundle["feature_names"]
    label_encoder = bundle["label_encoder"]

    X_aligned = X.reindex(columns=feat, fill_value=0)
    pred_codes = model.predict(X_aligned)
    tiers = label_encoder.inverse_transform(pred_codes)

    rows = []
    for tier in tiers:
        rec = segmentation.recommend(tier)
        rows.append({"tier": tier, "risk": rec["risk"],
                     "decision": rec["decision"], "action": rec["action"]})
    return pd.DataFrame(rows, index=X.index)


def _demo(n: int = 5) -> None:
    """Rebuild the encoded feature matrix and score a sample of borrowers."""
    from .data_prep import get_clean_data
    from .feature_selection import select_features
    from .train import build_feature_matrix, scale_features

    bundle = load_bundle()
    df = get_clean_data(verbose=False)
    features = select_features(df, verbose=False)
    encoded = scale_features(build_feature_matrix(df, features))

    sample = encoded.drop(columns=[config.TARGET]).sample(
        n=n, random_state=config.RANDOM_STATE
    )
    actual = df.loc[sample.index, config.TARGET]

    result = score_borrowers(sample)
    result.insert(0, "actual_tier", actual.values)

    print("Sample borrower scoring (predicted tier + underwriting recommendation):\n")
    for idx, row in result.iterrows():
        print(f"Borrower {idx} | actual={row['actual_tier']} | "
              f"predicted={row['tier']} ({row['risk']})")
        print(f"    -> {row['decision']}: {row['action']}\n")


if __name__ == "__main__":
    _demo()
