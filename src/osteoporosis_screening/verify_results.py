"""Independently reload locally trained models and verify exported predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .external import audit_external_cohort
from .reproduce import verify_run


def verify_predictions(root: Path) -> dict:
    verify_run(root)
    metadata = json.loads((root / "artifacts/classification_model_metadata.json").read_text())
    cohort = pd.read_csv(root / "data/processed/classification_cohort.csv.gz", low_memory=False)
    cohort = cohort.set_index("participant_id", verify_integrity=True)
    checks = {}
    for label, model_name, prefix in (
        ("internal", "classification_model.joblib", "classification"),
        ("temporal", "classification_temporal_model.joblib", "classification_temporal_validation"),
    ):
        saved = pd.read_csv(root / f"reports/tables/{prefix}_test_predictions.csv")
        features = cohort.loc[saved.participant_id, metadata["features"]]
        model = joblib.load(root / "artifacts" / model_name)
        probability = model.predict_proba(features)[:, 1]
        np.testing.assert_allclose(probability, saved.probability, rtol=0, atol=1e-12)
        checks[label] = {
            "n": len(saved),
            "max_absolute_difference": float(np.max(np.abs(probability - saved.probability))),
        }
    external_path = root / "reports/tables/external_knhanes_2008_2011_predictions.csv"
    if external_path.exists():
        audit = audit_external_cohort(
            root,
            "knhanes_2008_2011",
            root / "data/processed/knhanes_2008_2011_harmonized.csv.gz",
        )
        saved = pd.read_csv(external_path).set_index("participant_id", verify_integrity=True)
        fresh = audit.predictions.set_index("participant_id", verify_integrity=True)
        np.testing.assert_array_equal(saved.index, fresh.index)
        np.testing.assert_allclose(saved.probability, fresh.probability, rtol=0, atol=1e-12)
        checks["external"] = {
            "n": len(saved),
            "max_absolute_difference": float((saved.probability - fresh.probability).abs().max()),
        }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_predictions(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
