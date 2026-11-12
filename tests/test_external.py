import json

import numpy as np
import pandas as pd

from osteoporosis_screening import external


class _FakeModel:
    def predict_proba(self, frame):
        probability = np.where(frame["age"].to_numpy() >= 70, 0.8, 0.2)
        return np.column_stack([1 - probability, probability])


def test_external_adapter_derives_identical_osteoporosis_endpoint(
    tmp_path, monkeypatch
):
    (tmp_path / "configs").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "configs" / "external_datasets.yaml").write_text(
        """
harmonized_outcome:
  reference_mean_bmd: 0.858
  reference_sd_bmd: 0.120
datasets:
  example: {}
""",
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "classification_model_metadata.json").write_text(
        json.dumps(
            {
                "features": ["age", "sex", "age_sex_interaction"],
                "sensitivity_oriented_threshold": 0.3,
                "operating_points": {
                    "default": 0.5,
                    "sensitivity_oriented": 0.3,
                    "precision_0.20": 0.7,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "classification_model.joblib").write_bytes(b"fake-model")
    source = tmp_path / "external.csv"
    pd.DataFrame(
        {
            "participant_id": [1, 2, 3, 4],
            "age": [55, 60, 70, 80],
            "sex": [1, 2, 1, 2],
            "femoral_neck_bmd": [0.8, 0.75, 0.55, 0.5],
        }
    ).to_csv(source, index=False)
    monkeypatch.setattr(external.joblib, "load", lambda _: _FakeModel())

    metrics, predictions = external.evaluate_external_cohort(
        tmp_path, "example", source
    )

    assert predictions["osteoporosis"].tolist() == [0, 0, 1, 1]
    assert set(metrics["target"]) == {"osteoporosis"}
    assert set(metrics["evaluation"]) == {
        "default",
        "sensitivity_oriented",
        "precision_0.20",
    }
    assert metrics["evaluation_mode"].eq(
        "frozen_model_frozen_nhanes_threshold"
    ).all()
    assert metrics.set_index("evaluation").loc["sensitivity_oriented", "threshold"] == 0.3

    audit = external.audit_external_cohort(tmp_path, "example", source)
    assert set(audit.tradeoffs["measure"]) == {
        "precision_at_recall",
        "recall_at_precision",
    }
    assert not audit.calibration.empty
    assert audit.manifest["model_fitted_on_external_data"] is False
    assert audit.manifest["deployment_threshold_selected_on_external_data"] is False
    assert audit.manifest["input_sha256"]
    assert audit.cohort_flow.iloc[-1]["n"] == 4


def test_external_adapter_rejects_invalid_survey_weights(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "configs" / "external_datasets.yaml").write_text(
        """
harmonized_outcome:
  reference_mean_bmd: 0.858
  reference_sd_bmd: 0.120
datasets:
  example: {}
""",
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "classification_model_metadata.json").write_text(
        json.dumps(
            {
                "features": ["age", "sex", "age_sex_interaction"],
                "sensitivity_oriented_threshold": 0.3,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "classification_model.joblib").write_bytes(b"fake-model")
    source = tmp_path / "external.csv"
    pd.DataFrame(
        {
            "age": [55, 60, 70, 80],
            "sex": [1, 2, 1, 2],
            "femoral_neck_bmd": [0.8, 0.75, 0.55, 0.5],
            "external_weight": [1.0, 0.0, 1.0, 1.0],
        }
    ).to_csv(source, index=False)
    monkeypatch.setattr(external.joblib, "load", lambda _: _FakeModel())

    with np.testing.assert_raises_regex(ValueError, "finite and positive"):
        external.audit_external_cohort(tmp_path, "example", source)
