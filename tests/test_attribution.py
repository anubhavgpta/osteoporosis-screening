import numpy as np
import pandas as pd
import pytest

from osteoporosis_screening.attribution import write_shap_attributions
from osteoporosis_screening.config import ProjectPaths


class HistoryModel:
    def predict_proba(self, frame):
        probability = np.where(frame.history.eq("yes"), 0.8, 0.2)
        return np.column_stack([1 - probability, probability])


def test_shap_decodes_text_categories_before_prediction(tmp_path):
    pytest.importorskip("shap")
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    frame = pd.DataFrame(
        {
            "age": [50.0, 60.0, 70.0, 80.0],
            "history": ["yes", "no", None, "yes"],
            "osteoporosis": [0, 1, 0, 1],
        }
    )
    status = write_shap_attributions(
        HistoryModel(),
        frame,
        ["age", "history"],
        paths,
        {"seed": 42, "shap_sample_n": 4, "shap_background_n": 4},
    )
    assert status["status"] == "completed"
    result = pd.read_csv(paths.tables / "classification_shap_patient_attributions.csv.gz")
    assert set(result.probability) == {0.2, 0.8}
    assert np.isfinite(result.shap_value).all()


def test_shap_can_be_disabled_explicitly(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    status = write_shap_attributions(None, None, [], paths, {"shap_enabled": False})
    assert status["status"] == "skipped"
