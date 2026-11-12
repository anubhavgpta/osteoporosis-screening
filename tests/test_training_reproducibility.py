import joblib
import numpy as np
import pandas as pd

from osteoporosis_screening.modeling import run_classification


def test_seeded_training_and_saved_model_reproduce_predictions(tmp_path):
    """Exercise real weighted fitting, calibration, thresholds, and serialization."""
    rng = np.random.default_rng(42)
    n = 100
    frame = pd.DataFrame(
        {
            "participant_id": np.arange(n),
            "age": rng.uniform(50, 85, n),
            "sex": np.tile([1, 2], n // 2),
            "race_ethnicity": 1,
            "survey_cycle": np.tile(["2007-2008", "2017-2018"], n // 2),
            "survey_stratum": 1,
            "survey_psu": np.tile([1, 2], n // 2),
            "osteoporosis": np.tile([0, 0, 0, 1], n // 4),
            "mec_weight": rng.uniform(1, 2, n),
            "dietary_day1_weight": 1.0,
        }
    )
    config = {
        "seed": 42,
        "test_size": 0.2,
        "cv_folds": 2,
        "cv_n_jobs": 1,
        "screening_sensitivity_target": 0.9,
    }
    results = [
        run_classification(frame, ["age", "sex"], {"sex"}, config, add_clusters=False)
        for _ in range(2)
    ]
    np.testing.assert_allclose(
        results[0].predictions.probability, results[1].predictions.probability, rtol=0, atol=1e-12
    )
    pd.testing.assert_frame_equal(results[0].thresholds, results[1].thresholds)
    artifact = tmp_path / "model.joblib"
    joblib.dump(results[0].model, artifact)
    heldout = frame.set_index("participant_id").loc[
        results[0].predictions.participant_id, ["age", "sex"]
    ]
    np.testing.assert_allclose(
        joblib.load(artifact).predict_proba(heldout)[:, 1],
        results[0].predictions.probability,
        rtol=0,
        atol=1e-12,
    )
