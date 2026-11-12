from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ProjectPaths


def _write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_shap_attributions(
    model,
    frame: pd.DataFrame,
    features: list[str],
    paths: ProjectPaths,
    config: dict,
    prefix: str = "classification",
    target: str = "osteoporosis",
) -> dict:
    """Write optional SHAP summaries for the final calibrated classifier."""
    manifest_path = paths.tables / f"{prefix}_shap_manifest.json"
    if importlib.util.find_spec("shap") is None:
        payload = {
            "status": "skipped",
            "reason": "shap is not installed",
            "install_hint": "Install the optional attribution extra or add shap to the environment.",
        }
        _write_json(payload, manifest_path)
        return payload

    import shap

    usable = frame.dropna(subset=[target]).copy()
    sample_size = min(int(config.get("shap_sample_n", 250)), len(usable))
    background_size = min(int(config.get("shap_background_n", 100)), len(usable))
    seed = int(config.get("seed", 42))
    background = usable.sample(background_size, random_state=seed)
    explained = usable.sample(sample_size, random_state=seed + 1)

    def predict_probability(values):
        data = pd.DataFrame(values, columns=features)
        return model.predict_proba(data)[:, 1]

    masker = shap.maskers.Independent(background[features])
    explainer = shap.PermutationExplainer(predict_probability, masker, seed=seed)
    shap_values = explainer(
        explained[features],
        max_evals=int(config.get("shap_max_evals", 2 * len(features) + 1)),
    )
    values = np.asarray(shap_values.values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, -1]

    importance = (
        pd.DataFrame(
            {
                "feature": features,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "mean_shap": values.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(paths.tables / f"{prefix}_shap_feature_importance.csv", index=False)

    probabilities = predict_probability(explained[features])
    long_rows = []
    for row_number, (_, row) in enumerate(explained.iterrows()):
        participant_id = row.get("participant_id", row_number)
        for column_number, feature in enumerate(features):
            long_rows.append(
                {
                    "participant_id": participant_id,
                    "feature": feature,
                    "feature_value": row[feature],
                    "shap_value": values[row_number, column_number],
                    "probability": probabilities[row_number],
                }
            )
    pd.DataFrame(long_rows).to_csv(
        paths.tables / f"{prefix}_shap_patient_attributions.csv.gz",
        index=False,
        compression="gzip",
    )

    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#3b6ea8")
    ax.set(xlabel="Mean absolute SHAP value", ylabel="", title="Final model SHAP importance")
    fig.tight_layout()
    fig.savefig(paths.figures / f"{prefix}_shap_feature_importance.png", dpi=160)
    plt.close(fig)

    payload = {
        "status": "completed",
        "method": "PermutationExplainer on calibrated predict_proba",
        "background_n": background_size,
        "explained_n": sample_size,
        "feature_count": len(features),
        "outputs": {
            "importance": str(
                (paths.tables / f"{prefix}_shap_feature_importance.csv").relative_to(
                    paths.root
                )
            ),
            "patient_attributions": str(
                (
                    paths.tables / f"{prefix}_shap_patient_attributions.csv.gz"
                ).relative_to(paths.root)
            ),
            "plot": str(
                (
                    paths.figures / f"{prefix}_shap_feature_importance.png"
                ).relative_to(paths.root)
            ),
        },
    }
    _write_json(payload, manifest_path)
    return payload
