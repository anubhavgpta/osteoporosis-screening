import pandas as pd
import pytest

from osteoporosis_screening.target_definition import (
    FEATURE_COMBINATION_RIGOROUS_METRIC_COLUMNS,
    _naive_lasso_full_dataset_selection,
    derive_low_bmd_label,
    rigorous_feature_combination_metric_columns,
    run_target_definition_experiment,
)


def test_low_bmd_label_derivation_is_strict_below_minus_one():
    frame = pd.DataFrame({"femoral_neck_t_score": [-0.99, -1.0, -1.01, -2.5, None]})

    result = derive_low_bmd_label(frame, cutoff=-1.0)

    assert result.tolist()[:4] == [0, 0, 1, 1]
    assert pd.isna(result.iloc[4])


def test_naive_feature_selection_leak_fits_full_pre_split_dataset():
    frame = pd.DataFrame(
        {
            "analysis_outcome": [0, 1, 0, 1, 0, 1],
            "age": [50, 55, 60, 65, 70, 75],
            "bmi": [31, 25, 30, 24, 29, 23],
            "sex": [1, 2, 1, 2, 1, 2],
        }
    )

    audit = _naive_lasso_full_dataset_selection(
        frame,
        ["age", "bmi", "sex"],
        {"sex"},
        {"seed": 42, "naive_lasso_selection_c": 1.0},
    )

    assert audit["selector_fit_n"] == len(frame)
    assert audit["selected_features"].shape[0] == len(frame)
    assert audit["selected_feature_count"] > 0


def test_rigorous_output_structure_matches_feature_combination_path():
    assert rigorous_feature_combination_metric_columns() == (
        FEATURE_COMBINATION_RIGOROUS_METRIC_COLUMNS
    )


def test_command_refuses_existing_target_definition_outputs_without_overwrite(tmp_path):
    (tmp_path / "reports" / "tables").mkdir(parents=True)
    existing = tmp_path / "reports" / "tables" / "target_definition_cell_metrics.csv"
    existing.write_text("already,there\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_target_definition_experiment(tmp_path, overwrite=False)
