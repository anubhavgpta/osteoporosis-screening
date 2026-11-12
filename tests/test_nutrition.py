import numpy as np
import pandas as pd

from osteoporosis_screening.nutrition import (
    NUTRITION_CLUSTER_FEATURES,
    NutritionClusterAugmenter,
)


def test_cluster_augmenter_handles_missing_and_new_rows():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.lognormal(size=(40, len(NUTRITION_CLUSTER_FEATURES))),
        columns=NUTRITION_CLUSTER_FEATURES,
    )
    frame.loc[0, NUTRITION_CLUSTER_FEATURES[0]] = np.nan
    transformer = NutritionClusterAugmenter(
        NUTRITION_CLUSTER_FEATURES, candidates=(2, 3), random_state=7
    ).fit(frame.iloc[:30])
    transformed = transformer.transform(frame.iloc[30:])
    assert "nutrition_cluster" in transformed
    assert transformed["nutrition_cluster"].notna().all()
    assert transformer.n_clusters_ in (2, 3)

