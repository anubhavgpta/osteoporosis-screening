from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


class NutritionClusterAugmenter(BaseEstimator, TransformerMixin):
    """Learn dietary-pattern clusters on each training fold and append membership."""

    def __init__(
        self,
        features: tuple[str, ...],
        candidates: tuple[int, ...] = (2, 3, 4, 5, 6),
        random_state: int = 42,
    ) -> None:
        self.features = features
        self.candidates = candidates
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        values = X.loc[:, self.features]
        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        transformed = self.imputer_.fit_transform(values)
        transformed = self.scaler_.fit_transform(np.log1p(np.clip(transformed, 0, None)))

        valid_candidates = [k for k in self.candidates if 2 <= k < len(X)]
        if not valid_candidates:
            raise ValueError("Not enough rows to fit nutrition clusters")
        sample = transformed[: min(len(transformed), 5000)]
        metrics: dict[int, dict[str, float]] = {}
        models: dict[int, KMeans] = {}
        for k in valid_candidates:
            model = KMeans(n_clusters=k, n_init=20, random_state=self.random_state)
            labels = model.fit_predict(transformed)
            models[k] = model
            rng = np.random.default_rng(self.random_state + k)
            stability = []
            for repeat in range(5):
                subset = rng.choice(len(transformed), size=max(k * 5, int(0.8 * len(transformed))), replace=False)
                alternate = KMeans(
                    n_clusters=k,
                    n_init=10,
                    random_state=self.random_state + repeat + 1,
                ).fit(transformed[subset])
                stability.append(adjusted_rand_score(labels, alternate.predict(transformed)))
            counts = np.bincount(labels, minlength=k)
            metrics[k] = {
                "silhouette": float(silhouette_score(sample, labels[: len(sample)])),
                "calinski_harabasz": float(calinski_harabasz_score(transformed, labels)),
                "davies_bouldin": float(davies_bouldin_score(transformed, labels)),
                "stability_ari": float(np.mean(stability)),
                "smallest_cluster_fraction": float(counts.min() / counts.sum()),
            }
        eligible = [k for k in valid_candidates if metrics[k]["smallest_cluster_fraction"] >= 0.05]
        eligible = eligible or valid_candidates
        rank_order = {
            "silhouette": sorted(eligible, key=lambda k: metrics[k]["silhouette"], reverse=True),
            "calinski_harabasz": sorted(
                eligible, key=lambda k: metrics[k]["calinski_harabasz"], reverse=True
            ),
            "davies_bouldin": sorted(eligible, key=lambda k: metrics[k]["davies_bouldin"]),
            "stability_ari": sorted(eligible, key=lambda k: metrics[k]["stability_ari"], reverse=True),
        }
        average_rank = {
            k: float(np.mean([order.index(k) + 1 for order in rank_order.values()]))
            for k in eligible
        }
        self.cluster_metrics_ = metrics
        self.cluster_scores_ = {k: -rank for k, rank in average_rank.items()}
        self.n_clusters_ = min(average_rank, key=lambda k: (average_rank[k], k))
        self.model_ = models[self.n_clusters_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        transformed = self.imputer_.transform(result.loc[:, self.features])
        transformed = self.scaler_.transform(np.log1p(np.clip(transformed, 0, None)))
        result["nutrition_cluster"] = self.model_.predict(transformed).astype(str)
        return result


NUTRITION_CLUSTER_FEATURES = (
    "calcium_per_1000kcal",
    "vitamin_d_per_1000kcal",
    "magnesium_per_1000kcal",
    "protein_per_kg",
    "vitamin_e_mg",
    "phosphorus_mg",
)
