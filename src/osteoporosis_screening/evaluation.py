from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def threshold_for_sensitivity(
    y_true, probability, target: float, sample_weight=None
) -> float:
    false_positive, true_positive, thresholds = roc_curve(
        y_true, probability, sample_weight=sample_weight
    )
    eligible = np.flatnonzero(true_positive >= target)
    if len(eligible) == 0:
        return 0.5
    best = eligible[np.argmin(false_positive[eligible])]
    return float(thresholds[best])


def threshold_for_precision(
    y_true, probability, target: float, sample_weight=None
) -> float:
    precision, recall, thresholds = precision_recall_curve(
        y_true, probability, sample_weight=sample_weight
    )
    eligible = np.flatnonzero(precision[:-1] >= target)
    if len(eligible) == 0:
        return 1.0
    best_recall = recall[eligible].max()
    best = eligible[np.flatnonzero(recall[eligible] == best_recall)[-1]]
    return float(thresholds[best])


def threshold_for_f1(y_true, probability, sample_weight=None) -> float:
    """Return the threshold that maximizes weighted F1 on validation predictions."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    weights = (
        np.ones(len(y_true), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if len(y_true) == 0 or y_true.sum() == 0:
        return 0.5
    order = np.argsort(-probability, kind="mergesort")
    sorted_y = y_true[order]
    sorted_probability = probability[order]
    sorted_weights = weights[order]
    positive = sorted_y == 1
    true_positive = np.cumsum(sorted_weights * positive)
    false_positive = np.cumsum(sorted_weights * ~positive)
    false_negative = float(weights[y_true == 1].sum()) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=denominator > 0,
    )
    block_end = np.r_[sorted_probability[1:] != sorted_probability[:-1], True]
    candidates = np.flatnonzero(block_end)
    if len(candidates) == 0:
        return 0.5
    best = candidates[np.argmax(f1[candidates])]
    return float(sorted_probability[best])


def classification_metrics(y_true, probability, threshold=0.5, sample_weight=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    clipped_probability = np.clip(probability, 1e-6, 1 - 1e-6)
    if sample_weight is None:
        metric_weight = np.ones(len(y_true), dtype=float)
    else:
        metric_weight = np.asarray(sample_weight, dtype=float)
    predicted = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[0, 1], sample_weight=sample_weight
    ).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator else np.nan
    prevalence = np.average(y_true, weights=metric_weight)
    brier = brier_score_loss(y_true, probability, sample_weight=metric_weight)
    null_brier = np.average((y_true - prevalence) ** 2, weights=metric_weight)
    logits = np.log(clipped_probability / (1 - clipped_probability)).reshape(-1, 1)
    calibration = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    calibration.fit(logits, y_true, sample_weight=metric_weight)
    return {
        "roc_auc": roc_auc_score(y_true, probability, sample_weight=sample_weight),
        "pr_auc": average_precision_score(y_true, probability, sample_weight=sample_weight),
        "accuracy": accuracy_score(y_true, predicted, sample_weight=sample_weight),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1_score(y_true, predicted, sample_weight=sample_weight),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predicted, sample_weight=sample_weight
        ),
        "mcc": mcc,
        "log_loss": log_loss(y_true, clipped_probability, sample_weight=metric_weight),
        "brier_score": brier,
        "brier_skill_score": 1 - (brier / null_brier) if null_brier else np.nan,
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
        "threshold": threshold,
        "weighted_tn": tn,
        "weighted_fp": fp,
        "weighted_fn": fn,
        "weighted_tp": tp,
    }


def decision_curve(
    y_true, probability, thresholds, sample_weight=None
) -> pd.DataFrame:
    """Return weighted net benefit for model, test-all, and test-none policies."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    weights = (
        np.ones(len(y_true), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    total_weight = weights.sum()
    prevalence = np.average(y_true, weights=weights)
    rows = []
    for threshold in thresholds:
        if not 0 < threshold < 1:
            raise ValueError("Decision-curve thresholds must be strictly between 0 and 1")
        predicted = probability >= threshold
        tp_rate = weights[predicted & (y_true == 1)].sum() / total_weight
        fp_rate = weights[predicted & (y_true == 0)].sum() / total_weight
        odds = threshold / (1 - threshold)
        model_benefit = tp_rate - fp_rate * odds
        all_benefit = prevalence - (1 - prevalence) * odds
        rows.append(
            {
                "threshold_probability": float(threshold),
                "model_net_benefit": model_benefit,
                "test_all_net_benefit": all_benefit,
                "test_none_net_benefit": 0.0,
                "standardized_model_net_benefit": (
                    model_benefit / prevalence if prevalence else np.nan
                ),
                "unnecessary_referrals_avoided_per_100": (
                    (model_benefit - all_benefit) / odds * 100
                ),
            }
        )
    return pd.DataFrame(rows)


def regression_metrics(y_true, predicted, sample_weight=None) -> dict:
    return {
        "mae": mean_absolute_error(y_true, predicted, sample_weight=sample_weight),
        "rmse": mean_squared_error(y_true, predicted, sample_weight=sample_weight) ** 0.5,
        "r2": r2_score(y_true, predicted, sample_weight=sample_weight),
    }
