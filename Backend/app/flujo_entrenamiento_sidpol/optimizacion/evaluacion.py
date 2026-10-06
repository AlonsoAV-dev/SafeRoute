from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix


def predict_levels(probability, cutoffs):
    labels = np.zeros(len(probability), dtype=np.int8)
    labels[probability[:, 1] >= cutoffs["medio"]] = 1
    labels[probability[:, 2] >= cutoffs["alto"]] = 2
    return labels


# Calcula las métricas multicategoría y la Average Precision de Alto usando probabilidades.
def metrics(truth, probability, cutoffs):
    matrix = confusion_matrix(truth, predict_levels(probability, cutoffs), labels=[0, 1, 2])
    actual, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    diagonal = matrix.diagonal()
    precision = np.divide(diagonal, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(diagonal, actual, out=np.zeros(3), where=actual > 0)
    f1 = np.divide(2 * diagonal, actual + predicted, out=np.zeros(3), where=(actual + predicted) > 0)
    return {
        "n": int(matrix.sum()), "accuracy": float(diagonal.sum() / matrix.sum()),
        "f1_medio_alto": float(f1[1:].mean()), "f1_macro": float(f1.mean()),
        "clases": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                   "reales": int(actual[i]), "predichos": int(predicted[i]), "aciertos": int(diagonal[i]),
                   "average_precision": float(average_precision_score(truth == i, probability[:, i]))}
            for i, name in enumerate(("bajo", "medio", "alto"))
        },
        "matriz_confusion": matrix.tolist(),
    }


# Explora umbrales conjuntos sobre las observaciones destinadas a decidir la configuración.
def tune_joint_thresholds(truth, probability):
    """Optimiza F1 medio/alto conjuntamente mediante histogramas bidimensionales."""
    grids = []
    for label in (1, 2):
        score = probability[:, label]
        positive = score[truth == label]
        grids.append(np.unique(np.r_[0.0, np.quantile(positive, np.linspace(0, 1, 55)),
                                     np.quantile(score, np.linspace(0.9, 0.9999, 35)), 1.000001]))
    medium, high = grids
    bins_m = np.searchsorted(medium, probability[:, 1], side="right")
    bins_h = np.searchsorted(high, probability[:, 2], side="right")
    shape = (len(medium) + 1, len(high) + 1)
    linear = bins_m * shape[1] + bins_h
    medium_counts, high_counts = [], []
    for label in range(3):
        hist = np.bincount(linear[truth == label], minlength=shape[0] * shape[1]).reshape(shape)
        cumulative = hist.cumsum(axis=1)[::-1].cumsum(axis=0)[::-1]
        medium_counts.append(cumulative[1:, :-1])
        high_counts.append(hist.sum(axis=0)[::-1].cumsum()[::-1][1:])
    m_counts = np.stack(medium_counts)
    h_counts = np.stack(high_counts)
    supports = np.bincount(truth, minlength=3)
    f1_medium = 2 * m_counts[1] / np.maximum(supports[1] + m_counts.sum(axis=0), 1)
    f1_high = 2 * h_counts[2] / np.maximum(supports[2] + h_counts.sum(axis=0), 1)
    objective = (f1_medium + f1_high[None, :]) / 2
    i, j = np.unravel_index(np.argmax(objective), objective.shape)
    return {"medio": float(medium[i]), "alto": float(high[j])}


# Ajusta la transformación de probabilidades antes de aplicar los cortes de decisión.
class ProbabilityCalibration:
    """Calibración multinomial con distribución natural reconstruida por pesos."""

    @staticmethod
    def transform(probability):
        safe = np.clip(probability, 1e-7, 1)
        return np.log(safe[:, 1:] / safe[:, :1])

    def fit(self, truth, probability, seed=42):
        rng = np.random.default_rng(seed)
        low = np.flatnonzero(truth == 0)
        sampled_low = rng.choice(low, min(160_000, len(low)), replace=False)
        positive = np.flatnonzero(truth > 0)
        chosen = np.r_[positive, sampled_low]
        weight = np.r_[np.ones(len(positive)), np.full(len(sampled_low), len(low) / max(len(sampled_low), 1))]
        weight /= weight.mean()
        self.model = LogisticRegression(C=10.0, max_iter=200, random_state=seed)
        self.model.fit(self.transform(probability[chosen]), truth[chosen], sample_weight=weight)
        return self

    def predict_proba(self, probability):
        return self.model.predict_proba(self.transform(probability)).astype(np.float32)
