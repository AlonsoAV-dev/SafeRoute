from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Separa la detección de actividad delictiva y la clasificación condicional de su nivel.
class HurdleClassifier:
    """P(algún delito) × P(alto | algún delito), con tres probabilidades coherentes."""

    def __init__(self, parameters):
        self.parameters = parameters
        self.event = XGBClassifier(**parameters, objective="binary:logistic")
        self.severity = XGBClassifier(**parameters, objective="binary:logistic")
        self.classes_ = np.array([0, 1, 2])

    def fit(self, x, y, sample_weight=None):
        self.event.fit(x, (y > 0).astype(np.int8), sample_weight=sample_weight)
        positive = y > 0
        self.severity.fit(x[positive], (y[positive] == 2).astype(np.int8),
                          sample_weight=None if sample_weight is None else sample_weight[positive])
        return self

    def predict_proba(self, x):
        event = self.event.predict_proba(x)[:, 1]
        high_given_event = self.severity.predict_proba(x)[:, 1]
        return np.column_stack((1 - event, event * (1 - high_given_event), event * high_given_event)).astype(np.float32)


# Instancia la familia de clasificador y sus parámetros a partir de la especificación del ensayo.
def create_model(spec, threads=10, seed=42):
    if spec["family"] == "rf":
        return RandomForestClassifier(
            n_estimators=spec.get("trees", 160), max_depth=spec.get("depth", 16),
            min_samples_leaf=8, max_features=spec.get("max_features", 0.75), class_weight="balanced_subsample",
            random_state=seed, n_jobs=threads,
        )
    parameters = dict(
        n_estimators=spec.get("trees", 300), max_depth=spec.get("depth", 4),
        learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
        min_child_weight=8, reg_lambda=5, tree_method="hist", n_jobs=threads, random_state=seed,
    )
    if spec["family"] == "hurdle":
        return HurdleClassifier(parameters)
    return XGBClassifier(**parameters, objective="multi:softprob", num_class=3)
