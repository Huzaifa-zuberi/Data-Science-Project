import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Tuple
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import optuna
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    def __init__(self, random_state: int = 42, n_trials: int = 50):
        self.random_state = random_state
        self.n_trials = n_trials
        self.best_params: Dict[str, Any] = {}

    def _get_xgboost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": self.random_state,
        }

    def _get_random_forest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": self.random_state,
        }

    def tune_xgboost(
        self, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5
    ) -> xgb.XGBClassifier:
        skf = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.random_state
        )

        def objective(trial):
            params = self._get_xgboost_params(trial)
            model = xgb.XGBClassifier(
                **params, eval_metric="logloss", use_label_encoder=False, n_jobs=-1
            )
            scores = cross_val_score(
                model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=self.n_trials)

        logger.info(f"Best XGBoost AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        best_model = xgb.XGBClassifier(
            **study.best_params, eval_metric="logloss",
            use_label_encoder=False, n_jobs=-1
        )
        best_model.fit(X_train, y_train)
        self.best_params["xgboost"] = study.best_params
        return best_model

    def tune_random_forest(
        self, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5
    ) -> RandomForestClassifier:
        skf = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.random_state
        )

        def objective(trial):
            params = self._get_random_forest_params(trial)
            model = RandomForestClassifier(**params, n_jobs=-1)
            scores = cross_val_score(
                model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials)

        logger.info(f"Best RF AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        best_model = RandomForestClassifier(**study.best_params, n_jobs=-1)
        best_model.fit(X_train, y_train)
        self.best_params["random_forest"] = study.best_params
        return best_model
