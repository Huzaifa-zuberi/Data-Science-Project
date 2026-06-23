import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import joblib
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Any = None

    def get_models(self) -> Dict[str, Any]:
        return {
            "logistic_regression": LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            "random_forest": RandomForestClassifier(
                random_state=self.random_state, n_jobs=-1, n_estimators=300,
                max_depth=15, min_samples_split=10, min_samples_leaf=4
            ),
            "xgboost": xgb.XGBClassifier(
                random_state=self.random_state, n_jobs=-1, eval_metric="logloss",
                n_estimators=300, max_depth=8,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
            ),
            "lightgbm": lgb.LGBMClassifier(
                random_state=self.random_state, n_jobs=-1, n_estimators=300,
                max_depth=8, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, verbose=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                random_state=self.random_state, n_estimators=200,
                max_depth=6, learning_rate=0.05, subsample=0.8
            ),
        }

    def train_and_evaluate(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Dict[str, float]]:
        self.models = self.get_models()
        skf = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.random_state
        )

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            self.results[name] = {
                "cv_mean_auc": cv_scores.mean(),
                "cv_std_auc": cv_scores.std(),
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred, zero_division=0),
                "test_recall": recall_score(y_test, y_pred, zero_division=0),
                "test_f1": f1_score(y_test, y_pred, zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, y_proba),
            }

            logger.info(
                f"{name}: CV AUC={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f}), "
                f"Test AUC={self.results[name]['test_roc_auc']:.4f}"
            )

        self._select_best_model()
        return self.results

    def _select_best_model(self):
        self.best_model_name = max(
            self.results, key=lambda k: self.results[k]["test_roc_auc"]
        )
        self.best_model = self.models[self.best_model_name]
        logger.info(f"Best model: {self.best_model_name} (AUC={self.results[self.best_model_name]['test_roc_auc']:.4f})")

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        if self.best_model is None:
            return pd.DataFrame()

        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importances = np.abs(self.best_model.coef_[0])
        else:
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            "feature": feature_names[:len(importances)],
            "importance": importances,
        }).sort_values("importance", ascending=False)

        importance_df["importance_pct"] = (
            importance_df["importance"] / importance_df["importance"].sum() * 100
        )
        return importance_df

    def save_model(self, path: Path):
        if self.best_model is None:
            raise ValueError("No model to save. Train models first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> Any:
        model = joblib.load(path)
        self.best_model = model
        logger.info(f"Model loaded from {path}")
        return model



