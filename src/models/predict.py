import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, Union
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_path: Path, preprocessor_path: Path):
        self.model = self._load_model(model_path)
        self.preprocessor = self._load_preprocessor(preprocessor_path)

    def _load_model(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
        logger.info(f"Loading model from {path}")
        return joblib.load(path)

    def _load_preprocessor(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {path}")
        logger.info(f"Loading preprocessor from {path}")
        return joblib.load(path)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        processed = self.preprocessor.transform(df)
        return self.model.predict(processed)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        processed = self.preprocessor.transform(df)
        return self.model.predict_proba(processed)

    def predict_with_details(self, df: pd.DataFrame) -> pd.DataFrame:
        processed = self.preprocessor.transform(df)
        predictions = self.model.predict(processed)
        probabilities = self.model.predict_proba(processed)[:, 1]

        results = df.copy()
        results["prediction"] = predictions
        results["churn_probability"] = probabilities
        results["risk_level"] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )
        return results

    def get_risk_factors(self, df: pd.DataFrame) -> Dict[str, Any]:
        processed = self.preprocessor.transform(df)
        proba = self.model.predict_proba(processed)[:, 1][0]

        risk_factors = {}
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            for i, col in enumerate(processed.columns):
                if i < len(importances) and processed.iloc[0, i] > 0:
                    risk_factors[col] = float(importances[i] * processed.iloc[0, i])

            risk_factors = dict(
                sorted(risk_factors.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            )

        return {
            "churn_probability": float(proba),
            "risk_factors": risk_factors,
        }
