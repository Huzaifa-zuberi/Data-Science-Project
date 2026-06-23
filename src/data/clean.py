import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional


class DataCleaner:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.label_encoders: dict = {}
        self._fitted = False

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop_duplicates()
        df = df.drop(columns=["customer_id"], errors="ignore")

        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        df["tenure"] = df["tenure"].clip(lower=0)
        df["monthly_charges"] = df["monthly_charges"].clip(lower=0)

        return df

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "churn"
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = self.clean(df)
        y = df[target_col].copy() if target_col in df.columns else None

        if y is not None:
            valid = y.notna()
            df = df[valid]
            y = y[valid]

        feature_df = df.drop(columns=[target_col], errors="ignore")
        cat_cols = feature_df.select_dtypes(include=["object"]).columns
        num_cols = feature_df.select_dtypes(include=[np.number]).columns

        for col in cat_cols:
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col].astype(str))
            self.label_encoders[col] = le

        feature_df[num_cols] = self.imputer.fit_transform(feature_df[num_cols])
        feature_df[num_cols] = self.scaler.fit_transform(feature_df[num_cols])

        self._fitted = True
        return feature_df, y

    def transform(self, df: pd.DataFrame, target_col: str = "churn") -> pd.DataFrame:
        if not self._fitted:
            raise ValueError("Cleaner not fitted. Call fit_transform first.")

        df = self.clean(df)
        y = df[target_col].copy() if target_col in df.columns else None

        feature_df = df.drop(columns=[target_col], errors="ignore")
        cat_cols = [c for c in self.label_encoders]

        for col in cat_cols:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype(str)
                le = self.label_encoders[col]
                feature_df[col] = feature_df[col].map(
                    lambda v: le.transform([v])[0]
                    if v in le.classes_
                    else -1
                )

        num_cols = feature_df.select_dtypes(include=[np.number]).columns
        feature_df[num_cols] = self.imputer.transform(feature_df[num_cols])
        feature_df[num_cols] = self.scaler.transform(feature_df[num_cols])

        return feature_df
