import numpy as np
import pandas as pd
from typing import List


class FeatureEngineer:
    def __init__(self):
        self.engineered_cols: List[str] = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["tenure_group"] = pd.cut(
            df["tenure"].fillna(0),
            bins=[0, 6, 12, 24, 48, 100],
            labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"],
        )
        df["tenure_group"] = df["tenure_group"].cat.add_categories("unknown").fillna("unknown")

        df["avg_monthly_charges_ratio"] = np.where(
            (df["tenure"] > 0) & df["total_charges"].notna(),
            df["total_charges"].fillna(0) / df["tenure"],
            0
        )

        df["charge_to_usage_ratio"] = np.where(
            df["avg_monthly_usage_gb"].fillna(0) > 0,
            df["monthly_charges"].fillna(0) / df["avg_monthly_usage_gb"].fillna(1),
            0
        )

        df["has_multiple_services"] = (
            (df["phone_service"].fillna("No") == "Yes")
            & (df["internet_service"].fillna("No") != "No")
        ).astype(int)

        df["service_count"] = (
            (df["online_security"].fillna("No") == "Yes").astype(int)
            + (df["online_backup"].fillna("No") == "Yes").astype(int)
            + (df["device_protection"].fillna("No") == "Yes").astype(int)
            + (df["tech_support"].fillna("No") == "Yes").astype(int)
            + (df["streaming_tv"].fillna("No") == "Yes").astype(int)
            + (df["streaming_movies"].fillna("No") == "Yes").astype(int)
        )

        df["high_support_needed"] = (
            (df["tech_support"].fillna("No") == "No")
            & (df["support_tickets"].fillna(0) > 3)
        ).astype(int)

        df["is_high_value"] = (
            (df["tenure"].fillna(0) > 24)
            & (df["monthly_charges"].fillna(0) > 70)
        ).astype(int)

        df["is_short_tenure_high_charge"] = (
            (df["tenure"].fillna(0) < 12)
            & (df["monthly_charges"].fillna(0) > 80)
        ).astype(int)

        df["satisfaction_risk"] = (df["satisfaction_score"].fillna(3) <= 2).astype(int)

        df["avg_charges_per_ticket"] = np.where(
            df["support_tickets"].fillna(0) > 0,
            df["monthly_charges"].fillna(0) / (df["support_tickets"].fillna(0) + 1),
            0
        )

        df["customer_lifetime_value"] = (
            df["monthly_charges"].fillna(0) * df["tenure"].fillna(0)
            - df["support_tickets"].fillna(0) * 10
        )

        df["senior_without_support"] = (
            (df["age"].fillna(0) >= 65)
            & (df["tech_support"].fillna("No") == "No")
            & (df["internet_service"].fillna("No") != "No")
        ).astype(int)

        df["family_plan"] = (
            (df["partner"].fillna("No") == "Yes")
            | (df["dependents"].fillna("No") == "Yes")
        ).astype(int)

        df["is_autopay"] = df["payment_method"].fillna("").str.contains(
            "automatic", case=False
        ).astype(int)

        self.engineered_cols = [
            "tenure_group",
            "avg_monthly_charges_ratio",
            "charge_to_usage_ratio",
            "has_multiple_services",
            "service_count",
            "high_support_needed",
            "is_high_value",
            "is_short_tenure_high_charge",
            "satisfaction_risk",
            "avg_charges_per_ticket",
            "customer_lifetime_value",
            "senior_without_support",
            "family_plan",
            "is_autopay",
        ]

        return df
