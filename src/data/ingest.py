import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path


class DataIngestor:
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def generate_telco_churn_data(
        self, n_samples: int = 10000, churn_rate: float = 0.27
    ) -> pd.DataFrame:
        churned = np.zeros(n_samples, dtype=bool)
        n_churn = int(n_samples * churn_rate)
        churned_idx = self.rng.choice(n_samples, n_churn, replace=False)
        churned[churned_idx] = True

        tenure = self.rng.integers(1, 72, n_samples)
        monthly_charges = self.rng.uniform(18.0, 120.0, n_samples)
        satisfaction = self.rng.integers(1, 6, n_samples)
        support_tickets = self.rng.integers(0, 10, n_samples)

        noise_scale = 0.3
        tenure[churned] = np.clip(tenure[churned] * (0.3 + self.rng.random(n_churn) * noise_scale), 1, 72).astype(int)
        monthly_charges[churned] = monthly_charges[churned] * (1.3 - self.rng.random(n_churn) * noise_scale)
        satisfaction[churned] = np.clip(satisfaction[churned] - 1 + self.rng.integers(-1, 2, n_churn), 1, 5)
        support_tickets[churned] = np.clip(support_tickets[churned] + 2 - self.rng.integers(0, 3, n_churn), 0, 9)

        contract_churned = self.rng.choice(
            ["Month-to-month", "One year", "Two year"], n_churn, p=[0.8, 0.15, 0.05]
        )
        contract_stayed = self.rng.choice(
            ["Month-to-month", "One year", "Two year"], n_samples - n_churn, p=[0.4, 0.3, 0.3]
        )
        all_contracts = np.empty(n_samples, dtype=object)
        all_contracts[churned] = contract_churned
        all_contracts[~churned] = contract_stayed

        all_tech_support = np.empty(n_samples, dtype=object)
        all_tech_support[churned] = self.rng.choice(
            ["Yes", "No", "No internet service"], n_churn, p=[0.1, 0.7, 0.2]
        )
        all_tech_support[~churned] = self.rng.choice(
            ["Yes", "No", "No internet service"], n_samples - n_churn, p=[0.3, 0.5, 0.2]
        )

        all_internet = np.empty(n_samples, dtype=object)
        all_internet[churned] = self.rng.choice(
            ["DSL", "Fiber optic", "No"], n_churn, p=[0.25, 0.65, 0.1]
        )
        all_internet[~churned] = self.rng.choice(
            ["DSL", "Fiber optic", "No"], n_samples - n_churn, p=[0.45, 0.25, 0.3]
        )

        all_payment = np.empty(n_samples, dtype=object)
        all_payment[churned] = self.rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            n_churn, p=[0.6, 0.2, 0.1, 0.1]
        )
        all_payment[~churned] = self.rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            n_samples - n_churn, p=[0.2, 0.3, 0.25, 0.25]
        )

        data = {
            "customer_id": [f"CUST-{i:06d}" for i in range(n_samples)],
            "gender": self.rng.choice(["Male", "Female"], n_samples),
            "age": self.rng.integers(18, 80, n_samples),
            "partner": self.rng.choice(["Yes", "No"], n_samples, p=[0.48, 0.52]),
            "dependents": self.rng.choice(["Yes", "No"], n_samples, p=[0.3, 0.7]),
            "tenure": tenure,
            "phone_service": self.rng.choice(["Yes", "No"], n_samples, p=[0.9, 0.1]),
            "multiple_lines": self.rng.choice(
                ["Yes", "No", "No phone service"], n_samples, p=[0.4, 0.4, 0.2]
            ),
            "internet_service": all_internet,
            "online_security": self.rng.choice(
                ["Yes", "No", "No internet service"], n_samples, p=[0.25, 0.55, 0.2]
            ),
            "online_backup": self.rng.choice(
                ["Yes", "No", "No internet service"], n_samples, p=[0.3, 0.5, 0.2]
            ),
            "device_protection": self.rng.choice(
                ["Yes", "No", "No internet service"], n_samples, p=[0.3, 0.5, 0.2]
            ),
            "tech_support": all_tech_support,
            "streaming_tv": self.rng.choice(
                ["Yes", "No", "No internet service"], n_samples, p=[0.35, 0.45, 0.2]
            ),
            "streaming_movies": self.rng.choice(
                ["Yes", "No", "No internet service"], n_samples, p=[0.35, 0.45, 0.2]
            ),
            "contract_type": all_contracts.astype(str),
            "paperless_billing": self.rng.choice(["Yes", "No"], n_samples, p=[0.6, 0.4]),
            "payment_method": all_payment,
            "monthly_charges": monthly_charges,
            "total_charges": np.zeros(n_samples),
            "avg_monthly_usage_gb": self.rng.uniform(0.5, 50.0, n_samples),
            "support_tickets": support_tickets,
            "satisfaction_score": satisfaction,
        }

        df = pd.DataFrame(data)
        df["total_charges"] = df["tenure"] * df["monthly_charges"] * self.rng.uniform(
            0.85, 1.15, n_samples
        )

        df["churn"] = churned.astype(int)
        df = self._add_missing_values(df)
        return df

    def _add_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_perturb = [c for c in df.columns if c not in ("customer_id", "churn")]
        n_rows = len(df)
        n_cols = len(cols_to_perturb)
        missing_mask = self.rng.random((n_rows, n_cols)) < 0.005
        df = df.copy()
        for i, col in enumerate(cols_to_perturb):
            df.loc[missing_mask[:, i], col] = np.nan
        return df

    def save_data(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def load_data(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)
