import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from scipy.stats import ks_2samp, chi2_contingency

logger = logging.getLogger(__name__)


class DataDriftDetector:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.reference_stats: Dict[str, dict] = {}

    def fit_reference(self, df: pd.DataFrame):
        self.reference_stats = {}
        for col in df.columns:
            col_data = df[col].dropna()
            is_num = pd.api.types.is_numeric_dtype(col_data)
            stats = {
                "dtype": str(col_data.dtype),
                "mean": col_data.mean() if is_num else None,
                "std": col_data.std() if is_num else None,
                "distribution": col_data.value_counts(normalize=True).to_dict()
                if not is_num else None,
            }
            self.reference_stats[col] = stats
        logger.info(f"Reference fitted on {len(df)} samples, {len(df.columns)} features")

    def detect_drift(self, current_df: pd.DataFrame) -> Dict[str, dict]:
        if not self.reference_stats:
            raise ValueError("Reference not fitted. Call fit_reference first.")

        drift_results = {}
        for col in current_df.columns:
            if col not in self.reference_stats:
                continue

            current_data = current_df[col].dropna()
            ref_info = self.reference_stats[col]

            if pd.api.types.is_numeric_dtype(current_data):
                p_value = self._ks_test(ref_info, current_data)
            else:
                p_value = self._chi_squared_test(ref_info, current_data)

            drifted = p_value < self.threshold
            drift_results[col] = {
                "drifted": drifted,
                "p_value": p_value,
                "drift_severity": "high" if p_value < self.threshold / 10
                else "medium" if drifted
                else "low",
            }

            if drifted:
                logger.warning(
                    f"Drift detected in '{col}' (p-value={p_value:.6f}, severity={drift_results[col]['drift_severity']})"
                )

        drift_count = sum(1 for v in drift_results.values() if v["drifted"])
        logger.info(
            f"Drift check complete: {drift_count}/{len(drift_results)} features drifted"
        )
        return drift_results

    def _ks_test(self, ref_info: dict, current_data: pd.Series) -> float:
        ref_mean = ref_info["mean"]
        ref_std = ref_info["std"]
        if ref_mean is None or ref_std is None or ref_std == 0:
            return 1.0

        ref_data = np.random.normal(ref_mean, ref_std, size=len(current_data))
        _, p_value = ks_2samp(ref_data, current_data)
        return p_value

    def _chi_squared_test(self, ref_info: dict, current_data: pd.Series) -> float:
        ref_dist = ref_info["distribution"]
        if ref_dist is None:
            return 1.0

        current_dist = current_data.value_counts(normalize=True).to_dict()
        all_categories = set(list(ref_dist.keys()) + list(current_dist.keys()))

        ref_counts = [int(ref_dist.get(cat, 0) * len(current_data)) for cat in all_categories]
        current_counts = [int(current_dist.get(cat, 0) * len(current_data)) for cat in all_categories]

        if len(all_categories) < 2:
            return 1.0

        try:
            _, p_value, _, _ = chi2_contingency([ref_counts, current_counts])
            return p_value
        except ValueError:
            return 1.0
