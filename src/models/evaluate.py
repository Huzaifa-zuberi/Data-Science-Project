import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from typing import Any, Dict, Optional
from sklearn.metrics import (
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve,
    RocCurveDisplay, PrecisionRecallDisplay,
    precision_score, recall_score, f1_score,
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, report_dir: Path = Path("reports/figures")):
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            cmap="Blues", ax=ax,
            display_labels=["Not Churned", "Churned"],
            values_format="d"
        )
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.report_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_roc_curve(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_estimator(
            model, X_test, y_test, ax=ax
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.50)")
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.report_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_precision_recall_curve(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.report_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_shap_analysis(
        self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame
    ):
        try:
            if hasattr(model, "predict"):
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                shap.summary_plot(
                    shap_values, X_test, show=False, max_display=15,
                    plot_size=(8, 5)
                )
                plt.tight_layout()
                plt.savefig(
                    self.report_dir / "shap_summary.png",
                    dpi=150, bbox_inches="tight"
                )
                plt.close()

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.bar(shap_values, max_display=15, show=False)
                plt.tight_layout()
                plt.savefig(
                    self.report_dir / "shap_importance.png",
                    dpi=150, bbox_inches="tight"
                )
                plt.close()

        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")

    def plot_models_comparison(self, results: Dict[str, Dict[str, float]]):
        metrics = ["test_roc_auc", "test_f1", "test_precision", "test_recall"]
        df = pd.DataFrame(results).T[metrics]
        df.columns = ["ROC AUC", "F1", "Precision", "Recall"]

        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black", width=0.8)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.legend(loc="lower left", bbox_to_anchor=(1, 0))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_ylim(0, 1)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=7, padding=1)

        plt.tight_layout()
        plt.savefig(self.report_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_threshold_analysis(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ):
        y_proba = model.predict_proba(X_test)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 50)

        metrics = {"threshold": [], "precision": [], "recall": [], "f1": []}
        for t in thresholds:
            y_pred_t = (y_proba >= t).astype(int)
            metrics["threshold"].append(t)
            metrics["precision"].append(
                precision_score(y_test, y_pred_t, zero_division=0)
            )
            metrics["recall"].append(
                recall_score(y_test, y_pred_t, zero_division=0)
            )
            metrics["f1"].append(
                f1_score(y_test, y_pred_t, zero_division=0)
            )

        df = pd.DataFrame(metrics)
        best_idx = df["f1"].idxmax()
        best_threshold = df.loc[best_idx, "threshold"]

        fig, ax = plt.subplots(figsize=(10, 6))
        for metric in ["precision", "recall", "f1"]:
            ax.plot(
                df["threshold"], df[metric],
                label=metric.capitalize(), linewidth=2
            )

        ax.axvline(
            best_threshold, color="red", linestyle="--", alpha=0.7,
            label=f"Best Threshold={best_threshold:.2f} (F1={df.loc[best_idx, 'f1']:.3f})"
        )
        ax.set_title("Threshold Optimization Analysis", fontsize=14, fontweight="bold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.report_dir / "threshold_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

        return best_threshold



