import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


class EDAReport:
    def __init__(self, df: pd.DataFrame, target_col: str = "churn"):
        self.df = df
        self.target = target_col
        self.report_dir = Path("reports/figures")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(self) -> dict:
        summary = {
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing": self.df.isnull().sum().to_dict(),
            "target_distribution": self.df[self.target].value_counts().to_dict(),
        }

        self._plot_target_distribution()
        self._plot_correlation_matrix()
        self._plot_numerical_distributions()
        self._plot_categorical_analysis()
        self._plot_churn_by_features()

        return summary

    def _plot_target_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        colors = ["#2ecc71", "#e74c3c"]

        self.df[self.target].value_counts().plot(
            kind="bar", ax=axes[0], color=colors, edgecolor="black"
        )
        axes[0].set_title("Target Distribution", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Churn")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=0)

        self.df[self.target].value_counts().plot(
            kind="pie", ax=axes[1], autopct="%1.1f%%",
            colors=colors, explode=(0, 0.05)
        )
        axes[1].set_title("Churn Proportion", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("")

        plt.tight_layout()
        plt.savefig(self.report_dir / "target_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_correlation_matrix(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) < 2:
            return

        corr = self.df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.report_dir / "correlation_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_numerical_distributions(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns[:8]
        n_cols = 4
        n_rows = (len(num_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i, col in enumerate(num_cols):
            for churn_val, color, label in zip(
                [0, 1], ["#2ecc71", "#e74c3c"], ["Not Churned", "Churned"]
            ):
                subset = self.df[self.df[self.target] == churn_val][col].dropna()
                if len(subset) > 0:
                    sns.kdeplot(
                        subset, ax=axes[i], color=color, label=label,
                        fill=True, alpha=0.3
                    )
            axes[i].set_title(f"{col} by Churn", fontsize=11, fontweight="bold")
            axes[i].legend(fontsize=8)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(self.report_dir / "numerical_distributions.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_categorical_analysis(self):
        cat_cols = self.df.select_dtypes(include=["object"]).columns[:6]
        n_cols = 3
        n_rows = (len(cat_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(cat_cols):
            ct = pd.crosstab(
                self.df[col], self.df[self.target], normalize="index"
            )
            ct.plot(kind="barh", stacked=True, ax=axes[i], color=["#2ecc71", "#e74c3c"])
            axes[i].set_title(f"Churn Rate by {col}", fontsize=12, fontweight="bold")
            axes[i].set_xlabel("Proportion")
            axes[i].legend(loc="lower right", fontsize=8)
            axes[i].tick_params(axis="y", labelsize=8)

            for container in axes[i].containers:
                axes[i].bar_label(
                    container, fmt="%.2f", fontsize=7, label_type="center"
                )

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(self.report_dir / "categorical_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_churn_by_features(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        churn_cols = [
            ("tenure", "hist", "Churn by Tenure"),
            ("monthly_charges", "box", "Churn by Monthly Charges"),
            ("satisfaction_score", "bar", "Churn by Satisfaction Score"),
            ("contract_type", "bar", "Churn by Contract Type"),
            ("payment_method", "bar", "Churn by Payment Method"),
            ("support_tickets", "bar", "Churn by Support Tickets"),
        ]

        for ax, (col, plot_type, title) in zip(axes.flatten(), churn_cols):
            if col not in self.df.columns:
                ax.text(0.5, 0.5, f"{col} not found", ha="center", va="center")
                ax.set_title(title)
                continue

            if plot_type == "hist":
                for val, color, label in zip(
                    [0, 1], ["#2ecc71", "#e74c3c"], ["Not Churned", "Churned"]
                ):
                    subset = self.df[self.df[self.target] == val][col].dropna()
                    ax.hist(subset, bins=30, alpha=0.6, color=color, label=label)
                ax.legend()

            elif plot_type == "box":
                self.df.boxplot(column=col, by=self.target, ax=ax)
                ax.set_title("")

            elif plot_type == "bar":
                grouped = self.df.groupby(col)[self.target].mean().sort_values()
                grouped.plot(kind="barh", ax=ax, color=["#2ecc71" if v < 0.5 else "#e74c3c" for v in grouped])
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.1f%%", fontsize=8)

            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Churn Rate" if plot_type == "bar" else "")

        plt.tight_layout()
        plt.savefig(self.report_dir / "churn_by_features.png", dpi=150, bbox_inches="tight")
        plt.close()
