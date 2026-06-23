import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score
from typing import Any, Dict
from plotly.subplots import make_subplots
from typing import Any, Dict, Optional


class PlotlyVisualizer:
    @staticmethod
    def plot_churn_donut(df: pd.DataFrame, target_col: str = "churn"):
        counts = df[target_col].value_counts()
        fig = go.Figure(data=[
            go.Pie(
                labels=["Not Churned", "Churned"],
                values=counts.values,
                hole=0.6,
                marker=dict(colors=["#2ecc71", "#e74c3c"]),
                textinfo="label+percent",
                textfont=dict(size=14),
            )
        ])
        fig.update_layout(
            title="Customer Churn Distribution",
            annotations=[dict(
                text=f"{counts.get(1, 0)}<br>Churned",
                x=0.5, y=0.5, font_size=16, showarrow=False
            )],
            height=400,
        )
        return fig

    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15):
        df = importance_df.head(top_n)
        colors = px.colors.sequential.Viridis_r[:len(df)]

        fig = go.Figure(go.Bar(
            x=df["importance_pct"],
            y=df["feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.3)", width=1)),
            text=df["importance_pct"].round(2).astype(str) + "%",
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Relative Importance (%)",
            yaxis=dict(autorange="reversed"),
            height=500,
            margin=dict(l=200),
        )
        return fig

    @staticmethod
    def plot_roc_curves(models_results: Dict[str, Any], X_test, y_test):
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        for i, (name, model) in enumerate(models_results.items()):
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} (AUC={auc:.3f})",
                    line=dict(color=colors[i % len(colors)], width=2),
                ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random (AUC=0.50)",
            line=dict(color="gray", dash="dash", width=1),
        ))
        fig.update_layout(
            title="ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.7, y=0.1),
            height=500,
        )
        return fig

    @staticmethod
    def plot_churn_by_category(df: pd.DataFrame, cat_col: str):
        grouped = df.groupby(cat_col)["churn"].agg(["mean", "count"]).reset_index()
        grouped.columns = [cat_col, "churn_rate", "count"]
        grouped = grouped.sort_values("churn_rate")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Churn Rate by {cat_col}", "Customer Count"),
            column_widths=[0.7, 0.3],
        )

        fig.add_trace(
            go.Bar(
                x=grouped["churn_rate"],
                y=grouped[cat_col],
                orientation="h",
                marker=dict(
                    color=grouped["churn_rate"],
                    colorscale=[[0, "#2ecc71"], [1, "#e74c3c"]],
                    showscale=False,
                ),
                text=grouped["churn_rate"].round(3),
                textposition="outside",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Bar(
                x=grouped["count"],
                y=grouped[cat_col],
                orientation="h",
                marker=dict(color="rgba(52, 152, 219, 0.7)"),
                text=grouped["count"],
                textposition="outside",
            ),
            row=1, col=2,
        )

        fig.update_layout(
            title=f"Churn Analysis by {cat_col}",
            height=400,
            showlegend=False,
        )
        return fig

    @staticmethod
    def plot_risk_gauge(probability: float):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            title={"text": "Churn Risk Score"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": probability * 100,
                },
            },
        ))
        fig.update_layout(height=400)
        return fig

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame):
        num_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[num_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
        ))
        fig.update_layout(
            title="Feature Correlation Heatmap",
            height=800,
            width=900,
        )
        return fig


