import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import DataIngestor
from src.data.clean import DataCleaner
from src.data.features import FeatureEngineer
from src.eda.explore import EDAReport
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.visualization.plots import PlotlyVisualizer
from src.monitoring.drift import DataDriftDetector


def load_or_generate_data():
    data_path = Path("data/raw/telco_churn.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    ingestor = DataIngestor()
    df = ingestor.generate_telco_churn_data()
    ingestor.save_data(df, data_path)
    return df


def main():
    st.set_page_config(
        page_title="Telco Churn Prediction Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("📊 Churn Analytics Platform")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Exploratory Data Analysis",
            "Feature Engineering",
            "Model Training",
            "Model Evaluation",
            "Prediction",
            "Monitoring",
        ],
    )

    if "df" not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state.df = load_or_generate_data()

    df = st.session_state.df

    if page == "Overview":
        show_overview(df)
    elif page == "Exploratory Data Analysis":
        show_eda(df)
    elif page == "Feature Engineering":
        show_feature_engineering(df)
    elif page == "Model Training":
        show_model_training(df)
    elif page == "Model Evaluation":
        show_model_evaluation()
    elif page == "Prediction":
        show_prediction(df)
    elif page == "Monitoring":
        show_monitoring(df)


def show_overview(df):
    st.title("📊 Telco Customer Churn Prediction Platform")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        churn_rate = df["churn"].mean()
        st.metric("Churn Rate", f"{churn_rate:.2%}",
                  f"{'🔴' if churn_rate > 0.25 else '🟢'}")
    with col3:
        avg_tenure = df["tenure"].mean()
        st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")
    with col4:
        avg_charges = df["monthly_charges"].mean()
        st.metric("Avg Monthly Charges", f"${avg_charges:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = PlotlyVisualizer.plot_churn_donut(df)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        num_cols = df.select_dtypes(include=[np.number]).columns[:2]
        if len(num_cols) >= 2:
            fig = px.box(df, x="churn", y="monthly_charges",
                         color="churn", title="Monthly Charges by Churn")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Sample")
    st.dataframe(df.head(10), use_container_width=True)


def show_eda(df):
    st.title("🔍 Exploratory Data Analysis")

    st.subheader("Data Quality")
    col1, col2 = st.columns(2)
    with col1:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            fig = px.bar(x=missing.index, y=missing.values,
                         title="Missing Values by Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found!")

    with col2:
        st.write(f"**Rows:** {df.shape[0]:,}")
        st.write(f"**Columns:** {df.shape[1]:,}")
        st.write(f"**Numerical:** {len(df.select_dtypes(include=[np.number]).columns)}")
        st.write(f"**Categorical:** {len(df.select_dtypes(include=['object']).columns)}")

    st.subheader("Numerical Distributions")
    num_cols = df.select_dtypes(include=[np.number]).columns[:4]
    selected_cols = st.multiselect("Select columns", num_cols, default=list(num_cols)[:2])
    if selected_cols:
        fig = go.Figure()
        for col in selected_cols:
            for churn_val, color, label in zip(
                [0, 1], ["#2ecc71", "#e74c3c"], ["Not Churned", "Churned"]
            ):
                subset = df[df["churn"] == churn_val][col].dropna()
                fig.add_trace(go.Histogram(
                    x=subset, name=label, opacity=0.6,
                    marker_color=color, nbinsx=30,
                    histnorm="probability density",
                ))
        fig.update_layout(barmode="overlay", title="Distribution by Churn Status")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorical Analysis")
    cat_cols = df.select_dtypes(include=["object"]).columns
    selected_cat = st.selectbox("Select categorical feature", cat_cols)
    if selected_cat:
        fig = PlotlyVisualizer.plot_churn_by_category(df, selected_cat)
        st.plotly_chart(fig, use_container_width=True)


def show_feature_engineering(df):
    st.title("⚙️ Feature Engineering")

    engineer = FeatureEngineer()
    df_engineered = engineer.create_features(df)

    st.subheader("Engineered Features")
    st.write(f"Created {len(engineer.engineered_cols)} new features:")
    st.write(engineer.engineered_cols)

    st.subheader("Sample of Engineered Features")
    st.dataframe(df_engineered[engineer.engineered_cols].head(10),
                 use_container_width=True)

    fig = PlotlyVisualizer.plot_correlation_heatmap(
        df_engineered[engineer.engineered_cols + ["churn"]]
    )
    st.plotly_chart(fig, use_container_width=True)


def show_model_training(df):
    st.title("🧠 Model Training")

    col1, col2 = st.columns([1, 3])
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("CV Folds", 3, 10, 5)

    from sklearn.model_selection import train_test_split

    cleaner = DataCleaner()
    X, y = cleaner.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training models... This may take a moment."):
            trainer = ModelTrainer()
            results = trainer.train_and_evaluate(
                X_train, y_train, X_test, y_test, cv_folds=cv_folds
            )

            st.session_state.trainer = trainer
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_train = X_train

            results_df = pd.DataFrame(results).T
            st.success("Training complete!")
            st.dataframe(
                results_df.style.highlight_max(axis=0),
                use_container_width=True
            )

            fig = go.Figure()
            metrics = ["test_roc_auc", "test_f1", "test_precision", "test_recall"]
            for metric in metrics:
                fig.add_trace(go.Bar(
                    name=metric.replace("test_", "").replace("_", " ").title(),
                    x=list(results.keys()),
                    y=[results[m][metric] for m in results],
                ))
            fig.update_layout(
                title="Model Performance Comparison",
                barmode="group",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

    if "trainer" in st.session_state and st.session_state.trainer.best_model:
        trainer = st.session_state.trainer
        st.subheader(f"Best Model: **{trainer.best_model_name}**")
        st.json(trainer.results[trainer.best_model_name])

        importance_df = trainer.get_feature_importance(X_train.columns.tolist())
        if not importance_df.empty:
            fig = PlotlyVisualizer.plot_feature_importance(importance_df)
            st.plotly_chart(fig, use_container_width=True)


def show_model_evaluation():
    st.title("📈 Model Evaluation")

    if "trainer" not in st.session_state:
        st.warning("Please train models first in the Model Training section.")
        return

    trainer = st.session_state.trainer
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    evaluator = ModelEvaluator()

    st.subheader("Confusion Matrix")
    evaluator.plot_confusion_matrix(trainer.best_model, X_test, y_test)
    st.image("reports/figures/confusion_matrix.png")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve")
        evaluator.plot_roc_curve(trainer.best_model, X_test, y_test)
        st.image("reports/figures/roc_curve.png")
    with col2:
        st.subheader("Precision-Recall Curve")
        evaluator.plot_precision_recall_curve(trainer.best_model, X_test, y_test)
        st.image("reports/figures/precision_recall_curve.png")

    st.subheader("Classification Report")
    from sklearn.metrics import classification_report
    y_pred = trainer.best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)


def show_prediction(df):
    st.title("🎯 Customer Churn Prediction")

    st.subheader("Enter Customer Details")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", 18, 100, 35)
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        with col2:
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        with col3:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)

        col1, col2, col3 = st.columns(3)
        with col1:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        with col2:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)
        with col3:
            avg_monthly_usage_gb = st.number_input("Avg Monthly Usage (GB)", 0.0, 100.0, 20.0)

        support_tickets = st.number_input("Support Tickets", 0, 20, 1)

        submitted = st.form_submit_button("🔮 Predict Churn", type="primary")

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender, "age": age, "partner": partner,
            "dependents": dependents, "tenure": tenure,
            "phone_service": phone_service, "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security, "online_backup": online_backup,
            "device_protection": device_protection, "tech_support": tech_support,
            "streaming_tv": streaming_tv, "streaming_movies": streaming_movies,
            "contract_type": contract_type, "paperless_billing": paperless_billing,
            "payment_method": payment_method, "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
            "support_tickets": support_tickets,
            "satisfaction_score": satisfaction_score,
        }])

        cleaner = DataCleaner()
        engineer = FeatureEngineer()

        with st.spinner("Making prediction..."):
            customer_with_eng = engineer.create_features(customer)
            X_processed = cleaner.fit_transform(
                pd.concat([customer_with_eng, df.sample(100)], ignore_index=True)
            )[0].iloc[:1]

            if "trainer" in st.session_state:
                model = st.session_state.trainer.best_model
                proba = model.predict_proba(X_processed)[0][1]
                pred = int(proba >= 0.5)

                col1, col2 = st.columns(2)
                with col1:
                    fig = PlotlyVisualizer.plot_risk_gauge(proba)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    risk = "High Risk" if proba >= 0.7 else "Medium Risk" if proba >= 0.3 else "Low Risk"
                    st.markdown(f"""
                    ### Prediction Results
                    - **Prediction:** {'🔴 Will Churn' if pred == 1 else '🟢 Will Stay'}
                    - **Probability:** {proba:.1%}
                    - **Risk Level:** {risk}
                    - **Confidence:** {max(proba, 1-proba):.1%}
                    """)

                st.subheader("Key Risk Indicators")
                risk_indicators = {
                    "Tenure < 12 months": tenure < 12,
                    "Month-to-month contract": contract_type == "Month-to-month",
                    "Fiber optic internet": internet_service == "Fiber optic",
                    "No tech support": tech_support == "No",
                    "Low satisfaction": satisfaction_score <= 2,
                    "High support tickets": support_tickets >= 5,
                    "High monthly charges": monthly_charges > 80,
                    "Electronic check payment": payment_method == "Electronic check",
                }
                risk_df = pd.DataFrame(
                    [{"Risk Factor": k, "Present": "⚠️ Yes" if v else "✅ No"}
                     for k, v in risk_indicators.items()]
                )
                st.dataframe(risk_df, use_container_width=True)


def show_monitoring(df):
    st.title("📡 Model Monitoring & Data Drift")

    st.subheader("Data Drift Detection")
    monitor = DataDriftDetector(threshold=0.05)

    reference = df.sample(frac=0.3, random_state=42)
    monitor.fit_reference(reference)

    current = df.sample(frac=0.3, random_state=99)

    drift_results = monitor.detect_drift(current)

    drift_df = pd.DataFrame(drift_results).T
    drifted_count = drift_df["drifted"].sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Features Checked", len(drift_df))
    with col2:
        st.metric("Drifted Features", f"{drifted_count}")
    with col3:
        st.metric("Drift Rate", f"{drifted_count/len(drift_df):.1%}")

    if drifted_count > 0:
        st.warning(f"⚠️ {drifted_count} features show signs of drift!")
        drifted_df = drift_df[drift_df["drifted"]]
        st.dataframe(drifted_df[["p_value", "drift_severity"]], use_container_width=True)
    else:
        st.success("✅ No significant drift detected in features.")

    st.subheader("Feature Stability")
    fig = go.Figure()
    num_cols = df.select_dtypes(include=[np.number]).columns[:5]
    for col in num_cols:
        fig.add_trace(go.Box(y=df[col], name=col, boxmean="sd"))
    fig.update_layout(title="Feature Distributions (Stability Check)")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
