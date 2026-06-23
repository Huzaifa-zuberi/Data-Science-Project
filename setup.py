from setuptools import setup, find_packages

setup(
    name="telco-churn-prediction",
    version="1.0.0",
    description="End-to-end Telco Customer Churn Prediction Pipeline",
    author="Data Science Portfolio",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "optuna>=3.0.0",
        "shap>=0.40.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.3.0",
        "fastapi>=0.75.0",
        "uvicorn[standard]>=0.17.0",
        "streamlit>=1.10.0",
        "pytest>=7.0.0",
    ],
)
