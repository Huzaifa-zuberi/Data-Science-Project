#!/usr/bin/env python3
"""
End-to-end ML Pipeline for Telco Customer Churn Prediction.

Usage:
    python run_pipeline.py --generate-data --train --evaluate --serve
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger("pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Telco Churn ML Pipeline")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data")
    parser.add_argument("--clean", action="store_true", help="Clean and preprocess data")
    parser.add_argument("--features", action="store_true", help="Engineer features")
    parser.add_argument("--eda", action="store_true", help="Generate EDA report")
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--tune", action="store_true", help="Hyperparameter tuning")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate and plot results")
    parser.add_argument("--save-model", action="store_true", help="Save best model")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--dashboard", action="store_true", help="Start Streamlit dashboard")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    return parser.parse_args()


def step_generate_data():
    from src.data.ingest import DataIngestor
    logger.info("Step 1: Generating synthetic telco churn data...")
    ingestor = DataIngestor()
    df = ingestor.generate_telco_churn_data(n_samples=10000, churn_rate=0.27)
    ingestor.save_data(df, Path("data/raw/telco_churn.csv"))
    logger.info(f"Generated {len(df)} records with churn rate {df['churn'].mean():.2%}")
    return df


def step_clean_data(df: pd.DataFrame):
    from src.data.clean import DataCleaner
    logger.info("Step 2: Cleaning and preprocessing data...")
    cleaner = DataCleaner()
    X, y = cleaner.fit_transform(df)
    logger.info(f"Cleaned: {X.shape[0]} samples, {X.shape[1]} features")
    joblib.dump(cleaner, Path("models/saved/preprocessor.pkl"))
    logger.info("Preprocessor saved to models/saved/preprocessor.pkl")
    return X, y, cleaner


def step_feature_engineering(df: pd.DataFrame):
    from src.data.features import FeatureEngineer
    logger.info("Step 3: Engineering features...")
    engineer = FeatureEngineer()
    df_fe = engineer.create_features(df)
    logger.info(f"Created {len(engineer.engineered_cols)} new features: {engineer.engineered_cols}")
    return df_fe


def step_eda(df: pd.DataFrame):
    from src.eda.explore import EDAReport
    logger.info("Step 4: Generating EDA report...")
    report = EDAReport(df)
    summary = report.generate_full_report()
    logger.info(f"EDA report generated in reports/figures/")
    logger.info(f"Data shape: {summary['shape']}")
    logger.info(f"Target distribution: {summary['target_distribution']}")
    return summary


def step_train_models(X_train, y_train, X_test, y_test, cv_folds=5):
    from src.models.train import ModelTrainer
    logger.info("Step 5: Training models...")
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test, cv_folds)
    logger.info(f"Best model: {trainer.best_model_name}")
    logger.info(f"Results: {results}")
    return trainer, results


def step_hyperparameter_tuning(X_train, y_train):
    from src.models.tuning import HyperparameterTuner
    logger.info("Step 5b: Hyperparameter tuning with Optuna...")
    tuner = HyperparameterTuner(n_trials=30)
    best_xgb = tuner.tune_xgboost(X_train, y_train)
    logger.info(f"Tuned XGBoost best params: {tuner.best_params.get('xgboost', {})}")
    return best_xgb, tuner


def step_evaluate(trainer, X_test, y_test, X_train):
    from src.models.evaluate import ModelEvaluator
    logger.info("Step 6: Evaluating model performance...")
    evaluator = ModelEvaluator()

    evaluator.plot_confusion_matrix(trainer.best_model, X_test, y_test)
    evaluator.plot_roc_curve(trainer.best_model, X_test, y_test)
    evaluator.plot_precision_recall_curve(trainer.best_model, X_test, y_test)
    evaluator.plot_models_comparison(trainer.results)

    best_threshold = evaluator.plot_threshold_analysis(trainer.best_model, X_test, y_test)
    logger.info(f"Optimal threshold: {best_threshold:.3f}")

    evaluator.plot_shap_analysis(trainer.best_model, X_train, X_test)

    logger.info("All evaluation plots saved to reports/figures/")
    return evaluator


def step_save_model(trainer):
    logger.info("Saving best model...")
    trainer.save_model(Path("models/saved/best_model.pkl"))
    logger.info(f"Best model ({trainer.best_model_name}) saved to models/saved/best_model.pkl")
    return trainer


def step_serve():
    logger.info("Starting API server...")
    import uvicorn
    from src.api.app import create_app
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


def step_dashboard():
    logger.info("Starting Streamlit dashboard...")
    import subprocess
    subprocess.run(["streamlit", "run", "app/dashboard.py"])


def main():
    args = parse_args()
    df = None
    X, y = None, None
    cleaner = None

    if args.all:
        args.generate_data = True
        args.clean = True
        args.features = True
        args.eda = True
        args.train = True
        args.evaluate = True
        args.save_model = True

    if args.generate_data or args.clean or args.features or args.eda:
        df = step_generate_data()

    if args.clean:
        X, y, cleaner = step_clean_data(df)

    if args.features and df is not None:
        df = step_feature_engineering(df)

    if args.eda and df is not None:
        step_eda(df)

    if args.train:
        if X is None:
            df = pd.read_csv("data/raw/telco_churn.csv")
            X, y, cleaner = step_clean_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        trainer, results = step_train_models(X_train, y_train, X_test, y_test)

        if args.tune:
            best_xgb, tuner = step_hyperparameter_tuning(X_train, y_train)

        if args.evaluate:
            step_evaluate(trainer, X_test, y_test, X_train)

        if args.save_model:
            trainer.save_model(Path("models/saved/best_model.pkl"))
            if cleaner:
                joblib.dump(cleaner, Path("models/saved/preprocessor.pkl"))

    if args.serve:
        step_serve()

    if args.dashboard:
        step_dashboard()

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
