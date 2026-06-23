import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import DataIngestor
from src.data.clean import DataCleaner
from src.data.features import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.predict import Predictor
from src.monitoring.drift import DataDriftDetector


@pytest.fixture(scope="session")
def sample_data():
    ingestor = DataIngestor(random_seed=42)
    df = ingestor.generate_telco_churn_data(n_samples=500, churn_rate=0.3)
    return df


@pytest.fixture(scope="session")
def cleaned_data(sample_data):
    cleaner = DataCleaner()
    X, y = cleaner.fit_transform(sample_data)
    return X, y, cleaner


class TestDataIngestion:
    def test_generate_data_shape(self, sample_data):
        assert sample_data.shape[0] == 500
        assert "customer_id" in sample_data.columns
        assert "churn" in sample_data.columns

    def test_churn_rate(self, sample_data):
        churn_rate = sample_data["churn"].mean()
        assert 0.15 <= churn_rate <= 0.50

    def test_no_duplicate_ids(self, sample_data):
        assert sample_data["customer_id"].is_unique

    def test_data_persist(self, sample_data):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            ingestor = DataIngestor()
            ingestor.save_data(sample_data, Path(f.name))
            loaded = ingestor.load_data(Path(f.name))
            pd.testing.assert_frame_equal(sample_data, loaded)


class TestDataCleaning:
    def test_cleaner_output_shape(self, cleaned_data):
        X, y, cleaner = cleaned_data
        assert X.shape[0] == 500
        assert "customer_id" not in X.columns

    def test_all_numerical_output(self, cleaned_data):
        X, y, cleaner = cleaned_data
        assert all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number)))

    def test_no_missing_values(self, cleaned_data):
        X, y, cleaner = cleaned_data
        assert X.isnull().sum().sum() == 0

    def test_y_alignment(self, cleaned_data):
        X, y, cleaner = cleaned_data
        assert len(y) == 500
        assert set(y.unique()) <= {0, 1}


class TestFeatureEngineering:
    def test_feature_count(self, sample_data):
        engineer = FeatureEngineer()
        df = engineer.create_features(sample_data)
        assert len(engineer.engineered_cols) > 0

    def test_engineered_features_present(self, sample_data):
        engineer = FeatureEngineer()
        df = engineer.create_features(sample_data)
        for col in engineer.engineered_cols:
            assert col in df.columns

    def test_no_nan_in_engineered(self, sample_data):
        engineer = FeatureEngineer()
        df = engineer.create_features(sample_data)
        for col in engineer.engineered_cols:
            assert df[col].isnull().sum() == 0, f"NaN in {col}"


class TestModelTraining:
    def test_model_training(self, cleaned_data):
        X, y, cleaner = cleaned_data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        trainer = ModelTrainer()
        results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test, cv_folds=3)

        assert len(results) > 0
        assert trainer.best_model is not None
        assert trainer.best_model_name is not None

    def test_model_accuracy_above_baseline(self, cleaned_data):
        X, y, cleaner = cleaned_data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        trainer = ModelTrainer()
        results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test, cv_folds=3)

        churn_rate = y_test.mean()
        best_score = max(
            r["test_roc_auc"] for r in results.values()
        )
        assert best_score > churn_rate + 0.1

    def test_feature_importance(self, cleaned_data):
        X, y, cleaner = cleaned_data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        trainer = ModelTrainer()
        trainer.train_and_evaluate(X_train, y_train, X_test, y_test, cv_folds=3)

        importance = trainer.get_feature_importance(X_train.columns.tolist())
        assert len(importance) > 0
        assert "feature" in importance.columns
        assert "importance" in importance.columns


class TestDriftDetection:
    def test_drift_detection(self, sample_data):
        detector = DataDriftDetector(threshold=0.05)
        detector.fit_reference(sample_data)
        results = detector.detect_drift(sample_data)
        assert len(results) > 0

    def test_no_drift_on_same_data(self, sample_data):
        detector = DataDriftDetector(threshold=0.05)
        detector.fit_reference(sample_data)
        results = detector.detect_drift(sample_data)
        drifted = sum(1 for v in results.values() if v["drifted"])
        assert drifted < len(results) / 2
