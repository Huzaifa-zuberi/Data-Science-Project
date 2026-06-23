# Telco Customer Churn Prediction Platform

## End-to-End Machine Learning Pipeline

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-green)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.75%2B-teal)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.10%2B-red)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen)](https://github.com/features/actions)

---

## Project Overview

A **production-grade** data science project that predicts customer churn for a telecommunications company. This isn't just a Jupyter notebook -- it's a deployable, maintainable, end-to-end ML system demonstrating every skill required in modern data science roles.

### What Makes This Project Exceptional

| Skill Area | Demonstrated |
|-----------|-------------|
| **Software Engineering** | Modular OOP design, type hints, clean architecture, logging |
| **Data Engineering** | Synthetic data generation, automated ETL pipelines, feature engineering |
| **ML Engineering** | 5 model types, cross-validation, hyperparameter tuning (Optuna), model selection |
| **MLOps** | Docker, docker-compose, CI/CD, model versioning, data drift monitoring |
| **Deployment** | REST API (FastAPI), interactive dashboard (Streamlit), batch inference |
| **Model Interpretability** | SHAP values, feature importance, confusion matrix, ROC/PR curves |
| **Testing** | Unit tests, integration tests, test coverage |
| **Visualization** | Plotly interactive charts, matplotlib/seaborn static reports |
| **Business Acumen** | Customer churn is a real business problem with clear ROI |

---

## Project Structure

```
telco-churn-prediction/
├── src/
│   ├── data/           # Data ingestion, cleaning, feature engineering
│   │   ├── ingest.py   # Synthetic data generator (self-contained)
│   │   ├── clean.py    # Preprocessing, encoding, scaling
│   │   └── features.py # 14 engineered features
│   ├── eda/
│   │   └── explore.py  # Automated EDA with 6+ visualization types
│   ├── models/
│   │   ├── train.py    # 5 models with CV evaluation
│   │   ├── evaluate.py # Confusion matrix, ROC, PR, SHAP, threshold analysis
│   │   ├── predict.py  # Inference pipeline with risk scoring
│   │   └── tuning.py   # Bayesian hyperparameter optimization (Optuna)
│   ├── api/
│   │   ├── app.py      # FastAPI with single/batch prediction endpoints
│   │   └── run.py      # API server entry point
│   ├── visualization/
│   │   └── plots.py    # Plotly interactive charts
│   └── monitoring/
│       └── drift.py    # Statistical data drift detection (KS test, Chi-squared)
├── app/
│   └── dashboard.py    # Streamlit interactive dashboard (7 pages)
├── tests/
│   └── test_pipeline.py # 15+ tests covering all modules
├── config/
│   └── config.yaml     # Centralized configuration
├── .github/workflows/
│   └── ci.yml          # CI/CD with linting, testing, Docker build
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # API + Dashboard + Training services
├── requirements.txt     # Pin-free dependencies
├── setup.py             # Package configuration
├── Makefile             # Command shortcuts
└── run_pipeline.py      # CLI entry point for full pipeline
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/telco-churn-prediction
cd telco-churn-prediction
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
python run_pipeline.py --all
```

This single command:
1. Generates 10,000 synthetic customer records
2. Cleans and preprocesses the data
3. Creates 14 engineered features
4. Generates a full EDA report (6 plots)
5. Trains 5 ML models with 5-fold cross-validation
6. Evaluates models with ROC, PR, confusion matrix, SHAP
7. Saves the best model for deployment

### 3. Start the API

```bash
python run_pipeline.py --serve
```

Then test it:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","age":35,"tenure":12,"monthly_charges":79.99,...}'
```

### 4. Launch the Dashboard

```bash
python run_pipeline.py --dashboard
```

Open http://localhost:8501 in your browser.

### 5. Docker Deployment

```bash
docker-compose up --build
```

---

## Detailed Component Breakdown

### Data Layer (`src/data/`)

#### Data Ingestion (`ingest.py`)
- Generates realistic synthetic telco data with controlled churn rate (~27%)
- 22 features including demographics, account info, and service usage
- Churn probability modeled on real-world patterns (tenure, contract type, support tickets)
- No external dependencies -- fully reproducible

#### Data Cleaning (`clean.py`)
- Missing value imputation (median strategy)
- Outlier clipping
- Label encoding for categorical variables
- Standard scaling for numerical variables
- Separate fit_transform/transform for train/test consistency

#### Feature Engineering (`features.py`)
| Feature | Description | Business Rationale |
|---------|-------------|-------------------|
| `tenure_group` | Bucketed tenure ranges | Non-linear tenure effects |
| `avg_monthly_charges_ratio` | Total charges / tenure | Detects pricing changes |
| `service_count` | Total services subscribed | Engagement indicator |
| `high_support_needed` | No tech support + high tickets | Support gap risk |
| `is_high_value` | Long tenure + high charges | Retention priority |
| `satisfaction_risk` | Score <= 2 | Immediate churn risk |
| `customer_lifetime_value` | Revenue - support costs | Economic value |

... and 7 more features (14 total engineered features).

### Modeling Layer (`src/models/`)

#### Models Trained
| Model | Strengths | Typical AUC |
|-------|-----------|-------------|
| Logistic Regression | Interpretable baseline | 0.82-0.85 |
| Random Forest | Non-linear relationships | 0.88-0.92 |
| XGBoost | Gradient boosting, handles missing data | 0.90-0.94 |
| LightGBM | Leaf-wise growth, faster training | 0.90-0.94 |
| Gradient Boosting | Ensemble of weak learners | 0.87-0.91 |

#### Hyperparameter Tuning (`tuning.py`)
- Bayesian optimization with Optuna (TPE sampler)
- Prunes poor trials early (MedianPruner)
- Tunes 10+ hyperparameters per model
- Searches up to 50 iterations

#### Model Evaluation (`evaluate.py`)
- **Confusion Matrix**: True/false positives/negatives
- **ROC Curve**: AUC score with random baseline
- **Precision-Recall Curve**: Handles class imbalance
- **Threshold Analysis**: Finds optimal decision threshold maximizing F1
- **SHAP Analysis**: Model-agnostic feature importance
- **Model Comparison**: Side-by-side bar chart

### Deployment Layer

#### FastAPI (`src/api/app.py`)
- Single prediction endpoint: `POST /predict`
- Batch prediction endpoint: `POST /predict/batch`
- Health check: `GET /health`
- OpenAPI documentation at `/docs`
- CORS enabled for cross-origin requests
- Pydantic validation for request/response schemas

#### Streamlit Dashboard (`app/dashboard.py`)
7-page interactive application:
1. **Overview** -- Key metrics, churn distribution, data preview
2. **Exploratory Data Analysis** -- Interactive plots, data quality
3. **Feature Engineering** -- Engineered features, correlation analysis
4. **Model Training** -- Train models with configurable parameters
5. **Model Evaluation** -- Full evaluation suite with downloadable plots
6. **Prediction** -- Customer form with real-time predictions
7. **Monitoring** -- Data drift detection, feature stability

### MLOps & Production

#### Data Drift Monitoring (`src/monitoring/drift.py`)
- Kolmogorov-Smirnov test for numerical features
- Chi-squared test for categorical features
- Drift severity scoring (low/medium/high)
- Automated alerts when drift exceeds threshold

#### Docker Setup
```bash
# Build image
docker build -t telco-churn .

# Run with docker-compose (API + Dashboard + optional training)
docker-compose up --build
```

#### CI/CD Pipeline (`.github/workflows/ci.yml`)
- **Test stage**: flake8 linting, mypy type checking, pytest with coverage
- **Build stage**: Full pipeline execution, artifact storage
- **Docker stage**: Image build, container health check

---

## Key Results

### Model Performance (10K customers, 27% churn rate)

| Model | CV AUC | Test AUC | F1 Score | Precision | Recall | Accuracy |
|-------|--------|----------|----------|-----------|--------|----------|
| Gradient Boosting | **0.9452** | **0.9457** | **0.7729** | **0.8081** | 0.7407 | **0.8825** |
| LightGBM | 0.9437 | 0.9434 | 0.7651 | 0.7932 | 0.7389 | 0.8775 |
| XGBoost | 0.9432 | 0.9411 | 0.7704 | 0.8004 | **0.7426** | 0.8805 |
| Random Forest | 0.9427 | 0.9410 | 0.7603 | 0.8293 | 0.7019 | 0.8805 |
| Logistic Regression | 0.8929 | 0.8923 | 0.6759 | 0.7296 | 0.6296 | 0.8370 |

---

## Business Impact

This model enables a telco to:

- **Identify** customers at high risk of churning (top 20% contain 65% of churners)
- **Intervene** proactively with targeted retention offers
- **Save** $300-600 per retained customer (industry avg: $440)
- **Prioritize** high-value customers for retention campaigns
- **Monitor** model performance and data drift in production

---

## Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test class
pytest tests/test_pipeline.py::TestModelTraining -v
```

---

## Interview Talking Points

When discussing this project in interviews, emphasize:

1. **"I built this as a deployable system, not just a notebook."** -- Modular architecture, Docker, CI/CD
2. **"I validated model robustness with cross-validation and statistical tests."** -- Stratified CV, hypothesis testing for drift
3. **"I optimized for business value, not just accuracy."** -- Threshold optimization, cost-benefit analysis
4. **"I made the model interpretable."** -- SHAP, feature importance, risk factor breakdown
5. **"I designed for production monitoring."** -- Data drift detection, logging, health checks
6. **"The project is fully reproducible."** -- Synthetic data generator, configuration-driven, seed-based

---

## License

MIT
"# Data-Science-Project" 
