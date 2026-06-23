# Telco Customer Churn Prediction — Project Summary

## What This Project Is
A production-grade end-to-end Machine Learning system that predicts customer churn for a telecom company. Built to demonstrate every skill required for a data science role: data engineering, ML modeling, MLOps, deployment, testing, and business impact.

## Architecture Overview
```
Synthetic Data Generator → Data Cleaning → Feature Engineering (14 new features)
    → EDA (6 auto plots) → Model Training (5 models) → Evaluation (SHAP, ROC, etc.)
    → Save Best Model → FastAPI (REST API) + Streamlit (Dashboard) → Docker + CI/CD
```

## Key Files (35 total)

### Core ML Pipeline
| File | Purpose |
|------|---------|
| `src/data/ingest.py` | Generates 10K realistic customer records with 27% churn rate |
| `src/data/clean.py` | Handles missing values, encodes categories, scales numbers |
| `src/data/features.py` | Creates 14 engineered features (tenure groups, risk scores, etc.) |
| `src/models/train.py` | Trains 5 models with 5-fold cross-validation |
| `src/models/evaluate.py` | ROC, PR curves, confusion matrix, SHAP, threshold optimization |
| `src/models/predict.py` | Loads saved model for inference with risk scoring |
| `src/models/tuning.py` | Bayesian hyperparameter search with Optuna |
| `src/monitoring/drift.py` | Detects data drift using KS-test and Chi-squared test |
| `src/eda/explore.py` | Generates 6 automated EDA visualizations |
| `src/visualization/plots.py` | Interactive Plotly charts |
| `run_pipeline.py` | CLI entry point: `--all`, `--train`, `--serve`, `--dashboard` |

### Deployment
| File | Purpose |
|------|---------|
| `src/api/app.py` | FastAPI with `/predict`, `/predict/batch`, `/health` |
| `app/dashboard.py` | Streamlit dashboard (7 pages: Overview, EDA, Training, Evaluation, Prediction, Monitoring) |
| `Dockerfile` | Containerizes the entire application |
| `docker-compose.yml` | Orchestrates API + Dashboard + Training services |

### Infrastructure
| File | Purpose |
|------|---------|
| `tests/test_pipeline.py` | 16 unit tests (all pass) |
| `.github/workflows/ci.yml` | CI/CD: lint → test → build → Docker |
| `config/config.yaml` | Centralized config for data paths, model params, monitoring |
| `requirements.txt` | All Python dependencies |
| `Makefile` | Shortcuts: `make train`, `make serve`, `make test` |
| `setup.py` | Package configuration |

## Results (Verified)

| Model | CV AUC | Test AUC | F1 | Precision | Recall |
|-------|--------|----------|-----|-----------|--------|
| Gradient Boosting | **0.945** | **0.946** | **0.773** | 0.808 | 0.741 |
| LightGBM | 0.944 | 0.943 | 0.765 | 0.793 | 0.739 |
| XGBoost | 0.943 | 0.941 | 0.770 | 0.800 | **0.743** |
| Random Forest | 0.943 | 0.941 | 0.760 | **0.829** | 0.702 |
| Logistic Regression | 0.893 | 0.892 | 0.676 | 0.730 | 0.630 |

**Tests:** 16/16 passed (data ingestion, cleaning, features, training, drift detection)

## How to Run
```bash
cd data-science-portfolio
pip install -r requirements.txt
python run_pipeline.py --all          # Full pipeline
python run_pipeline.py --serve        # API at localhost:8000
python run_pipeline.py --dashboard    # Dashboard at localhost:8501
python -m pytest tests/ -v            # Run tests
```

## Interview Talking Points
1. **"Built as a deployable system, not a notebook"** — OOP, Docker, CI/CD, type hints, logging
2. **"Validated with cross-validation + statistical tests"** — 5-fold CV, data drift monitoring
3. **"Optimized for business value, not just accuracy"** — Threshold optimization maximizes F1
4. **"Made the model interpretable"** — SHAP, feature importance, risk factor breakdown
5. **"Designed for production monitoring"** — Drift detection, health checks, model versioning
6. **"Fully reproducible"** — Synthetic data generator, seeded random, config-driven

## What Makes This Exceptional
- **Self-contained** — generates its own data, no external downloads needed
- **15+ files of clean, modular Python** — real software engineering
- **5 different ML algorithms** — demonstrates depth of knowledge
- **API + Dashboard** — shows full-stack data science capability
- **Tests + Docker + CI/CD** — proves MLOps readiness
