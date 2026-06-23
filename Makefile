.PHONY: help install generate clean eda train tune evaluate serve dashboard test docker-build docker-run all

help:
	@echo "Telco Churn Prediction Pipeline"
	@echo "================================"
	@echo "install      - Install dependencies"
	@echo "generate     - Generate synthetic data"
	@echo "clean        - Clean and preprocess data"
	@echo "features     - Engineer features"
	@echo "eda          - Generate EDA report"
	@echo "train        - Train all models"
	@echo "tune         - Hyperparameter tuning with Optuna"
	@echo "evaluate     - Evaluate models and generate plots"
	@echo "save         - Save best model"
	@echo "serve        - Start FastAPI server"
	@echo "dashboard    - Start Streamlit dashboard"
	@echo "test         - Run tests"
	@echo "docker-build - Build Docker image"
	@echo "docker-run   - Run with docker-compose"
	@echo "all          - Run complete pipeline"

install:
	pip install -r requirements.txt

generate:
	python run_pipeline.py --generate-data

clean:
	python run_pipeline.py --generate-data --clean

eda:
	python run_pipeline.py --generate-data --clean --features --eda

train:
	python run_pipeline.py --generate-data --clean --features --train

tune:
	python run_pipeline.py --generate-data --clean --train --tune

evaluate:
	python run_pipeline.py --generate-data --clean --train --evaluate

save:
	python run_pipeline.py --generate-data --clean --train --evaluate --save-model

serve:
	uvicorn src.api.run:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run app/dashboard.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

docker-build:
	docker build -t telco-churn-pipeline .

docker-run:
	docker-compose up --build

all:
	python run_pipeline.py --all
