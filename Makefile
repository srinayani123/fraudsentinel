.PHONY: help install download features train-xgb train-lstm kb sample dashboard api docker-build docker-up docker-down test pipeline clean

help:
	@echo "FraudSentinel Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install Python dependencies"
	@echo ""
	@echo "Offline pipeline (run in this order):"
	@echo "  make download        Download IEEE-CIS from Kaggle"
	@echo "  make features        Run Spark feature engineering"
	@echo "  make train-xgb       Train XGBoost (with MLflow)"
	@echo "  make train-lstm      Train LSTM autoencoder (with MLflow)"
	@echo "  make kb              Build ChromaDB knowledge base"
	@echo "  make sample          Build the 10k demo sample"
	@echo "  make pipeline        Run all of the above in order"
	@echo ""
	@echo "Run:"
	@echo "  make dashboard       Run the Streamlit SOC dashboard"
	@echo "  make api             Run the FastAPI scoring service"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker images"
	@echo "  make docker-up       Run dashboard + API via docker compose"
	@echo "  make docker-down     Stop docker compose"
	@echo ""
	@echo "Test:"
	@echo "  make test            Run smoke tests"
	@echo ""
	@echo "Clean:"
	@echo "  make clean           Remove caches and build artifacts"

install:
	pip install -r requirements.txt

download:
	python -m src.data_ingestion.download_ieee_cis

features:
	python -m src.spark_pipeline.build_features

train-xgb:
	python -m src.ml_models.train_xgboost

train-lstm:
	python -m src.dl_models.train_lstm_ae

kb:
	python -m src.agentic.build_knowledge_base

sample:
	python -m src.data_ingestion.sample_for_demo

pipeline: download features train-xgb train-lstm kb sample
	@echo "✅ Full pipeline complete. Run 'make dashboard' to launch the demo."

dashboard:
	streamlit run src/dashboard/app.py

api:
	uvicorn src.api.main:app --reload --port 8000

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

test:
	pytest tests/ -v

mlflow:
	mlflow ui --port 5000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/
