# BankPredict: End-to-End Predictive Analytics and MLOps Platform

An end-to-end machine learning platform for predicting bank marketing campaign subscriptions, built using Python, PostgreSQL, XGBoost, MLflow, FastAPI, Docker and AWS.

The project demonstrates the development of a complete predictive analytics workflow, from raw data ingestion and SQL-based analysis to machine learning, experiment tracking, API development, containerisation and cloud deployment.

## 1. Project Overview

BankPredict predicts whether a customer is likely to subscribe to a bank term deposit following a marketing campaign.

The project uses the UCI Bank Marketing dataset and implements a binary classification pipeline to identify patterns associated with customer subscription behaviour.

Beyond model development, the project demonstrates how machine learning solutions can be developed, packaged, deployed and made accessible through production-oriented services.

### Key Objectives

- Develop a structured data ingestion pipeline using PostgreSQL.
- Perform exploratory data analysis and data quality checks.
- Implement data cleaning and feature engineering using Python and Pandas.
- Demonstrate distributed data processing concepts using PySpark.
- Develop and compare Logistic Regression and XGBoost models.
- Track experiments, parameters and evaluation metrics using MLflow.
- Build a RESTful prediction service using FastAPI.
- Containerise the application using Docker.
- Deploy the containerised prediction service using AWS.

## 2. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming | Python | Data processing and machine learning |
| Database | PostgreSQL | Structured data storage and SQL analysis |
| Data Processing | Pandas | Data cleaning and transformation |
| Distributed Processing | PySpark | Distributed data processing demonstration |
| Machine Learning | Scikit-learn | Preprocessing and baseline modelling |
| Gradient Boosting | XGBoost | Predictive modelling |
| Experiment Tracking | MLflow | Experiment and model performance tracking |
| API Development | FastAPI | Model inference and REST API |
| Testing | Pytest | Automated testing |
| Containerisation | Docker | Application packaging |
| Version Control | Git and GitHub | Source code management |
| Cloud Deployment | AWS ECR and ECS | Container storage and deployment |
| CI | GitHub Actions | Automated testing |

## 3. System Architecture

The overall architecture follows this workflow:

```text
                 UCI BANK MARKETING DATASET
                            |
                            v
                       PostgreSQL
                            |
                            v
                 SQL Analysis and Extraction
                            |
                            v
                    Python / Pandas
                            |
                            v
                     Data Cleaning
                            |
                            v
                    Feature Engineering
                            |
                            v
                  Machine Learning Pipeline
                            |
                 +----------+----------+
                 |                     |
                 v                     v
           Logistic Regression      XGBoost
                 |                     |
                 +----------+----------+
                            |
                            v
                       MLflow
                 Experiment Tracking
                            |
                            v
                    Selected Model
                            |
                            v
                         FastAPI
                            |
                            v
                         Docker
                            |
                            v
                      AWS ECR / ECS
                            |
                            v
                  Cloud Prediction API
```

PySpark is included as an optional data processing implementation.

## 4. Dataset

**Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

The project uses the `bank-full.csv` dataset.

Dataset characteristics:

- Total observations: 45,211.
- Classification type: Binary classification.
- Target variable: `y`.
- Positive class: `yes`.
- Negative class: `no`.

The target is transformed into:

- `1`: Customer subscribed.
- `0`: Customer did not subscribe.

The `duration` variable is excluded from the model features because call duration is only known after the marketing call has ended and would introduce information leakage for a pre-call prediction task.

## 5. Project Structure

```text
bankpredict-portfolio/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── artifacts/
│   ├── model.joblib
│   └── model_metadata.json
│
├── data/
│   ├── raw/
│   │   └── bank-full.csv
│   └── processed/
│       └── bank_clean.csv
│
├── examples/
│   └── predict.json
│
├── reports/
│   ├── metrics.json
│   ├── test_predictions.csv
│   ├── test_classification_report.txt
│   ├── test_precision_recall.png
│   └── test_confusion_matrix.png
│
├── sql/
│   └── analysis.sql
│
├── src/
│   └── bankpredict/
│       ├── __init__.py
│       ├── config.py
│       ├── schema.py
│       ├── download.py
│       ├── ingest.py
│       ├── prepare.py
│       ├── features.py
│       ├── model.py
│       ├── train.py
│       └── api.py
│
├── tests/
│   ├── test_features.py
│   └── test_api.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements-core.txt
├── requirements-api.txt
├── requirements-spark.txt
└── README.md
```

Generated datasets, trained models and experiment databases are excluded from version control where appropriate.

## 6. Installation and Setup

### Prerequisites

Install the following:

- Python 3.11 or later.
- Docker Desktop.
- Git.
- PostgreSQL client tools (optional).
- AWS CLI (for cloud deployment).

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/bankpredict-portfolio.git

cd bankpredict-portfolio
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

### Step 3: Install dependencies

```powershell
pip install -r requirements-core.txt -r requirements-api.txt
```

For the optional PySpark implementation:

```powershell
pip install -r requirements-spark.txt
```

### Step 4: Configure environment variables

Create your local environment configuration:

```powershell
Copy-Item .env.example .env
```

Configure the database credentials in `.env`:

```dotenv
DB_USER=bankuser
DB_PASSWORD=your_local_password
DB_NAME=bankpredict
DB_HOST=localhost
DB_PORT=5433
```

Never commit `.env` or expose database credentials in source code.

### Step 5: Configure Python imports

From the project root, run:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

This allows Python to locate the `bankpredict` package.

## 7. Running the Pipeline

Execute the following stages in order.

### Stage 1: Start PostgreSQL

```powershell
docker compose up -d db
```

Check the database container:

```powershell
docker compose ps
```

### Stage 2: Download the dataset

```powershell
python -m bankpredict.download
```

The original dataset is downloaded and saved in:

```text
data/raw/bank-full.csv
```

### Stage 3: Ingest data into PostgreSQL

```powershell
python -m bankpredict.ingest
```

This imports the dataset into the `bank_marketing` PostgreSQL table.

To reload the table:

```powershell
python -m bankpredict.ingest --replace
```

### Stage 4: Perform SQL analysis

Execute the SQL analysis script:

```powershell
Get-Content sql\analysis.sql | docker compose exec -T db psql -U bankuser -d bankpredict
```

This stage demonstrates SQL-based data exploration and analysis.

### Stage 5: Prepare the dataset

Using Pandas:

```powershell
python -m bankpredict.prepare --engine pandas
```

Using PySpark:

```powershell
python -m bankpredict.prepare --engine spark
```

The processed dataset is saved in:

```text
data/processed/bank_clean.csv
```

### Stage 6: Train machine learning models

```powershell
python -m bankpredict.train
```

The training pipeline:

1. Partitions the data into training, validation and test sets.
2. Performs feature engineering and preprocessing.
3. Trains Logistic Regression.
4. Trains XGBoost.
5. Compares models using validation average precision.
6. Selects the best-performing model.
7. Determines a classification threshold using validation data.
8. Evaluates the selected model on the held-out test partition.
9. Saves the model, metadata and evaluation reports.
10. Logs experiments and metrics using MLflow.

## 8. Experiment Tracking with MLflow

Start the MLflow tracking server:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open the MLflow dashboard:

http://127.0.0.1:5000

The dashboard allows inspection of:

- Experiment runs.
- Model parameters.
- Validation metrics.
- Test metrics.
- Model artefacts.
- Experiment comparisons.

## 9. Model Evaluation

The models are evaluated using multiple classification metrics.

| Metric | Purpose |
|---|---|
| ROC-AUC | Measures ranking discrimination |
| Average Precision | Evaluates precision and recall across thresholds |
| Precision | Measures the reliability of positive predictions |
| Recall | Measures the proportion of positive cases identified |
| F1-score | Balances precision and recall |
| Brier Score | Evaluates probability prediction quality |
| Top-decile lift | Evaluates targeting performance within the highest-ranked 10% |

Evaluation outputs are saved in the `reports/` directory.

No fixed performance result is assumed; actual results are generated by running the training pipeline.

## 10. FastAPI Prediction Service

Start the API:

```powershell
uvicorn bankpredict.api:app --host 127.0.0.1 --port 8000 --reload
```

API documentation:

http://127.0.0.1:8000/docs

Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service information |
| `/health` | GET | Service and model health |
| `/model-info` | GET | Model metadata |
| `/predict` | POST | Customer subscription prediction |

### Example prediction request

The example input is available in:

```text
examples/predict.json
```

Test the API using PowerShell:

```powershell
$body = Get-Content examples\predict.json -Raw

Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

The API returns:

- Predicted subscription probability.
- Predicted subscription class.
- Decision threshold.
- Contact recommendation.
- Model name.

## 11. Automated Testing

Run the test suite:

```powershell
python -m pytest -q
```

The tests cover:

- Feature engineering.
- Model pipeline fitting and prediction.
- API request validation.
- Prediction endpoint functionality.
- API health checks.

GitHub Actions is used to automate testing when code is pushed or a pull request is created.

## 12. Docker Deployment

Build the Docker image:

```powershell
docker build -t bankpredict-api:1.0 .
```

Run the container:

```powershell
docker run --rm -p 127.0.0.1:8000:8000 bankpredict-api:1.0
```

Access the API:

http://127.0.0.1:8000/docs

The container packages the trained model and its inference dependencies, allowing the prediction service to run independently of the local development environment.

## 13. AWS Cloud Deployment

The cloud deployment architecture uses:

- Amazon Elastic Container Registry (ECR) for storing Docker images.
- Amazon Elastic Container Service (ECS) Express Mode for running the application.
- HTTPS endpoints for accessing the deployed prediction service.

The deployment workflow is:

1. Build the Docker image.
2. Authenticate with Amazon ECR.
3. Push the image to the ECR repository.
4. Create an ECS Express Mode service.
5. Configure port 8000 and the `/health` endpoint.
6. Deploy the container.
7. Verify the cloud-hosted prediction API.

AWS deployment documentation:

- [Amazon ECR](https://docs.aws.amazon.com/ecr/)
- [Amazon ECS](https://docs.aws.amazon.com/ecs/)
- [ECS Express Mode deployment](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-first-run.html)

Cloud deployment may incur charges. Review AWS pricing and remove unused resources.

## 14. Engineering Considerations and Limitations

This project is a portfolio and learning implementation of an end-to-end predictive analytics system.

Important considerations for further development include:

- **Data leakage:** Features must only contain information available at the intended prediction time.
- **Data partitioning:** The current ordered 60:20:20 split is a demonstration, not proof of future-period or customer-independent generalisation. Reliable campaign dates and customer identifiers are needed for stronger evaluation.
- **Model explainability:** SHAP analysis can be added to understand feature contributions.
- **Model monitoring:** Prediction distributions, data drift, latency and service errors should be monitored.
- **Model governance:** Fairness, calibration, reproducibility and model limitations require further investigation.
- **Business validation:** Predicted subscription probabilities do not establish causal campaign uplift. Controlled experiments are needed to measure the impact of targeting decisions.
- **Scalability:** The current local PySpark implementation demonstrates Spark transformations but is not yet a distributed, large-scale ingestion architecture.

## 15. Learning Resources

| Technology | Documentation |
|---|---|
| Python | https://docs.python.org/3/tutorial/ |
| PostgreSQL | https://www.postgresql.org/docs/current/tutorial.html |
| Pandas | https://pandas.pydata.org/docs/ |
| PySpark | https://spark.apache.org/docs/latest/api/python/ |
| Scikit-learn | https://scikit-learn.org/stable/ |
| XGBoost | https://xgboost.readthedocs.io/en/stable/ |
| MLflow | https://mlflow.org/docs/latest/ |
| FastAPI | https://fastapi.tiangolo.com/tutorial/ |
| Docker | https://docs.docker.com/get-started/ |
| AWS | https://docs.aws.amazon.com/ |

## 16. Project Status

The project is being developed incrementally, covering data engineering, machine learning, model evaluation, API development, containerisation and cloud deployment.

The objective is to demonstrate practical capabilities across the end-to-end machine learning lifecycle, with an emphasis on reproducibility, maintainability and deployment.

**Project:** BankPredict – End-to-End Predictive Analytics and MLOps Platform

**Primary language:** Python

**Architecture:** PostgreSQL → Python/PySpark → XGBoost → MLflow → FastAPI → Docker → AWS