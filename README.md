# Credit Risk Scoring Pipeline

End-to-end credit risk prediction system — from raw Lending Club data to a deployed FastAPI scoring API on Azure Cloud.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-EC4E20?style=for-the-badge&logo=xgboost&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Azure-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-Cloud-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/Git-VCS-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.1.4-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-41_Tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

![Tests](https://github.com/gitadi2/credit-risk-final/actions/workflows/ci.yml/badge.svg)

---

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Raw Data    │───▶│ Data Cleaning│───▶│   Feature     │───▶│   Model      │
│ Lending Club │    │ Dedup, FICO, │    │ Engineering   │    │  Training    │
│  10K loans   │    │  Outliers    │    │ 14 features   │    │ 3 models     │
└─────────────┘    └──────────────┘    └───────────────┘    └──────┬───────┘
                                                                   │
                   ┌──────────────┐    ┌───────────────┐           │
                   │   FastAPI    │◀───│    Policy     │◀──────────┘
                   │ Scoring API  │    │  Simulator    │
                   │  /predict    │    │ Binary Search │
                   └──────┬───────┘    └───────────────┘
                          │
              ┌───────────┴────────────┐
              │   Azure Cloud Deploy   │
              │ PostgreSQL + Blob Store│
              └────────────────────────┘
```

---

## Model Performance

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| **Logistic Regression** | 0.663 | 0.403 | **0.704** ✓ Best |
| Random Forest | 0.800 | 0.236 | 0.698 |
| XGBoost | 0.758 | 0.288 | 0.651 |

Best model selected by AUC-ROC. SMOTE applied for class imbalance. SHAP used for feature explainability.

<p align="center">
  <img src="reports/figures/roc_comparison.png" width="45%" alt="ROC Curves"/>
  <img src="reports/figures/shap_summary.png" width="45%" alt="SHAP Feature Importance"/>
</p>

---

## Policy Optimization

Binary search finds the optimal approval threshold balancing acceptance rate vs. default rate:

| Metric | Value |
|--------|-------|
| Optimal Threshold | 0.4453 |
| Acceptance Rate | 51.45% |
| Default Rate | 9.91% |
| Binary Search | 1.17 ms |
| Linear Search | 2.40 ms |
| **Speedup** | **2.1x faster** |

<p align="center">
  <img src="reports/figures/policy_tradeoff.png" width="60%" alt="Policy Tradeoff"/>
</p>

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML** | scikit-learn, XGBoost, SHAP, SMOTE (imbalanced-learn) |
| **DSA** | LRU Cache, Binary Search, Min-Heap, Sliding Window, Prefix Sums, Hash Map |
| **API** | FastAPI, Uvicorn, Pydantic validation |
| **Database** | PostgreSQL on Azure, SQLAlchemy ORM |
| **Cloud** | Azure Blob Storage, Azure PostgreSQL, Azure Identity |
| **Data** | Pandas, NumPy, Statsmodels |
| **DevOps** | Docker, GitHub Actions CI/CD, Git |
| **Testing** | pytest (41 test cases) |
| **Visualization** | Matplotlib, Seaborn, Power BI, Chart.js |

---

## Project Structure

```
credit-risk-final/
├── run_pipeline.py              # One-click: clean → engineer → train → simulate → test
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci.yml    # CI: lint + test on every push
│
├── src/
│   ├── data_cleaning.py         # Dedup, FICO calc, outlier capping, validation
│   ├── feature_engineering.py   # 14 engineered features + one-hot encoding
│   ├── model_training.py        # LogReg, RF, XGBoost + SHAP explainability
│   ├── policy_simulator.py      # Binary search threshold optimization
│   ├── api/
│   │   ├── scoring_api.py       # FastAPI endpoints: /predict, /batch, /policy
│   │   └── schemas.py           # Pydantic request/response models
│   └── utils/
│       ├── algorithms.py        # Binary search, sliding window, benchmarks
│       └── data_structures.py   # LRU Cache, SortedRiskArray, RiskBucketMap
│
├── sql/
│   ├── 01_create_schema.sql     # PostgreSQL schema
│   ├── 02_create_tables.sql     # Table definitions
│   ├── 03_etl_pipeline.sql      # SQL-based ETL
│   └── 04_feature_engineering.sql
│
├── tests/
│   ├── test_algorithms.py       # Binary search, sliding window tests
│   ├── test_api.py              # API schema validation tests
│   └── test_data_structures.py  # LRU cache, sorted array, bucket map tests
│
├── data/
│   ├── raw/lending_club_sample.csv
│   └── processed/               # Generated CSVs
│
├── models/                      # Trained .pkl + metrics JSON
├── reports/figures/             # ROC curves, SHAP plots, policy charts
├── dashboards/                  # Interactive HTML dashboard
├── screenshots/                 # Power BI + Swagger UI + Azure Cloud Deployment screenshots
└── benchmarks/                  # Performance benchmarking suite
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/gitadi2/credit-risk-final.git
cd credit-risk-final
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp config/.env.example .env
# Edit .env with your database credentials
```

### 3. Run Full Pipeline

```bash
python run_pipeline.py
```

This runs all 5 stages:
1. **Generate** sample data (10K loans)
2. **Clean** — dedup, FICO scores, outlier capping
3. **Engineer** — 14 features + encoding (25 final features)
4. **Train** — 3 models, SHAP analysis, best model selection
5. **Simulate** — Policy optimization + benchmarks

### 4. Launch Scoring API

```bash
uvicorn src.api.scoring_api:app --port 8000
```

Open Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model status, cache stats, version |
| `/predict` | POST | Score a single loan application |
| `/predict/batch` | POST | Score up to 1,000 applications |
| `/policy/simulate` | POST | Simulate approval policy |

**Example Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000,
    "annual_inc": 65000,
    "dti": 18.5,
    "int_rate": 0.12,
    "installment": 450,
    "grade": "B",
    "fico_score": 710
  }'
```

**Example Response:**

```json
{
  "applicant_id": "a1b2c3d4",
  "default_probability": 0.15,
  "risk_segment": "Low Risk",
  "recommendation": "APPROVE",
  "confidence": 0.70,
  "top_risk_factors": ["moderate_dti"],
  "cached": false
}
```

---

## Algorithms & Data Structures

| Component | Complexity | Purpose |
|-----------|-----------|---------|
| **Binary Search Threshold** | O(log(1/ε) × log n) | Optimal approval cutoff |
| **LRU Cache** | O(1) get/put | Cache repeated applicant scores |
| **SortedRiskArray** | O(log n) query | Fast percentile & threshold lookups |
| **RiskBucketMap** | O(1) lookup | Default rate by risk segment |
| **Sliding Window** | O(n) single pass | Rolling default rate monitoring |
| **Prefix Sums** | O(1) range query | Count defaults in score ranges |

---

## Screenshots

<p align="center">
  <img src="screenshots/swagger_api.jpg" width="80%" alt="Swagger API"/>
</p>

<p align="center">
  <img src="screenshots/executive_overview_powerbi.jpg" width="45%" alt="Executive Overview"/>
  <img src="screenshots/model_performance_powerbi.jpg" width="45%" alt="Model Performance"/>
</p>

---

## Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

---

## Author

ADITYA SATAPATHY
[https://www.linkedin.com/in/adisatapathy]
