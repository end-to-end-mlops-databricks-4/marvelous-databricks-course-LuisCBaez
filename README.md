# MLOps with Databricks (Walmart Forecasting)

## Business Problem
Walmart needs reliable **monthly demand forecasts** to improve **inventory planning, supply chain optimization, and category management**.  

This project focuses on forecasting **monthly sales at the department level (`dept_id`) per U.S. state** using the **M5 dataset**.  

- **Granularity**: `(state_id, dept_id)`  
- **Forecast horizon**: 12–24 months ahead  
- **Target variable**: `sales` (sum of item sales in each department per month per state)  

By working at the **department level**, I both balance granularity (less noisy than item-level, more actionable than category-level) **and reduce the total number of time series to forecast**, making the project more tractable for experimentation and MLOps implementation.

---

## Week 1: Dataset Preparation

### Raw data
The dataset is derived from the **M5 competition data**:
- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

### Preprocessing  
I will **only use sales history**. No external regressors (events, SNAP, prices) are included to keep the project simple and focused.  
That’s why I aggregated everything to **monthly frequency** and at the **department level (`dept_id`)**.  
This project is primarily to learn how to implement **MLOps principles in Databricks**.

### Final dataset
| unique_id   | state_id | dept_id      | date       | sales |
|-------------|----------|--------------|------------|-------|
| CA_HOBBIES_1 | CA       | HOBBIES_1    | 2011-01-01 | 1523  |
| TX_FOODS_2   | TX       | FOODS_2      | 2011-01-01 | 2847  |
| WI_HOUSEHOLD_1   | WI       | HOUSEHOLD_1  | 2011-01-01 | 1194  |

👉 **Target**: `Sales`

### Output: Saved to Unity Catalog as Delta tables:
- `mlops_dev.lcbaezlo.m5_prepared`
- `mlops_dev.lcbaezlo.m5_train`
- `mlops_dev.lcbaezlo.m5_test`

### Flow:
1. Load raw CSVs from UC Volume  
2. Build monthly series (`make_monthly_forecast_df`)  
3. Split holdout (last 4 months = test)  
4. Write Delta tables (prepared/train/test)

### Week 1 Deliverable Checklist
- [x] Raw data uploaded to UC Volume
- [x] Config in `project_config.yaml`
- [x] Processing module (`config.py`, `data_processor.py`)
- [x] Script to run pipeline (`01_process_data.py`)
- [x] Delta tables created in Unity Catalog
- [x] Verified in Databricks UI

---

## Project Structure

```bash
src/
  timeseries_mlops/
    config.py          # paths, schema, catalog
    data_processor.py  # load, preprocess, split, write Delta
scripts/
  01_process_data.py   # Week 1 deliverable


```


---

## Modeling

### Models
I restrict the project to **classical statistical forecasting methods**:

- **Naïve**  (benchmark)  
- **Seasonal Naïve** (benchmark)  
- **ARIMA**  
- **Exponential Smoothing (ETS)**  
- **Theta method**  
- **Prophet** 

Modules: `statforecast`, `prophet`, `mlflow`


### Evaluation
- **Holdout**: last 4–8 months of data.  
- **Metrics**: MAE, MAPE, sMAPE, MASE.  
- **Diagnostics**: ACF, PACF, Ljung–Box, QQ plot, residual histograms  

---

## Environment Setup

```bash
# clone repo & enter
git clone <repo-url>
cd marvelous-databricks-course-LuisCBaez

# create env
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# install package
pip install -e .

# authenticate Databricks
databricks auth login --host <url> --profile DEFAULT --configure-cluster
```

---

## How to Run

```bash
python scripts/01_process_data.py
```
Expected Output:
```bash
✅ Connected to Databricks cluster
✅ Wrote Delta tables:
  mlops_dev.lcbaezlo.m5_prepared (rows=1365)
  mlops_dev.lcbaezlo.m5_train    (rows=1281)
  mlops_dev.lcbaezlo.m5_test     (rows=84)
```


---


## 7. Notes

This repository is part of the MLOps with Databricks course.
- Week 1 deliverable = preprocessing pipeline → Delta tables.
- Later weeks → model training, MLflow tracking, feature store, deployment.

### Set up your environment
In this course, we use Databricks 16.04 LTS runtime, which uses Python 3.11.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

Install task: https://taskfile.dev/installation/

Update .env file with the following:
```
GIT_TOKEN=<your github PAT>
```

To create a new environment and create a lockfile, run:
```
task sync-dev
source .venv/bin/activate
```

Or, alternatively:
```
export GIT_TOKEN=<your github PAT>
uv venv -p 3.11 .venv
source .venv/bin/activate
uv sync --extra dev
```



