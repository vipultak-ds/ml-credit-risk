import json
import yaml
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

# ---------------- LOAD PIPELINE CONFIG ----------------
with open("pipeline_config.yml", "r") as f:
    pipeline_cfg = yaml.safe_load(f)

EVAL_TABLE = pipeline_cfg["tables"]["evaluation_log"]
TRACKED_METRICS = pipeline_cfg["metrics"]["classification"]["tracked_metrics"]

# Optional: duplicate handling config
DUPLICATE_CFG = pipeline_cfg.get("tables", {}).get("duplicate_handling", {})
DUPLICATE_ENABLED = DUPLICATE_CFG.get("enabled", True)

print(f"✅ Evaluation Table: {EVAL_TABLE}")
print(f"✅ Tracked Metrics: {TRACKED_METRICS}")
print(f"✅ Duplicate Handling Enabled: {DUPLICATE_ENABLED}")


# ---------------- CREATE TABLE IF NOT EXISTS ----------------
def create_eval_table_if_not_exists():
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {EVAL_TABLE} (
        model_name STRING,
        model_type STRING,
        run_id STRING,
        experiment_name STRING,
        created_timestamp TIMESTAMP,
        hyperparameters STRING,
        metrics STRING
    )
    USING DELTA
    """)
    print(f"✅ Evaluation table ready: {EVAL_TABLE}")


# ---------------- DUPLICATE CHECK ----------------
def is_duplicate(model_type: str, experiment_name: str, hyper_json: str) -> bool:
    """
    Duplicate means:
    same model_type + experiment_name + hyperparameters already exists
    """
    if not DUPLICATE_ENABLED:
        return False

    try:
        df = spark.read.table(EVAL_TABLE).filter(
            (col("model_type") == model_type) &
            (col("experiment_name") == experiment_name) &
            (col("hyperparameters") == hyper_json)
        )
        return df.limit(1).count() > 0
    except Exception as e:
        print(f"⚠️ Duplicate check skipped (table read error): {e}")
        return False


# ---------------- LOG ONE RUN TO DELTA ----------------
def log_run_to_table(
    model_name: str,
    model_type: str,
    run_id: str,
    experiment_name: str,
    hyperparams: dict,
    metrics: dict
):
    """
    Stores:
    - hyperparameters as JSON string
    - metrics as JSON string (only tracked metrics)
    - avoids duplicates
    """

    # keep only tracked metrics
    filtered_metrics = {k: metrics.get(k, None) for k in TRACKED_METRICS}

    # JSON normalize (stable ordering)
    hyper_json = json.dumps(hyperparams, sort_keys=True)
    metrics_json = json.dumps(filtered_metrics, sort_keys=True)

    # check duplicates
    if is_duplicate(model_type, experiment_name, hyper_json):
        print(f"⚠️ Duplicate row detected. Skipping insert for run_id={run_id}")
        return

    row = [{
        "model_name": model_name,
        "model_type": model_type,
        "run_id": run_id,
        "experiment_name": experiment_name,
        "created_timestamp": datetime.utcnow(),
        "hyperparameters": hyper_json,
        "metrics": metrics_json
    }]

    df = spark.createDataFrame(row)
    df.write.format("delta").mode("append").saveAsTable(EVAL_TABLE)

    print(f"✅ Logged evaluation row for run_id={run_id}")
