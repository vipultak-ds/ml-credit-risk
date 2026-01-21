# 🎯 MODEL REGISTRATION SCRIPT - MULTI MODEL (DYNAMIC + FIXED)

import mlflow
from mlflow.tracking import MlflowClient
import sys
import yaml
import os
import json
from typing import Dict, Optional, List
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType, BooleanType
)

print("=" * 80)
print("🎯 MODEL REGISTRATION SYSTEM - MULTI MODEL + DYNAMIC")
print("=" * 80)

# ---------------------- LOAD CONFIG FILES ----------------------
try:
    with open("pipeline_config.yml", "r") as f:
        pipeline_cfg = yaml.safe_load(f)

    with open("experiments_config.yml", "r") as f:
        experiments_cfg = yaml.safe_load(f)

    print("✅ Configuration files loaded\n")
except Exception as e:
    print(f"❌ Failed to load config: {e}")
    sys.exit(1)

# ---------------------- OPTIONAL WIDGETS ----------------------
# If running in Databricks job/notebook, this allows selecting models
try:
    dbutils.widgets.text("MODELS_TO_REGISTER", "all", "Models to Register (all or comma-separated)")
except:
    pass


# ---------------------- INIT SPARK + MLFLOW ----------------------
spark = SparkSession.builder.appName("ModelRegistrationMultiModel").getOrCreate()
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# ---------------------- GLOBAL CONFIG ----------------------
BASE_EXPERIMENT_NAME = pipeline_cfg["experiment"]["name"]
ARTIFACT_PATH = pipeline_cfg["experiment"]["artifact_path"]

PRIMARY_METRIC = pipeline_cfg["metrics"]["classification"]["primary_metric"]
DIRECTION = pipeline_cfg["metrics"]["classification"]["direction"]

DUP_CFG = pipeline_cfg["registry"]["duplicate_detection"]
DUPLICATE_CHECK_ENABLED = DUP_CFG.get("enabled", True)
TOLERANCE = DUP_CFG.get("tolerance", 0.001)
METRICS_TO_COMPARE = DUP_CFG.get("metrics_to_compare", [])

MODEL_NAMING_FMT = pipeline_cfg["models"]["naming"]["format"]
UC_CATALOG = pipeline_cfg["models"]["catalog"]
UC_SCHEMA = pipeline_cfg["models"]["schema"]
BASE_NAME = pipeline_cfg["models"]["base_name"]

# IMPORTANT: Your pipeline_config.yml currently DOES NOT have registration_log
# so we provide a safe default table if missing
REGISTRATION_LOG_TABLE = pipeline_cfg.get("tables", {}).get(
    "registration_log",
    f"{UC_CATALOG}.{UC_SCHEMA}.model_registration_log"
)

print(f"✅ Primary Metric: {PRIMARY_METRIC} ({DIRECTION})")
print(f"✅ Artifact Path: {ARTIFACT_PATH}")
print(f"✅ Duplicate Detection: {DUPLICATE_CHECK_ENABLED} (tolerance={TOLERANCE})")
print(f"✅ Registration Log Table: {REGISTRATION_LOG_TABLE}\n")


# ---------------------- MODEL SHORT NAME (DYNAMIC) ----------------------
def get_model_short_name(model_type: str) -> str:
    """
    Makes short name like:
      random_forest -> RF
      logistic_regression -> LR
      xgboost -> XGB
    """
    words = model_type.split("_")
    return "".join([w[0].upper() for w in words if w])


def get_experiment_name_for_model(model_type: str) -> str:
    """
    Training script uses:
      /Shared/CreditRisk_ML_Experiments_RF
    so we follow the same.
    """
    return f"{BASE_EXPERIMENT_NAME}_{get_model_short_name(model_type)}"


def get_uc_model_name(model_type: str) -> str:
    """
    Uses naming format from pipeline_config.yml
    """
    return MODEL_NAMING_FMT.format(
        catalog=UC_CATALOG,
        schema=UC_SCHEMA,
        base_name=BASE_NAME,
        model_type=model_type
    )


# ---------------------- MODELS TO REGISTER (DYNAMIC) ----------------------
def get_models_to_register() -> List[str]:
    """
    Priority:
    1) Databricks widget MODELS_TO_REGISTER
    2) ENV MODELS_TO_REGISTER
    3) Default: all models from experiments_config.yml
    """
    available_models = list(experiments_cfg.get("models", {}).keys())

    if not available_models:
        raise ValueError("❌ No models defined in experiments_config.yml")

    value = None
    try:
        value = dbutils.widgets.get("MODELS_TO_REGISTER")
        print(f"📌 MODELS_TO_REGISTER from Widget: '{value}'")
    except:
        value = os.getenv("MODELS_TO_REGISTER", "all")
        print(f"📌 MODELS_TO_REGISTER from ENV: '{value}'")

    value = (value or "").strip()

    if value.lower() == "all" or value == "":
        return available_models

    models = [m.strip() for m in value.split(",") if m.strip()]
    invalid = [m for m in models if m not in available_models]

    if invalid:
        raise ValueError(f"❌ Invalid models: {invalid} | Available: {available_models}")

    return models


# ---------------------- REGISTRATION LOG TABLE SCHEMA ----------------------
def get_table_schema():
    return StructType([
        StructField("timestamp", TimestampType(), True),
        StructField("run_id", StringType(), True),
        StructField("run_name", StringType(), True),
        StructField("model_type", StringType(), True),
        StructField("model_name", StringType(), True),
        StructField("experiment_name", StringType(), True),
        StructField("primary_metric", StringType(), True),
        StructField("primary_metric_value", DoubleType(), True),
        StructField("metrics_json", StringType(), True),
        StructField("params_json", StringType(), True),
        StructField("registered", BooleanType(), True),
        StructField("registered_version", StringType(), True),
        StructField("reason", StringType(), True)
    ])


def ensure_table_exists(table_name: str):
    try:
        spark.sql(f"DESCRIBE TABLE {table_name}")
        print(f"   ✅ Table exists: {table_name}")
    except:
        print(f"   🆕 Creating Delta table: {table_name}")
        empty_df = spark.createDataFrame([], schema=get_table_schema())
        empty_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
        print(f"   ✅ Table created: {table_name}")


def is_already_logged(run_id: str) -> bool:
    try:
        df = spark.sql(f"""
            SELECT run_id
            FROM {REGISTRATION_LOG_TABLE}
            WHERE run_id = '{run_id}'
            LIMIT 1
        """)
        return df.count() > 0
    except:
        return False


# ---------------------- FETCH RUNS FROM MLFLOW EXPERIMENT ----------------------
def get_runs_for_model(model_type: str) -> List[Dict]:
    exp_name = get_experiment_name_for_model(model_type)

    print(f"   🔬 Searching Experiment: {exp_name}")

    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        print(f"   ⚠️ Experiment not found: {exp_name}")
        return []

    runs = client.search_runs(
        [exp.experiment_id],
        order_by=[f"metrics.{PRIMARY_METRIC} DESC"],
        max_results=500
    )

    # filter only runs of this model_type
    filtered = [r for r in runs if model_type in (r.info.run_name or "")]

    results = []
    for r in filtered:
        results.append({
            "run_id": r.info.run_id,
            "run_name": r.info.run_name or "unnamed_run",
            "params": r.data.params,
            "metrics": r.data.metrics,
            "primary_metric": r.data.metrics.get(PRIMARY_METRIC),
            "model_uri": f"runs:/{r.info.run_id}/{ARTIFACT_PATH}",
            "experiment_name": exp_name
        })

    print(f"   ✅ Found {len(results)} runs for {model_type}")
    return results


# ---------------------- DUPLICATE CHECK IN REGISTRY ----------------------
def is_duplicate_model(new_model: Dict, model_name: str) -> bool:
    """
    Duplicate means:
    - all METRICS_TO_COMPARE are within tolerance
    - AND params are exactly same
    """
    if not DUPLICATE_CHECK_ENABLED:
        return False

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return False
    except Exception as e:
        print(f"      ⚠️ Could not fetch model versions: {e}")
        return False

    for v in versions:
        try:
            existing_run = client.get_run(v.run_id)

            # compare metrics
            all_metrics_match = True
            for metric_name in METRICS_TO_COMPARE:
                old_val = existing_run.data.metrics.get(metric_name, 0)
                new_val = new_model["metrics"].get(metric_name, 0)

                if abs(old_val - new_val) > TOLERANCE:
                    all_metrics_match = False
                    break

            # compare params
            params_match = (existing_run.data.params == new_model["params"])

            if all_metrics_match and params_match:
                return True

        except:
            continue

    return False


# ---------------------- REGISTER MODEL ----------------------
def register_model(new_model: Dict, model_name: str) -> Optional[str]:
    try:
        reg = mlflow.register_model(new_model["model_uri"], model_name)
        return str(reg.version)
    except Exception as e:
        print(f"      ❌ Registration failed: {e}")
        return None


def log_decision(model: Dict, model_type: str, model_name: str, registered: bool, version: Optional[str], reason: str):
    record = {
        "timestamp": datetime.now(),
        "run_id": model["run_id"],
        "run_name": model["run_name"],
        "model_type": model_type,
        "model_name": model_name,
        "experiment_name": model["experiment_name"],
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_value": float(model["primary_metric"]) if model["primary_metric"] else 0.0,
        "metrics_json": json.dumps({k: model["metrics"].get(k) for k in METRICS_TO_COMPARE}, sort_keys=True),
        "params_json": json.dumps(model["params"], sort_keys=True),
        "registered": registered,
        "registered_version": version if version else "N/A",
        "reason": reason
    }

    spark_df = spark.createDataFrame([record], schema=get_table_schema())

    spark_df.write.format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .saveAsTable(REGISTRATION_LOG_TABLE)


# ---------------------- PROCESS ONE MODEL TYPE ----------------------
def process_model_type(model_type: str):
    print(f"\n{'='*80}")
    print(f"🚀 PROCESSING MODEL TYPE: {model_type.upper()}")
    print(f"{'='*80}")

    model_name = get_uc_model_name(model_type)
    print(f"📦 UC Model Name: {model_name}")

    runs = get_runs_for_model(model_type)
    if not runs:
        print(f"   ⚠️ No runs found for {model_type}")
        return {"registered": 0, "skipped": 0, "total": 0}

    registered_count = 0
    skipped_count = 0

    for idx, model in enumerate(runs, start=1):
        print(f"\n   [{idx}/{len(runs)}] Run: {model['run_name']} | run_id={model['run_id']}")

        # skip if already logged
        if is_already_logged(model["run_id"]):
            print("      ⏭️ Already logged → skip")
            skipped_count += 1
            continue

        # duplicate check in registry
        if is_duplicate_model(model, model_name):
            print("      ⚠️ Duplicate detected in Registry → skip")
            log_decision(model, model_type, model_name, False, None, "Duplicate metrics+params → skipped")
            skipped_count += 1
            continue

        # register
        version = register_model(model, model_name)
        if version:
            print(f"      ✅ Registered: {model_name} v{version}")
            log_decision(model, model_type, model_name, True, version, "Registered successfully")
            registered_count += 1
        else:
            log_decision(model, model_type, model_name, False, None, "Registration failed")
            skipped_count += 1

    return {"registered": registered_count, "skipped": skipped_count, "total": len(runs)}


# ---------------------- MAIN ----------------------
def main():
    ensure_table_exists(REGISTRATION_LOG_TABLE)

    try:
        model_types = get_models_to_register()
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"\n📋 Models to register: {model_types}")

    overall = {"registered": 0, "skipped": 0, "total": 0}

    for m in model_types:
        stats = process_model_type(m)
        overall["registered"] += stats["registered"]
        overall["skipped"] += stats["skipped"]
        overall["total"] += stats["total"]

    print("\n" + "=" * 80)
    print("🎉 ALL MODEL REGISTRATION COMPLETED")
    print("=" * 80)
    print(f"✅ Total Registered: {overall['registered']}")
    print(f"⚠️ Total Skipped: {overall['skipped']}")
    print(f"📊 Total Processed: {overall['total']}")
    print(f"📌 Registration Log Table: {REGISTRATION_LOG_TABLE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
