from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, round, regexp_replace, trim
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import mlflow

# CONFIG
INPUT_DELTA_TABLE = "workspace.new.credit"
OUTPUT_DELTA_TABLE = "workspace.new.credit_preprocessed"
MODEL_REGISTRY_PATH = "/Volumes/workspace/new/ml_feature_artifacts/preprocessing_pipeline"

# SPARK INITIALIZATION
def initialize_spark():
    if 'spark' in globals() and isinstance(globals()['spark'], SparkSession):
        return globals()['spark']
    return SparkSession.builder.appName("CreditRiskPreprocessing").getOrCreate()

# STEP 1: DATA INGESTION

def ingest_data(spark: SparkSession, table_name: str):
    print(f"📥 Loading: {table_name}")
    
    credit_df = spark.read.format("delta").table(table_name)

    feature_cols = [
        'checking_balance', 'months_loan_duration', 'credit_history', 'purpose',
        'amount', 'savings_balance', 'employment_duration', 'percent_of_income',
        'years_at_residence', 'age', 'other_credit', 'housing',
        'existing_loans_count', 'job', 'dependents', 'phone'
    ]
    
    df = credit_df.select("id", "start_date", "end_date", *feature_cols, col("default").alias("label"))
    print(f"✅ Loaded Rows: {df.count():,}")
    return df

# NEW STEP: FETCH ONLY NEW RECORDS (INCREMENTAL LOGIC)

def fetch_new_records(spark: SparkSession, raw_df, output_table: str):
    print("🔍 Checking for new records...")

    try:
        existing_df = spark.read.table(output_table).select("id", "start_date", "end_date").dropDuplicates()

        new_df = raw_df.join(
            existing_df,
            on=["id", "start_date", "end_date"],
            how="left_anti"
        )

        print(f"🆕 New records found: {new_df.count():,}")
        return new_df

    except Exception as e:
        print(f"⚠️ Output table not found or unreadable. Full load will run. Reason: {e}")
        return raw_df

# STEP 2: UNIT CLEANUP

def cleanup_units(df):
    print("🧹 Cleaning units...")

    df = df.withColumn("checking_balance", trim(regexp_replace(col("checking_balance"), r"\s*DM\s*$", "")))
    df = df.withColumn("savings_balance", trim(regexp_replace(col("savings_balance"), r"\s*DM\s*$", "")))
    df = df.withColumn("employment_duration", trim(regexp_replace(col("employment_duration"), r"\s*years?\s*$", "")))

    df = df.replace("", "unknown")
    
    print("✅ Unit cleanup done")
    return df

# STEP 3: DATA PREPARATION

def prepare_data(df):
    print("📊 Preparing data...")

    df = df.withColumn("label", when(col("label") == "yes", 1.0).otherwise(0.0))

    df = df.withColumn(
        "monthly_income",
        round(
            when(
                (col("percent_of_income") > 0) & (col("months_loan_duration") > 0),
                (col("amount") / col("months_loan_duration")) * (100 / col("percent_of_income"))
            ).otherwise(None),
            2
        )
    )

    print("✅ Preparation complete")
    return df

# STEP 4: ORDINAL ENCODING

def ordinal_encoding(df):
    print("🔢 Applying ordinal encoding...")

    ordinal_config = {
        'checking_balance': ['< 0', '1 - 200', '> 200', 'unknown'],
        'savings_balance': ['< 100', '100 - 500', '500 - 1000', '> 1000', 'unknown'],
        'employment_duration': ['unemployed', '< 1', '1 - 4', '4 - 7', '> 7', 'unknown'],
        'credit_history': ['critical', 'poor', 'good', 'very good', 'perfect']
    }

    for col_name, categories in ordinal_config.items():
        expr = None
        for idx, cat in enumerate(categories):
            expr = when(col(col_name) == cat, float(idx)) if expr is None else expr.when(col(col_name) == cat, float(idx))
        df = df.withColumn(col_name, expr.otherwise(float(len(categories))))

    print("✅ Ordinal encoding done")
    return df

# STEP 5: ONE-HOT ENCODING

def onehot_encoding(df):
    print("🔥 One-hot encoding...")

    nominal_cols = ['purpose', 'other_credit', 'housing', 'job', 'phone']

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep") for c in nominal_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vec", dropLast=True) for c in nominal_cols]

    pipeline = Pipeline(stages=indexers + encoders)
    model = pipeline.fit(df)
    df = model.transform(df)

    df = df.drop(*nominal_cols, *[f"{c}_index" for c in nominal_cols])

    print("✅ One-hot encoding done")
    return df, model

# STEP 6: STANDARD SCALING

def apply_standard_scaling(df):
    print("📏 Scaling features...")

    exclude_cols = ["id", "start_date", "end_date", "label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="unscaled_features")
    scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withMean=True, withStd=True)

    pipeline = Pipeline(stages=[assembler, scaler])
    model = pipeline.fit(df)
    df = model.transform(df).select("id", "start_date", "end_date", "features", "label")

    print("✅ Scaling complete")
    return df, model

# STEP 7: SAVE TO DELTA TABLE

def save_to_delta(df, table_name: str):
    print(f"💾 Saving to Delta table: {table_name}")
    
    # Append mode - add only new data
    df.write.format("delta").mode("append").saveAsTable(table_name)
    
    row_count = df.count()
    print(f"✅ Saved {row_count:,} rows to {table_name}")
    
    return row_count

# STEP 8: SAVE PIPELINE MODELS (OPTIONAL - FOR INFERENCE)

def save_pipeline_models(onehot_model, scaler_model, model_path: str):
    print(f"💾 Saving preprocessing pipeline to: {model_path}")
    
    try:
        # Save OneHot pipeline
        onehot_model.write().overwrite().save(f"{model_path}/onehot_pipeline")
        
        # Save Scaler pipeline
        scaler_model.write().overwrite().save(f"{model_path}/scaler_pipeline")
        
        print("✅ Pipeline models saved successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not save pipeline models: {e}")

# MAIN EXECUTION

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 CREDIT RISK PREPROCESSING PIPELINE")
    print("="*70 + "\n")
    
    spark = initialize_spark()

    # Step 1: Load full raw data
    df = ingest_data(spark, INPUT_DELTA_TABLE)

    # NEW: Fetch only new records (incremental)
    df = fetch_new_records(spark, df, OUTPUT_DELTA_TABLE)

    # If no new records, stop safely
    new_count = df.count()
    if new_count == 0:
        print("✅ No new records found. Output table will not be updated.")
        dbutils.notebook.exit("No new data - stopping pipeline")


    # Step 2-4: Basic preprocessing
    df = cleanup_units(df)
    df = prepare_data(df)
    df = ordinal_encoding(df)
    
    # Step 5: One-hot encoding (returns model too)
    df, onehot_model = onehot_encoding(df)
    
    # Step 6: Standard scaling (returns model too)
    processed_df, scaler_model = apply_standard_scaling(df)

    # Step 7: Save preprocessed data to Delta table
    row_count = save_to_delta(processed_df, OUTPUT_DELTA_TABLE)
    
    # Step 8: Save pipeline models (optional - for future inference)
    save_pipeline_models(onehot_model, scaler_model, MODEL_REGISTRY_PATH)

    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE!")
    print("="*70)
    print(f"✅ Preprocessed data saved to: {OUTPUT_DELTA_TABLE}")
    print(f"✅ Total rows: {row_count:,}")
    print(f"✅ Pipeline models saved to: {MODEL_REGISTRY_PATH}")
    print("\n📊 Schema:")
    processed_df.printSchema()
    print("\n📋 Sample data:")
    processed_df.display(5, truncate=False)
