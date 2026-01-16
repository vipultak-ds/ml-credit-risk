import os
import sys
import time
import json  # ✅ ADDED for serving config
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.jobs import TaskDependency
 
# 1️⃣ Initialize Databricks Client
try:
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")

    if not host or not token:
        raise EnvironmentError("DATABRICKS_HOST or DATABRICKS_TOKEN not set.")

    w = WorkspaceClient(host=host, token=token)
    print("✅ Databricks client initialized successfully")
    print(f"🔗 Workspace: {host}")
except Exception as e:
    print(f"❌ Databricks client initialization failed: {e}")
    sys.exit(1)

# =====================================================================
# 2️⃣ Configuration & Git Variables
# =====================================================================

# ✅ Get MODELS_TO_TRAIN from environment (set by GitHub Actions)
MODELS_TO_TRAIN = os.getenv("MODELS_TO_TRAIN", "").strip()

# ✅ Validate early - before creating any jobs
if not MODELS_TO_TRAIN or MODELS_TO_TRAIN.lower() in ["none", "null", "undefined", ""]:
    print("\n" + "=" * 70)
    print("❌ ERROR: MODELS_TO_TRAIN is not set!")
    print("=" * 70)
    print("\n📋 Available options:")
    print("   1. Set GitHub repository variable:")
    print("      Settings → Secrets and variables → Actions → Variables")
    print("      Name: MODELS_TO_TRAIN")
    print("      Value: random_forest,xgboost  (or 'all')")
    print("\n   2. Or use workflow_dispatch to manually trigger with models")
    print("\n   3. Models should match those in experiments_config.yml")
    print("\n💡 Example values:")
    print("   • 'random_forest,xgboost,logistic_regression'")
    print("   • 'random_forest'")
    print("   • 'all' (trains all models in experiments_config.yml)")
    print("=" * 70)
    sys.exit(1)

# ✅ Handle "all" keyword or parse comma-separated list
if MODELS_TO_TRAIN.lower() == "all":
    models_list = ["all"]
    print(f"📋 Training ALL models from experiments_config.yml")
else:
    models_list = [m.strip() for m in MODELS_TO_TRAIN.split(",") if m.strip()]
    if not models_list:
        print(f"❌ ERROR: No valid models found in MODELS_TO_TRAIN='{MODELS_TO_TRAIN}'")
        sys.exit(1)

# ✅ NEW: Get and validate MODEL_SERVING_CONFIG
JOB_SCHEDULE_CRON = os.getenv("JOB_SCHEDULE_CRON", "disabled")
MODEL_SERVING_CONFIG = os.getenv("MODEL_SERVING_CONFIG", "{}").strip()

# Validate serving config is valid JSON
try:
    if MODEL_SERVING_CONFIG and MODEL_SERVING_CONFIG != "{}":
        serving_config_dict = json.loads(MODEL_SERVING_CONFIG)
        print(f"✅ Serving config loaded: {len(serving_config_dict)} model(s) configured")
    else:
        serving_config_dict = {}
        print(f"⚠️  No serving config provided (using empty config)")
except json.JSONDecodeError as e:
    print(f"❌ ERROR: Invalid JSON in MODEL_SERVING_CONFIG")
    print(f"   Error: {e}")
    print(f"   Value: {MODEL_SERVING_CONFIG[:100]}...")
    sys.exit(1)

repo_name = "ml-credit-risk"
repo_path = f"/Repos/vipultak7171@gmail.com/{repo_name}"

print("\n" + "=" * 60)
print("🚀 MLOPS PIPELINE ORCHESTRATION (Serverless)")
print("=" * 60)
print(f"📁 Repository Path: {repo_path}")
print(f"📋 Models to train: {MODELS_TO_TRAIN}")
if MODELS_TO_TRAIN.lower() != "all":
    print(f"   Parsed as: {', '.join(models_list)}")
print(f"📅 Job Schedule: {JOB_SCHEDULE_CRON}")
print(f"🌐 Serving Config: {len(serving_config_dict)} model(s)")
if serving_config_dict:
    for model_key, config in serving_config_dict.items():
        active_status = "✅" if config.get('active', False) else "⚪"
        print(f"   {active_status} {model_key}: {config.get('model_name', 'N/A')} (traffic: {config.get('traffic', 0)}%)")
print("-" * 60 + "\n")

# =====================================================================
# 3️⃣ Helper Functions (UNCHANGED)
# =====================================================================

def get_job_id(job_name):
    for j in w.jobs.list():
        if j.settings.name == job_name:
            return j.job_id
    return None


def create_or_update_job(job_name, tasks, auto_run=False):
    existing_job_id = get_job_id(job_name)

    job_settings = jobs.JobSettings(
        name=job_name,
        tasks=tasks,
        max_concurrent_runs=1
    )

    if existing_job_id:
        print(f"🔄 Updating job: {job_name}")
        w.jobs.reset(job_id=existing_job_id, new_settings=job_settings)
        job_id = existing_job_id
    else:
        print(f"➕ Creating job: {job_name}")
        job_id = w.jobs.create(
            name=job_name,
            tasks=tasks,
            max_concurrent_runs=1
        ).job_id

    if auto_run:
        run_id = w.jobs.run_now(job_id=job_id).run_id
        return job_id, run_id

    return job_id, None


def wait_for_job_completion(job_id, run_id, job_name, timeout_minutes=30):
    start = time.time()
    while time.time() - start < timeout_minutes * 60:
        run = w.jobs.get_run(run_id)
        state = run.state.life_cycle_state.value

        if state == "TERMINATED":
            return run.state.result_state.value == "SUCCESS"

        time.sleep(15)

    return False


def handle_job_failure(job_name, stage):
    print(f"\n❌ PIPELINE FAILED AT {stage}")
    print(f"Job failed: {job_name}")
    sys.exit(1)

# =====================================================================
# 4️⃣ DEV JOB — ✅ PARALLEL MODEL TRAINING (ACTIVE)
# =====================================================================

print("\n[STEP 1/3] 🛠️ Creating DEV Training Pipeline...")
print("-" * 60)

dev_tasks = []

# ✅ Handle "all" keyword vs specific models
if MODELS_TO_TRAIN.lower() == "all":
    # Create single task that will train all models
    dev_tasks.append(
        jobs.Task(
            task_key="train_all_models",
            notebook_task=jobs.NotebookTask(
                notebook_path=f"{repo_path}/dev_env/train",
                base_parameters={
                    "environment": "development",
                    "MODELS_TO_TRAIN": "all"
                }
            )
        )
    )
    print("   📦 Created task: train_all_models (will train all models from config)")
    
else:
    # Create parallel tasks for each model
    for model in models_list:
        dev_tasks.append(
            jobs.Task(
                task_key=f"train_{model}",
                notebook_task=jobs.NotebookTask(
                    notebook_path=f"{repo_path}/dev_env/train",
                    base_parameters={
                        "environment": "development",
                        "MODELS_TO_TRAIN": model
                    }
                )
            )
        )
        print(f"   📦 Created task: train_{model}")

# ✅ Registration depends on all training tasks
if MODELS_TO_TRAIN.lower() == "all":
    registration_depends_on = [TaskDependency(task_key="train_all_models")]
else:
    registration_depends_on = [
        TaskDependency(task_key=f"train_{model}") 
        for model in models_list
    ]

dev_tasks.append(
    jobs.Task(
        task_key="model_registration_task",
        notebook_task=jobs.NotebookTask(
            notebook_path=f"{repo_path}/dev_env/register",
            base_parameters={
                "environment": "development",
                "MODELS_TO_TRAIN": MODELS_TO_TRAIN
            }
        ),
        depends_on=registration_depends_on
    )
)
print("   📦 Created task: model_registration_task")

# ✅ Serving endpoint creation (depends on registration)
# IMPORTANT: Pass MODEL_SERVING_CONFIG to notebook
dev_tasks.append(
    jobs.Task(
        task_key="create_serving_endpoint_task",
        notebook_task=jobs.NotebookTask(
            notebook_path=f"{repo_path}/dev_env/create_serving_endpoint_env_var",
            base_parameters={
                "environment": "development",
                "MODEL_SERVING_CONFIG": MODEL_SERVING_CONFIG  # ✅ Pass the config
            }
        ),
        depends_on=[
            TaskDependency(task_key="model_registration_task")
        ]
    )
)
print("   📦 Created task: create_serving_endpoint_task")

dev_job_id, dev_run_id = create_or_update_job(
    "1. dev-ml-training-pipeline",
    tasks=dev_tasks,
    auto_run=True
)

print(f"\n✅ DEV job created/updated: Job ID {dev_job_id}")
print(f"🚀 Training started: Run ID {dev_run_id}")

if not wait_for_job_completion(dev_job_id, dev_run_id, "DEV Training", 25):
    handle_job_failure("DEV Training Pipeline", "DEVELOPMENT")

# =====================================================================
# 5️⃣ UAT JOB (COMMENTED - UNCHANGED)
# =====================================================================

# print("\n[STEP 2/3] 🧪 Creating UAT Pipeline...")
# print("-" * 60)
#
# uat_tasks = [
#     jobs.Task(
#         task_key="model_staging_task",
#         notebook_task=jobs.NotebookTask(
#             notebook_path=f"{repo_path}/uat_env/uat_staging",
#             base_parameters={"alias": "Staging"}
#         )
#     ),
#     jobs.Task(
#         task_key="inference_test_task",
#         notebook_task=jobs.NotebookTask(
#             notebook_path=f"{repo_path}/uat_env/uat_inference",
#             base_parameters={"alias": "Staging", "environment": "uat"}
#         ),
#         depends_on=[TaskDependency(task_key="model_staging_task")]
#     )
# ]
#
# uat_job_id, uat_run_id = create_or_update_job(
#     "2. uat-ml-inference-pipeline",
#     tasks=uat_tasks,
#     auto_run=True
# )
#
# if not wait_for_job_completion(uat_job_id, uat_run_id, "UAT Pipeline", 20):
#     handle_job_failure("UAT Pipeline", "UAT")

# =====================================================================
# 6️⃣ PROD JOB (COMMENTED - UNCHANGED)
# =====================================================================

# print("\n[STEP 3/3] 🚀 Creating PROD Deployment Pipeline...")
# print("-" * 60)
#
# prod_tasks = [
#     jobs.Task(
#         task_key="model_promotion_task",
#         notebook_task=jobs.NotebookTask(
#             notebook_path=f"{repo_path}/prod_env/prod_promotion",
#             base_parameters={"alias": "Production", "action": "promote"}
#         )
#     ),
#     jobs.Task(
#         task_key="serving_endpoint_task",
#         notebook_task=jobs.NotebookTask(
#             notebook_path=f"{repo_path}/prod_env/prod_create_serving",
#             base_parameters={"environment": "prod"}
#         ),
#         depends_on=[TaskDependency(task_key="model_promotion_task")]
#     ),
#     jobs.Task(
#         task_key="model_inference_production",
#         notebook_task=jobs.NotebookTask(
#             notebook_path=f"{repo_path}/prod_env/prod_inference",
#             base_parameters={"alias": "Production", "environment": "prod"}
#         ),
#         depends_on=[TaskDependency(task_key="serving_endpoint_task")]
#     )
# ]
#
# prod_job_id, prod_run_id = create_or_update_job(
#     "3. prod-ml-deployment-pipeline",
#     tasks=prod_tasks,
#     auto_run=True
# )
#
# if not wait_for_job_completion(prod_job_id, prod_run_id, "PROD Pipeline", 30):
#     handle_job_failure("PROD Pipeline", "PRODUCTION")

# =====================================================================
# 7️⃣ SUCCESS
# =====================================================================

print("\n🎉 DEV MLOPS PIPELINE COMPLETED SUCCESSFULLY!")
print(f"Models configured: {MODELS_TO_TRAIN}")
if MODELS_TO_TRAIN.lower() != "all":
    print(f"Models trained in parallel: {', '.join(models_list)}")
if serving_config_dict:
    print(f"Serving configured for: {', '.join(serving_config_dict.keys())}")
sys.exit(0)
 
### for sequantial job and task 

# import os
# import sys
# import time
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service import jobs
# from databricks.sdk.service.jobs import TaskDependency
 
# # 1️⃣ Initialize Databricks Client

# try:
#     # Ensure env vars are loaded
#     host = os.getenv("DATABRICKS_HOST")
#     token = os.getenv("DATABRICKS_TOKEN")

#     if not host or not token:
#         raise EnvironmentError("DATABRICKS_HOST or DATABRICKS_TOKEN not set.")

#     # Initialize with explicit host/token (important for CI/CD)
#     w = WorkspaceClient(host=host, token=token)
#     print("✅ Databricks client initialized successfully")
#     print(f"🔗 Workspace: {host}")
# except Exception as e:
#     print(f"❌ Databricks client initialization failed: {e}")
#     sys.exit(1)  # ✅ Exit immediately on connection failure

# # =====================================================================
# # 2️⃣ Configuration & Git Variables
# # =====================================================================

# # 🔥 Get Git Variables
# MODELS_TO_TRAIN = os.getenv("MODELS_TO_TRAIN", "random_forest,xgboost")
# JOB_SCHEDULE_CRON = os.getenv("JOB_SCHEDULE_CRON", "disabled")
# MODEL_SERVING_CONFIG = os.getenv("MODEL_SERVING_CONFIG", "{}")

# # Parse models list
# models_list = [m.strip() for m in MODELS_TO_TRAIN.split(",")]

# repo_name = "ml-credit-risk"
# repo_path = f"/Repos/vipultak7171@gmail.com/{repo_name}"

# print("\n" + "=" * 60)
# print("🚀 MLOPS PIPELINE ORCHESTRATION (Serverless)")
# print("=" * 60)
# print(f"📁 Repository Path: {repo_path}")
# print(f"📋 Models to train: {', '.join(models_list)}")
# print(f"📅 Job Schedule: {JOB_SCHEDULE_CRON}")
# print("-" * 60 + "\n")

# # =====================================================================
# # 3️⃣ Helper Functions
# # =====================================================================

# def get_job_id(job_name):
#     """Get existing Databricks job ID by name."""
#     try:
#         for j in w.jobs.list():
#             if j.settings.name == job_name:
#                 return j.job_id
#         return None
#     except Exception as e:
#         print(f"⚠️ Error while listing jobs: {e}")
#         return None


# def create_or_update_job(job_name, tasks, auto_run=False):
#     """Create or update Databricks job with the given tasks."""
#     try:
#         existing_job_id = get_job_id(job_name)
#         job_settings = jobs.JobSettings(
#             name=job_name,
#             tasks=tasks,
#             max_concurrent_runs=1
#         )

#         if existing_job_id:
#             print(f"🔄 Updating job: {job_name} (ID: {existing_job_id})")
#             w.jobs.reset(job_id=existing_job_id, new_settings=job_settings)
#             job_id = existing_job_id
#         else:
#             print(f"➕ Creating new job: {job_name}")
#             result = w.jobs.create(
#                 name=job_settings.name,
#                 tasks=job_settings.tasks,
#                 max_concurrent_runs=1
#             )
#             job_id = result.job_id

#         print(f"✅ Job ready: {job_name} (ID: {job_id})")

#         if auto_run:
#             print(f"🚀 Triggering job run: {job_name}")
#             run_result = w.jobs.run_now(job_id=job_id)
#             print(f"   📋 Run ID: {run_result.run_id}")
#             return job_id, run_result.run_id

#         return job_id, None

#     except Exception as e:
#         print(f"❌ CRITICAL ERROR in job '{job_name}': {e}")
#         print(f"   This error prevents the pipeline from continuing.")
#         # ✅ Return None to indicate failure
#         return None, None


# def wait_for_job_completion(job_id, run_id, job_name, timeout_minutes=30):
#     """
#     Wait for Databricks job completion.
#     Returns: True if successful, False if failed/timeout
#     """
#     if not run_id:
#         print(f"❌ No run_id provided for {job_name}. Cannot wait for completion.")
#         return False

#     print(f"\n⏳ Waiting for {job_name} to complete (max {timeout_minutes} min)...")
#     start_time = time.time()
#     last_state = None

#     while time.time() - start_time < timeout_minutes * 60:
#         try:
#             run_info = w.jobs.get_run(run_id=run_id)
#             state = run_info.state.life_cycle_state.value if run_info.state else "UNKNOWN"

#             if state != last_state:
#                 elapsed = int(time.time() - start_time)
#                 print(f"   📊 [{elapsed}s] State: {state}")
#                 last_state = state

#             if state == "TERMINATED":
#                 result_state = run_info.state.result_state.value if run_info.state else "UNKNOWN"
                
#                 if result_state == "SUCCESS":
#                     print(f"✅ {job_name} completed successfully.")
#                     return True
#                 else:
#                     print(f"❌ {job_name} FAILED!")
#                     print(f"   Result State: {result_state}")
                    
#                     # Print error message if available
#                     if run_info.state.state_message:
#                         print(f"   Error Message: {run_info.state.state_message}")
                    
#                     return False

#             time.sleep(15)
            
#         except Exception as e:
#             print(f"⚠️ Error checking job status: {e}")
#             time.sleep(30)

#     print(f"⏰ TIMEOUT waiting for {job_name} (exceeded {timeout_minutes} minutes).")
#     return False


# def handle_job_failure(job_name, stage):
#     """
#     Handle job failure and exit pipeline
#     """
#     print("\n" + "=" * 60)
#     print(f"❌ PIPELINE FAILED AT {stage}")
#     print("=" * 60)
#     print(f"Job '{job_name}' did not complete successfully.")
#     print("\nReason for pipeline stop:")
#     print("  • Job execution failed or timed out")
#     print("  • Cannot proceed to next stage with failed dependencies")
#     print("  • Code or configuration issues need to be resolved")
#     print("\n📋 Troubleshooting Steps:")
#     print("  1. Check Databricks UI for detailed error logs")
#     print(f"  2. Review notebook code in: {repo_path}")
#     print("  3. Verify data availability and model parameters")
#     print("  4. Fix issues and re-run the pipeline")
#     print("\n🔗 Databricks Workspace: " + os.getenv("DATABRICKS_HOST"))
#     print("=" * 60)
#     sys.exit(1)  # ✅ Exit with error code


# # =====================================================================
# # 4️⃣ DEV JOB (Data Ingest -> Train -> Register)
# # =====================================================================
# print("\n[STEP 1/3] 🛠️ Creating DEV Training Pipeline...")
# print("-" * 60)

# dev_tasks = [
#     jobs.Task(
#         task_key="model_training_task",
#         notebook_task=jobs.NotebookTask(
#             notebook_path=f"{repo_path}/dev_env/train",
#             base_parameters={"environment": "development"}
#         )
#     ),
    
#     jobs.Task(
#         task_key="model_registration_task",
#         notebook_task=jobs.NotebookTask(
#             notebook_path=f"{repo_path}/dev_env/register",
#             base_parameters={"environment": "development"}
#         ),
#         depends_on=[TaskDependency(task_key="model_training_task")]
#     )
# ]

# dev_job_id, dev_run_id = create_or_update_job(
#     "1. dev-ml-training-pipeline",
#     tasks=dev_tasks,
#     auto_run=True
# )

# # ✅ Check if job creation failed
# if dev_job_id is None or dev_run_id is None:
#     print("\n❌ CRITICAL: Failed to create or trigger DEV job")
#     print("   Cannot proceed with pipeline execution")
#     sys.exit(1)

# # ✅ Wait for completion and exit if failed
# dev_success = wait_for_job_completion(dev_job_id, dev_run_id, "DEV Training Pipeline", 25)
# if not dev_success:
#     handle_job_failure("DEV Training Pipeline", "DEVELOPMENT STAGE")

# # =====================================================================
# # 5️⃣ UAT JOB (Model Staging -> Inference Test)
# # =====================================================================
# print("\n[STEP 2/3] 🧪 Creating UAT Pipeline...")
# print("-" * 60)

# uat_tasks = [
#     jobs.Task(
#         task_key="model_staging_task",
#         notebook_task=jobs.NotebookTask(
#             notebook_path=f"{repo_path}/uat_env/uat_staging",
#             base_parameters={"alias": "Staging"}
#         )
#     ),
#     jobs.Task(
#         task_key="inference_test_task",
#         notebook_task=jobs.NotebookTask(
#             notebook_path=f"{repo_path}/uat_env/uat_inference",
#             base_parameters={"alias": "Staging", "environment": "uat"}
#         ),
#         depends_on=[TaskDependency(task_key="model_staging_task")]
#     )
# ]

# uat_job_id, uat_run_id = create_or_update_job(
#     "2. uat-ml-inference-pipeline",
#     tasks=uat_tasks,
#     auto_run=True
# )

# # ✅ Check if job creation failed
# if uat_job_id is None or uat_run_id is None:
#     print("\n❌ CRITICAL: Failed to create or trigger UAT job")
#     print("   DEV completed but UAT cannot proceed")
#     sys.exit(1)

# # ✅ Wait for completion and exit if failed
# uat_success = wait_for_job_completion(uat_job_id, uat_run_id, "UAT Pipeline", 20)
# if not uat_success:
#     handle_job_failure("UAT Pipeline", "UAT STAGE")

# # =====================================================================
# # 6️⃣ PROD JOB (Promotion -> Serving -> Inference)
# # =====================================================================
# print("\n[STEP 3/3] 🚀 Creating PROD Deployment Pipeline...")
# print("-" * 60)

# prod_tasks = [
#     jobs.Task(
#         task_key="model_promotion_task",
#         notebook_task=jobs.NotebookTask(
#             notebook_path=f"{repo_path}/prod_env/prod_promotion",
#             base_parameters={"alias": "Production", "action": "promote"}
#         )
#     ),
#     jobs.Task(
#         task_key="serving_endpoint_task",
#         notebook_task=jobs.NotebookTask(
#             notebook_path=f"{repo_path}/prod_env/prod_create_serving",
#             base_parameters={"environment": "prod"}
#         ),
#         depends_on=[TaskDependency(task_key="model_promotion_task")]
#     ),
#     jobs.Task(
#         task_key="model_inference_production",
#         notebook_task=jobs.NotebookTask(
#             notebook_path=f"{repo_path}/prod_env/prod_inference",
#             base_parameters={"alias": "Production", "environment": "prod"}
#         ),
#         depends_on=[TaskDependency(task_key="serving_endpoint_task")]
#     )
# ]

# prod_job_id, prod_run_id = create_or_update_job(
#     "3. prod-ml-deployment-pipeline",
#     tasks=prod_tasks,
#     auto_run=True
# )

# # ✅ Check if job creation failed
# if prod_job_id is None or prod_run_id is None:
#     print("\n❌ CRITICAL: Failed to create or trigger PROD job")
#     print("   DEV and UAT completed but PROD cannot proceed")
#     sys.exit(1)

# # ✅ Wait for completion and exit if failed
# prod_success = wait_for_job_completion(prod_job_id, prod_run_id, "PROD Deployment Pipeline", 30)
# if not prod_success:
#     handle_job_failure("PROD Deployment Pipeline", "PRODUCTION STAGE")

# # =====================================================================
# # 7️⃣ SUCCESS SUMMARY
# # =====================================================================
# print("\n" + "=" * 60)
# print("🎉 MLOPS PIPELINE COMPLETED SUCCESSFULLY!")
# print("=" * 60)
# print("\n✅ All stages completed:")
# print("   1. ✓ DEV Training Pipeline")
# print("   2. ✓ UAT Inference Pipeline")
# print("   3. ✓ PROD Deployment Pipeline")
# print("\n📊 Pipeline Summary:")
# print(f"   • DEV Job ID: {dev_job_id}")
# print(f"   • UAT Job ID: {uat_job_id}")
# print(f"   • PROD Job ID: {prod_job_id}")
# print(f"   • Models Trained: {', '.join(models_list)}")
# print("\n🔗 View results in Databricks:")
# print(f"   {os.getenv('DATABRICKS_HOST')}")
# print("=" * 60)

# # ✅ Exit with success code
# sys.exit(0)