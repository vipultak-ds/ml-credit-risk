import os
import sys
import time
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

# ✅ FIX: Define MODELS_TO_TRAIN
MODELS_TO_TRAIN = "all"   # change this if needed like: "random_forest,xgboost"

# ✅ FIX: Create models_list safely
if MODELS_TO_TRAIN.lower() == "all":
    models_list = ["all"]
else:
    models_list = [m.strip() for m in MODELS_TO_TRAIN.split(",") if m.strip()]

repo_name = "ml-credit-risk"
repo_path = f"/Repos/vipultak7171@gmail.com/{repo_name}"

print("\n" + "=" * 60)
print("🚀 MLOPS PIPELINE ORCHESTRATION (Serverless)")
print("=" * 60)
print(f"📁 Repository Path: {repo_path}")
print(f"📋 Models to train: {MODELS_TO_TRAIN}")
print("-" * 60 + "\n")

# =====================================================================
# 3️⃣ Helper Functions
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
# 4️⃣ DEV JOB — Ingestion → Parallel Training → Evaluation → Register
# =====================================================================

print("\n[STEP 1/1] 🛠️ Creating DEV Training Pipeline...")
print("-" * 60)

dev_tasks = []

# ✅ STEP 1: Data Ingestion + Preprocessing Task (FIRST)
dev_tasks.append(
    jobs.Task(
        task_key="data_ingestion_preprocessing",
        notebook_task=jobs.NotebookTask(
            notebook_path=f"{repo_path}/Model_part_2/preprocessing"
        )
    )
)
print("   📦 Created task: data_ingestion_preprocessing")

# ✅ STEP 2: Training Tasks (DEPEND on ingestion)
if MODELS_TO_TRAIN.lower() == "all":
    dev_tasks.append(
        jobs.Task(
            task_key="train_all_models",
            notebook_task=jobs.NotebookTask(
                notebook_path=f"{repo_path}/Model_part_2/train"
            ),
            depends_on=[TaskDependency(task_key="data_ingestion_preprocessing")]
        )
    )
    print("   📦 Created task: train_all_models (depends on ingestion)")
else:
    for model in models_list:
        dev_tasks.append(
            jobs.Task(
                task_key=f"train_{model}",
                notebook_task=jobs.NotebookTask(
                    notebook_path=f"{repo_path}/Model_part_2/train"
                ),
                depends_on=[TaskDependency(task_key="data_ingestion_preprocessing")]
            )
        )
        print(f"   📦 Created task: train_{model} (depends on ingestion)")

# ✅ STEP 3: Model Evaluation depends on training tasks
if MODELS_TO_TRAIN.lower() == "all":
    evaluation_depends_on = [TaskDependency(task_key="train_all_models")]
else:
    evaluation_depends_on = [
        TaskDependency(task_key=f"train_{model}")
        for model in models_list
    ]

dev_tasks.append(
    jobs.Task(
        task_key="model_evaluation_task",
        notebook_task=jobs.NotebookTask(
            notebook_path=f"{repo_path}/Model_part_2/model_evaluation"
        ),
        depends_on=evaluation_depends_on
    )
)
print("   📦 Created task: model_evaluation_task (depends on training)")

# ✅ STEP 4: Registration depends on evaluation task
dev_tasks.append(
    jobs.Task(
        task_key="model_registration_task",
        notebook_task=jobs.NotebookTask(
            notebook_path=f"{repo_path}/Model_part_2/register"
        ),
        depends_on=[TaskDependency(task_key="model_evaluation_task")]
    )
)
print("   📦 Created task: model_registration_task (depends on evaluation)")

dev_job_id, dev_run_id = create_or_update_job(
    "1. dev-ml-training-pipeline",
    tasks=dev_tasks,
    auto_run=True
)

print(f"\n✅ DEV job created/updated: Job ID {dev_job_id}")
print(f"🚀 Pipeline started: Run ID {dev_run_id}")

if not wait_for_job_completion(dev_job_id, dev_run_id, "DEV Training", 25):
    handle_job_failure("DEV Training Pipeline", "DEVELOPMENT")

print("\n🎉 DEV MLOPS PIPELINE COMPLETED SUCCESSFULLY!")
print(f"Models configured: {MODELS_TO_TRAIN}")

sys.exit(0)
