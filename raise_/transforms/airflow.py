"""
Raise Airflow Integration

Airflow DAG generator for transform jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING
from textwrap import dedent, indent

from raise_.transforms.orchestrator import (
    Orchestrator,
    OrchestratorType,
    DeploymentResult,
    JobOrchestratorStatus,
)
from raise_.transforms.schedule import ScheduleType

if TYPE_CHECKING:
    from raise_.transforms.job import Job


@dataclass
class AirflowConfig:
    """
    Configuration for Airflow integration.

    Attributes:
        dag_folder: Path to Airflow DAGs folder
        default_args: Default DAG arguments
        catchup: Whether to backfill missed runs
        max_active_runs: Maximum concurrent DAG runs
        tags: Default tags for generated DAGs
        pool: Airflow pool to use
        queue: Airflow queue to use
        owner: Default owner for tasks
    """

    dag_folder: str = "/opt/airflow/dags"
    default_args: dict[str, Any] = field(default_factory=dict)
    catchup: bool = False
    max_active_runs: int = 1
    tags: list[str] = field(default_factory=lambda: ["raise", "feature-store"])
    pool: str | None = None
    queue: str | None = None
    owner: str = "raise"

    def __post_init__(self):
        if not self.default_args:
            self.default_args = {
                "owner": self.owner,
                "depends_on_past": False,
                "email_on_failure": False,
                "email_on_retry": False,
                "retries": 3,
                "retry_delay": timedelta(minutes=5),
            }


@dataclass
class AirflowOrchestrator(Orchestrator):
    """
    Airflow orchestrator for deploying jobs as DAGs.

    Generates Airflow DAG Python files from Job definitions.
    """

    name: str = "airflow"
    config: AirflowConfig = field(default_factory=AirflowConfig)
    airflow_url: str = "http://localhost:8080"

    # Internal state
    _deployed_jobs: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def orchestrator_type(self) -> str:
        return OrchestratorType.AIRFLOW

    def deploy(self, job: Job) -> DeploymentResult:
        """
        Deploy a job as an Airflow DAG.

        Generates a DAG file and writes it to the DAG folder.
        """
        try:
            dag_code = self.generate_definition(job)
            dag_id = self._get_dag_id(job)
            dag_path = f"{self.config.dag_folder}/{dag_id}.py"

            # In production, this would write to the actual file system
            # For now, we store in memory
            self._deployed_jobs[job.id] = dag_code

            return DeploymentResult(
                success=True,
                job_id=job.id,
                orchestrator_id=dag_id,
                message=f"Generated DAG: {dag_id}",
                url=f"{self.airflow_url}/dags/{dag_id}",
            )

        except Exception as e:
            return DeploymentResult(
                success=False,
                job_id=job.id,
                message=f"Failed to deploy: {e}",
            )

    def undeploy(self, job: Job) -> bool:
        """Remove a job's DAG."""
        if job.id in self._deployed_jobs:
            del self._deployed_jobs[job.id]
            return True
        return False

    def trigger(self, job: Job, execution_date: datetime | None = None) -> str:
        """
        Trigger a DAG run via Airflow API.

        In production, this would call the Airflow REST API.
        """
        dag_id = self._get_dag_id(job)
        run_id = f"manual__{datetime.now().isoformat()}"

        # In production: POST to /api/v1/dags/{dag_id}/dagRuns
        # For now, we simulate
        return run_id

    def get_status(self, job: Job) -> JobOrchestratorStatus:
        """Get DAG status from Airflow."""
        dag_id = self._get_dag_id(job)
        deployed = job.id in self._deployed_jobs

        # In production, this would query Airflow API
        return JobOrchestratorStatus(
            deployed=deployed,
            enabled=deployed,
            orchestrator_id=dag_id,
            orchestrator_url=f"{self.airflow_url}/dags/{dag_id}" if deployed else None,
        )

    def generate_definition(self, job: Job) -> str:
        """
        Generate Airflow DAG Python code from a Job.

        Returns:
            Python code string for the DAG file.
        """
        dag_id = self._get_dag_id(job)
        schedule = self._get_schedule_expression(job)
        tags = self.config.tags + job.tags

        # Generate the DAG code
        code = dedent(f'''
            """
            Auto-generated Airflow DAG for Raise Feature Store
            Job: {job.name}
            Generated: {datetime.now().isoformat()}

            DO NOT EDIT MANUALLY - Changes will be overwritten.
            """

            from datetime import datetime, timedelta
            from airflow import DAG
            from airflow.operators.python import PythonOperator
            from airflow.operators.empty import EmptyOperator
            from airflow.utils.dates import days_ago

            # Import Raise SDK
            try:
                from raise_ import FeatureStore
                from raise_.transforms import Job
            except ImportError:
                pass  # SDK may not be installed in Airflow environment


            # Default arguments
            default_args = {self._format_dict(self.config.default_args)}

            # Job configuration (serialized)
            JOB_CONFIG = {self._format_dict(job.to_dict())}


            def execute_transform(**context):
                """Execute the Raise transformation."""
                from raise_ import FeatureStore
                from raise_.transforms import Job

                # Reconstruct job from config
                job = Job.from_dict(JOB_CONFIG)

                # Get execution date from Airflow context
                execution_date = context.get('execution_date') or context.get('logical_date')

                # Run the job
                run_result = job.run(execution_date=execution_date)

                # Push metrics to XCom
                context['ti'].xcom_push(key='rows_read', value=run_result.rows_read)
                context['ti'].xcom_push(key='rows_written', value=run_result.rows_written)
                context['ti'].xcom_push(key='status', value=run_result.status.value)

                if run_result.status.value == 'failed':
                    raise Exception(f"Job failed: {{run_result.error}}")

                return run_result.to_dict()


            def run_quality_checks(**context):
                """Run data quality checks after transformation."""
                # Get transform result from previous task
                ti = context['ti']
                rows_written = ti.xcom_pull(task_ids='transform', key='rows_written')

                # Run quality checks
                # In production, this would execute the configured quality checks
                print(f"Quality check: {{rows_written}} rows written")

                return {{"passed": True}}


            # Define the DAG
            with DAG(
                dag_id="{dag_id}",
                description="""{job.description or f"Feature transformation job: {job.name}"}""",
                default_args=default_args,
                schedule_interval={schedule},
                start_date=days_ago(1),
                catchup={str(self.config.catchup)},
                max_active_runs={self.config.max_active_runs},
                tags={tags},
            ) as dag:

                # Start marker
                start = EmptyOperator(
                    task_id='start',
                )

                # Main transformation task
                transform = PythonOperator(
                    task_id='transform',
                    python_callable=execute_transform,
                    provide_context=True,
                    {self._get_pool_queue_args()}
                )

                # Quality checks task
                quality_checks = PythonOperator(
                    task_id='quality_checks',
                    python_callable=run_quality_checks,
                    provide_context=True,
                )

                # End marker
                end = EmptyOperator(
                    task_id='end',
                )

                # Define task dependencies
                start >> transform >> quality_checks >> end

        ''').strip()

        return code

    def _get_dag_id(self, job: Job) -> str:
        """Generate DAG ID from job name."""
        # Replace invalid characters
        dag_id = job.name.replace(" ", "_").replace("-", "_").lower()
        return f"raise_{dag_id}"

    def _get_schedule_expression(self, job: Job) -> str:
        """Convert job schedule to Airflow schedule expression."""
        schedule = job.schedule

        if schedule.schedule_type == ScheduleType.CRON:
            return f'"{schedule.to_cron()}"'
        elif schedule.schedule_type == ScheduleType.INTERVAL:
            cron = schedule.to_cron()
            if cron:
                return f'"{cron}"'
            # Fall back to timedelta
            return f"timedelta(seconds={int(schedule.timedelta.total_seconds())})"
        elif schedule.schedule_type == ScheduleType.MANUAL:
            return "None"
        elif schedule.schedule_type == ScheduleType.ON_CHANGE:
            # On-change triggers require external sensors
            return "None  # Triggered externally via CDC"
        else:
            return "None"

    def _get_pool_queue_args(self) -> str:
        """Generate pool and queue arguments for tasks."""
        args = []
        if self.config.pool:
            args.append(f'pool="{self.config.pool}"')
        if self.config.queue:
            args.append(f'queue="{self.config.queue}"')
        return ",\n                    ".join(args) if args else ""

    def _format_dict(self, d: dict[str, Any], indent_level: int = 0) -> str:
        """Format a dictionary as Python code."""
        import json

        # Use JSON for serialization, then clean up for Python
        json_str = json.dumps(d, indent=4, default=str)

        # Convert JSON booleans to Python
        json_str = json_str.replace(": true", ": True")
        json_str = json_str.replace(": false", ": False")
        json_str = json_str.replace(": null", ": None")

        return json_str

    def get_dag_code(self, job: Job) -> str | None:
        """Get the generated DAG code for a job."""
        return self._deployed_jobs.get(job.id)


@dataclass
class AirflowTaskGroup:
    """
    Helper for generating Airflow TaskGroups.

    Used when a job has multiple stages or complex dependencies.
    """

    group_id: str
    tasks: list[str] = field(default_factory=list)
    tooltip: str | None = None

    def generate_code(self) -> str:
        """Generate TaskGroup code."""
        tasks_code = "\n".join(f"            {task}" for task in self.tasks)
        return dedent(f'''
            with TaskGroup(group_id="{self.group_id}", tooltip="{self.tooltip or self.group_id}") as {self.group_id}:
{tasks_code}
        ''').strip()


def generate_airflow_dag(job: Job, config: AirflowConfig | None = None) -> str:
    """
    Convenience function to generate an Airflow DAG from a job.

    Args:
        job: The job to convert to a DAG
        config: Optional Airflow configuration

    Returns:
        Python code string for the DAG
    """
    orchestrator = AirflowOrchestrator(config=config or AirflowConfig())
    return orchestrator.generate_definition(job)
