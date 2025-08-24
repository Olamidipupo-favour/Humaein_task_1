#!/usr/bin/env python3
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import random

# Import our data processing modules
import sys
sys.path.append('..')
from data_cleaner import EMRDataCleaner, NormalizedClaim
from main import ResubmissionAnalyzer, ResubmissionCandidate


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Individual task status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class PipelineTask:
    """Represents a single task in the pipeline."""
    name: str
    function: Callable
    dependencies: List[str]
    retries: int = 3
    retry_delay: int = 5
    timeout: int = 300
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    attempt_count: int = 0


@dataclass
class PipelineRun:
    """Represents a pipeline execution run."""
    run_id: str
    pipeline_name: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    tasks: Dict[str, PipelineTask] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None


class PipelineOrchestrator:
    """Simulates a Dagster pipeline orchestrator."""
    
    def __init__(self, name: str = "EMR Data Processing Pipeline"):
        """Initialize the pipeline orchestrator."""
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.runs: Dict[str, PipelineRun] = {}
        self.run_counter = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def create_run(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new pipeline run."""
        run_id = f"run_{self.run_counter:06d}_{int(time.time())}"
        self.run_counter += 1
        
        run = PipelineRun(
            run_id=run_id,
            pipeline_name=self.name,
            status=PipelineStatus.PENDING,
            start_time=datetime.now(),
            tasks={},
            metadata=metadata or {}
        )
        
        self.runs[run_id] = run
        self.logger.info(f"Created pipeline run: {run_id}")
        
        return run_id
    
    def add_task(self, run_id: str, task: PipelineTask) -> None:
        """Add a task to a pipeline run."""
        if run_id not in self.runs:
            raise ValueError(f"Run ID {run_id} not found")
        
        self.runs[run_id].tasks[task.name] = task
        self.logger.info(f"Added task '{task.name}' to run {run_id}")
    
    def execute_pipeline(self, run_id: str) -> PipelineRun:
        """Execute the complete pipeline for a given run."""
        if run_id not in self.runs:
            raise ValueError(f"Run ID {run_id} not found")
        
        run = self.runs[run_id]
        run.status = PipelineStatus.RUNNING
        run.start_time = datetime.now()
        
        self.logger.info(f"Starting pipeline execution for run {run_id}")
        
        try:
            # Execute tasks in dependency order
            executed_tasks = set()
            task_results = {}
            
            while len(executed_tasks) < len(run.tasks):
                for task_name, task in run.tasks.items():
                    if task_name in executed_tasks:
                        continue
                    
                    # Check if dependencies are satisfied
                    if not all(dep in executed_tasks for dep in task.dependencies):
                        continue
                    
                    # Execute task
                    task_result = self._execute_task(task, task_results)
                    task_results[task_name] = task_result
                    executed_tasks.add(task_name)
                    
                    if task.status == TaskStatus.FAILED:
                        run.status = PipelineStatus.FAILED
                        run.error = f"Task '{task_name}' failed: {task.error}"
                        self.logger.error(f"Pipeline failed due to task '{task_name}': {task.error}")
                        return run
            
            # All tasks completed successfully
            run.status = PipelineStatus.COMPLETED
            run.end_time = datetime.now()
            self.logger.info(f"Pipeline execution completed successfully for run {run_id}")
            
        except Exception as e:
            run.status = PipelineStatus.FAILED
            run.error = str(e)
            run.end_time = datetime.now()
            self.logger.error(f"Pipeline execution failed for run {run_id}: {str(e)}")
        
        return run
    
    def _execute_task(self, task: PipelineTask, task_results: Dict[str, Any]) -> Any:
        """Execute a single task with retry logic."""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        task.attempt_count += 1
        
        self.logger.info(f"Executing task '{task.name}' (attempt {task.attempt_count})")
        
        try:
            # Execute task function
            if task.name == "data_ingestion":
                result = self._execute_data_ingestion(task_results)
            elif task.name == "data_normalization":
                result = self._execute_data_normalization(task_results)
            elif task.name == "resubmission_analysis":
                result = self._execute_resubmission_analysis(task_results)
            elif task.name == "export_results":
                result = self._execute_export_results(task_results)
            else:
                # Generic task execution
                result = task.function(task_results)
            
            task.status = TaskStatus.SUCCESS
            task.end_time = datetime.now()
            task.result = result
            
            self.logger.info(f"Task '{task.name}' completed successfully")
            return result
            
        except Exception as e:
            task.error = str(e)
            task.end_time = datetime.now()
            
            if task.attempt_count < task.retries:
                task.status = TaskStatus.RETRYING
                self.logger.warning(f"Task '{task.name}' failed, retrying in {task.retry_delay}s: {str(e)}")
                time.sleep(task.retry_delay)
                return self._execute_task(task, task_results)
            else:
                task.status = TaskStatus.FAILED
                self.logger.error(f"Task '{task.name}' failed after {task.attempt_count} attempts: {str(e)}")
                raise
    
    def _execute_data_ingestion(self, task_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data ingestion task."""
        self.logger.info("Executing data ingestion task")
        
        # Simulate data ingestion
        time.sleep(random.uniform(1, 3))
        
        return {
            "files_processed": 2,
            "total_records": 9,
            "ingestion_time": datetime.now().isoformat()
        }
    
    def _execute_data_normalization(self, task_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data normalization task."""
        self.logger.info("Executing data normalization task")
        
        # Simulate data normalization
        time.sleep(random.uniform(2, 4))
        
        return {
            "normalized_records": 9,
            "validation_errors": 0,
            "normalization_time": datetime.now().isoformat()
        }
    
    def _execute_resubmission_analysis(self, task_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute resubmission analysis task."""
        self.logger.info("Executing resubmission analysis task")
        
        # Simulate analysis
        time.sleep(random.uniform(1, 2))
        
        return {
            "eligible_claims": 4,
            "analysis_metrics": {
                "denied_claims": 5,
                "retryable_reasons": 4,
                "non_retryable_reasons": 1
            },
            "analysis_time": datetime.now().isoformat()
        }
    
    def _execute_export_results(self, task_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute results export task."""
        self.logger.info("Executing results export task")
        
        # Simulate export
        time.sleep(random.uniform(1, 2))
        
        return {
            "exported_files": ["resubmission_candidates.json", "processing_report.json"],
            "export_time": datetime.now().isoformat()
        }
    
    def get_run_status(self, run_id: str) -> Optional[PipelineRun]:
        """Get the status of a pipeline run."""
        return self.runs.get(run_id)
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all pipeline runs with their status."""
        return [
            {
                "run_id": run.run_id,
                "pipeline_name": run.pipeline_name,
                "status": run.status.value,
                "start_time": run.start_time.isoformat(),
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "error": run.error
            }
            for run in self.runs.values()
        ]
    
    def cancel_run(self, run_id: str) -> bool:
        """Cancel a running pipeline run."""
        if run_id not in self.runs:
            return False
        
        run = self.runs[run_id]
        if run.status == PipelineStatus.RUNNING:
            run.status = PipelineStatus.CANCELLED
            run.end_time = datetime.now()
            self.logger.info(f"Cancelled pipeline run {run_id}")
            return True
        
        return False


def create_emr_pipeline() -> PipelineOrchestrator:
    """Create a pre-configured EMR data processing pipeline."""
    orchestrator = PipelineOrchestrator("EMR Data Processing Pipeline")
    
    # Create a new run
    run_id = orchestrator.create_run({
        "description": "Process EMR data and identify resubmission candidates",
        "data_sources": ["sample_data/emr_alpha.csv", "sample_data/emr_beta.json"],
        "reference_date": "2025-07-30"
    })
    
    # Define pipeline tasks
    tasks = [
        PipelineTask(
            name="data_ingestion",
            function=None,  # Will use built-in implementation
            dependencies=[],
            retries=2,
            timeout=600
        ),
        PipelineTask(
            name="data_normalization",
            function=None,
            dependencies=["data_ingestion"],
            retries=2,
            timeout=900
        ),
        PipelineTask(
            name="resubmission_analysis",
            function=None,
            dependencies=["data_normalization"],
            retries=2,
            timeout=600
        ),
        PipelineTask(
            name="export_results",
            function=None,
            dependencies=["resubmission_analysis"],
            retries=2,
            timeout=300
        )
    ]
    
    # Add tasks to the pipeline
    for task in tasks:
        orchestrator.add_task(run_id, task)
    
    return orchestrator, run_id


def main():
    """Demonstrate the pipeline orchestration."""
    print("🚀 Creating EMR Data Processing Pipeline...")
    
    # Create pipeline
    orchestrator, run_id = create_emr_pipeline()
    
    print(f"📋 Pipeline created with run ID: {run_id}")
    print("🔧 Starting pipeline execution...")
    
    # Execute pipeline
    run = orchestrator.execute_pipeline(run_id)
    
    # Display results
    print(f"\n📊 Pipeline Execution Results:")
    print(f"   Run ID: {run.run_id}")
    print(f"   Status: {run.status.value}")
    print(f"   Start Time: {run.start_time}")
    print(f"   End Time: {run.end_time}")
    
    if run.error:
        print(f"   Error: {run.error}")
    
    # Display task results
    print(f"\n📋 Task Results:")
    for task_name, task in run.tasks.items():
        print(f"   {task_name}: {task.status.value}")
        if task.result:
            print(f"     Result: {task.result}")
    
    # List all runs
    print(f"\n📈 All Pipeline Runs:")
    runs = orchestrator.list_runs()
    for run_info in runs:
        print(f"   {run_info['run_id']}: {run_info['status']}")


if __name__ == "__main__":
    main()
