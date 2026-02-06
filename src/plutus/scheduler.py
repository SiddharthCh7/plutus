"""Task scheduler - Cron-based agent execution."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from plutus.config import get_schedule_config
from plutus.logging import get_logger

logger = get_logger(__name__)


class TaskScheduler:
    """Cron-based task scheduler for Plutus.
    
    Runs tasks at configured intervals:
    - portfolio_monitor: Every hour during market hours
    - opportunity_discovery: Every 4 hours
    - news_digest: Every 30 minutes
    """
    
    def __init__(self) -> None:
        self._config = get_schedule_config()
        self._scheduler = AsyncIOScheduler(
            timezone=self._config.timezone,
        )
        self._task_handlers: dict[str, Callable] = {}
    
    def register_handler(
        self,
        task_name: str,
        handler: Callable,
    ) -> None:
        """Register a handler for a task type."""
        self._task_handlers[task_name] = handler
        logger.info("Registered task handler", task=task_name)
    
    def schedule_tasks(self) -> None:
        """Schedule all enabled tasks from config."""
        for task_name, task_config in self._config.tasks.items():
            if not task_config.get("enabled", True):
                logger.info("Task disabled, skipping", task=task_name)
                continue
            
            if task_name not in self._task_handlers:
                logger.warning("No handler for task", task=task_name)
                continue
            
            # Parse schedule
            start_time = task_config.get("start_time", "08:00")
            frequency_mins = task_config.get("frequency_mins", 60)
            end_time = task_config.get("end_time")
            
            hour, minute = map(int, start_time.split(":"))
            
            # Create cron trigger
            trigger = CronTrigger(
                hour=f"{hour}-23" if not end_time else f"{hour}-{int(end_time.split(':')[0])}",
                minute=f"*/{frequency_mins}" if frequency_mins < 60 else str(minute),
            )
            
            # Add job
            self._scheduler.add_job(
                self._task_handlers[task_name],
                trigger=trigger,
                id=task_name,
                name=task_config.get("description", task_name),
                kwargs={"task_name": task_name, "agents": task_config.get("agents", [])},
            )
            
            logger.info(
                "Scheduled task",
                task=task_name,
                start_time=start_time,
                frequency_mins=frequency_mins,
            )
    
    def start(self) -> None:
        """Start the scheduler."""
        self._scheduler.start()
        logger.info("Scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self._scheduler.shutdown()
        logger.info("Scheduler stopped")
    
    def run_now(self, task_name: str, **kwargs) -> Any:
        """Manually trigger a task immediately."""
        if task_name not in self._task_handlers:
            raise ValueError(f"Unknown task: {task_name}")
        
        task_config = self._config.get_task(task_name) or {}
        
        logger.info("Running task manually", task=task_name)
        
        return self._task_handlers[task_name](
            task_name=task_name,
            agents=task_config.get("agents", []),
            **kwargs,
        )
    
    def list_scheduled(self) -> list[dict[str, Any]]:
        """List all scheduled tasks."""
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            })
        return jobs


# Singleton
_scheduler: TaskScheduler | None = None


def get_scheduler() -> TaskScheduler:
    """Get the scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler
