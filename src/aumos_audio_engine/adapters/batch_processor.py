"""Distributed audio batch processing adapter.

Implements AudioBatchProcessorProtocol. Manages a job queue for parallel
audio synthesis/transcription, with worker-pool management, progress tracking,
per-job error isolation, retry logic, resource accounting, and result aggregation.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from aumos_common.observability import get_logger

from aumos_audio_engine.settings import Settings

logger = get_logger(__name__)


class BatchJobStatus(str, Enum):
    """Status of an individual batch job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """Represents a single item in the batch processing queue."""

    job_id: str
    payload: dict
    handler_key: str
    status: BatchJobStatus = BatchJobStatus.QUEUED
    attempt: int = 0
    max_attempts: int = 3
    result: dict | None = None
    error: str | None = None
    queued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    completed_at: float | None = None
    resource_hint: str = "cpu"  # 'cpu' or 'gpu'


@dataclass
class BatchResult:
    """Aggregated result for a completed batch."""

    batch_id: str
    total: int
    completed: int
    failed: int
    cancelled: int
    results: list[dict]
    errors: list[dict]
    duration_seconds: float
    throughput_jobs_per_second: float


class AudioBatchProcessor:
    """Distributed audio batch processor with worker pool and progress tracking.

    Manages concurrent audio processing jobs across a configurable number of
    async workers. Each worker processes jobs from a shared async queue. Failed
    jobs are retried up to max_attempts times with exponential back-off before
    being moved to the error list.

    Resource management:
        - CPU workers process standard inference tasks.
        - GPU slot (optional semaphore) serialises GPU-bound operations when
          gpu_concurrency is set to 1 (e.g. for a single GPU node).

    Thread safety:
        - All shared state (job registry, result lists) is guarded by asyncio locks.
        - Handler callables must be coroutine functions (async).
    """

    def __init__(
        self,
        settings: Settings,
        max_workers: int = 4,
        gpu_concurrency: int = 1,
        default_max_attempts: int = 3,
    ) -> None:
        """Initialize the batch processor.

        Args:
            settings: Audio engine settings (used for resource limits, temp_dir).
            max_workers: Maximum number of concurrent async workers.
            gpu_concurrency: Maximum simultaneous GPU-bound jobs (1 for single GPU).
            default_max_attempts: Default number of retry attempts per failed job.
        """
        self._settings = settings
        self._max_workers = max_workers
        self._gpu_concurrency = gpu_concurrency
        self._default_max_attempts = default_max_attempts

        self._queue: asyncio.Queue[BatchJob] = asyncio.Queue()
        self._job_registry: dict[str, BatchJob] = {}
        self._handlers: dict[str, Callable[..., Coroutine[Any, Any, dict]]] = {}
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._registry_lock = asyncio.Lock()
        self._gpu_semaphore = asyncio.Semaphore(gpu_concurrency)
        self._running = False
        self._total_processed = 0
        self._total_failed = 0

    def register_handler(
        self,
        handler_key: str,
        handler: Callable[..., Coroutine[Any, Any, dict]],
    ) -> None:
        """Register a named async handler for a job type.

        The handler receives the BatchJob payload dict as its single argument
        and must return a result dict on success.

        Args:
            handler_key: Unique string key identifying this handler type.
            handler: Async callable accepting a dict payload and returning a dict.

        Raises:
            ValueError: If handler_key is already registered.
        """
        if handler_key in self._handlers:
            raise ValueError(
                f"Handler '{handler_key}' is already registered. "
                "Use replace_handler() to overwrite."
            )
        self._handlers[handler_key] = handler
        logger.info("Batch handler registered", handler_key=handler_key)

    def replace_handler(
        self,
        handler_key: str,
        handler: Callable[..., Coroutine[Any, Any, dict]],
    ) -> None:
        """Replace an existing handler registration.

        Args:
            handler_key: Handler key to replace.
            handler: New async callable.
        """
        self._handlers[handler_key] = handler
        logger.info("Batch handler replaced", handler_key=handler_key)

    async def start(self) -> None:
        """Start the worker pool.

        Spawns max_workers async worker tasks that continuously consume from the job queue.
        Idempotent — calling start() on an already-running processor has no effect.
        """
        if self._running:
            return

        self._running = True
        self._worker_tasks = [
            asyncio.create_task(self._worker(worker_index=i), name=f"batch-worker-{i}")
            for i in range(self._max_workers)
        ]
        logger.info(
            "Batch processor started",
            max_workers=self._max_workers,
            gpu_concurrency=self._gpu_concurrency,
        )

    async def stop(self, timeout_seconds: float = 30.0) -> None:
        """Gracefully stop the worker pool.

        Waits up to timeout_seconds for in-flight jobs to complete, then
        cancels remaining workers.

        Args:
            timeout_seconds: Maximum time to wait for graceful shutdown.
        """
        if not self._running:
            return

        self._running = False

        # Signal workers to stop via sentinel jobs
        for _ in range(self._max_workers):
            await self._queue.put(
                BatchJob(
                    job_id="__stop__",
                    payload={},
                    handler_key="__stop__",
                    status=BatchJobStatus.CANCELLED,
                )
            )

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._worker_tasks, return_exceptions=True),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Batch processor shutdown timed out — cancelling workers",
                timeout_seconds=timeout_seconds,
            )
            for task in self._worker_tasks:
                task.cancel()
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        self._worker_tasks.clear()
        logger.info("Batch processor stopped")

    async def submit(
        self,
        payload: dict,
        handler_key: str,
        job_id: str | None = None,
        max_attempts: int | None = None,
        resource_hint: str = "cpu",
    ) -> str:
        """Submit a single job for asynchronous processing.

        Args:
            payload: Job payload dict passed to the handler.
            handler_key: Registered handler key for this job type.
            job_id: Optional explicit job ID. Auto-generated if not provided.
            max_attempts: Override the default retry limit for this job.
            resource_hint: 'cpu' or 'gpu' — used for semaphore selection.

        Returns:
            Job ID string for status polling.

        Raises:
            ValueError: If handler_key is not registered.
            RuntimeError: If the processor is not started.
        """
        if not self._running:
            raise RuntimeError("BatchProcessor is not running — call start() first")

        if handler_key not in self._handlers and handler_key != "__stop__":
            raise ValueError(
                f"No handler registered for key '{handler_key}'. "
                f"Available: {list(self._handlers.keys())}"
            )

        effective_job_id = job_id or str(uuid.uuid4())
        job = BatchJob(
            job_id=effective_job_id,
            payload=payload,
            handler_key=handler_key,
            max_attempts=max_attempts or self._default_max_attempts,
            resource_hint=resource_hint,
        )

        async with self._registry_lock:
            self._job_registry[effective_job_id] = job

        await self._queue.put(job)

        logger.info(
            "Batch job submitted",
            job_id=effective_job_id,
            handler_key=handler_key,
            resource_hint=resource_hint,
        )
        return effective_job_id

    async def submit_batch(
        self,
        items: list[dict],
        handler_key: str,
        max_attempts: int | None = None,
        resource_hint: str = "cpu",
    ) -> list[str]:
        """Submit multiple jobs as a batch.

        Args:
            items: List of payload dicts, one per job.
            handler_key: Registered handler key applied to all jobs.
            max_attempts: Retry limit override for all jobs.
            resource_hint: Resource hint applied to all jobs.

        Returns:
            List of job ID strings in the same order as items.
        """
        job_ids = []
        for payload in items:
            job_id = await self.submit(
                payload=payload,
                handler_key=handler_key,
                max_attempts=max_attempts,
                resource_hint=resource_hint,
            )
            job_ids.append(job_id)

        logger.info(
            "Batch submitted",
            total_jobs=len(items),
            handler_key=handler_key,
        )
        return job_ids

    async def get_job_status(self, job_id: str) -> dict:
        """Get the current status and result for a job.

        Args:
            job_id: Job ID returned by submit().

        Returns:
            Dict with 'job_id', 'status', 'attempt', 'result', 'error',
            'queued_at', 'started_at', 'completed_at', 'duration_seconds'.

        Raises:
            KeyError: If job_id is not found in the registry.
        """
        async with self._registry_lock:
            if job_id not in self._job_registry:
                raise KeyError(f"Job '{job_id}' not found in batch registry")
            job = self._job_registry[job_id]

        duration: float | None = None
        if job.started_at is not None and job.completed_at is not None:
            duration = round(job.completed_at - job.started_at, 3)

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "handler_key": job.handler_key,
            "attempt": job.attempt,
            "max_attempts": job.max_attempts,
            "result": job.result,
            "error": job.error,
            "queued_at": job.queued_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "duration_seconds": duration,
            "resource_hint": job.resource_hint,
        }

    async def wait_for_jobs(
        self,
        job_ids: list[str],
        poll_interval_seconds: float = 0.25,
        timeout_seconds: float | None = None,
    ) -> BatchResult:
        """Wait for a list of jobs to reach terminal status, then return aggregated results.

        Args:
            job_ids: List of job IDs to wait for.
            poll_interval_seconds: Polling interval between status checks.
            timeout_seconds: Maximum time to wait. None = wait indefinitely.

        Returns:
            BatchResult with complete aggregated statistics and per-job results.

        Raises:
            asyncio.TimeoutError: If timeout_seconds is exceeded before all jobs complete.
        """
        batch_id = str(uuid.uuid4())
        start_time = time.monotonic()
        terminal_statuses = {
            BatchJobStatus.COMPLETED,
            BatchJobStatus.FAILED,
            BatchJobStatus.CANCELLED,
        }

        while True:
            statuses = await asyncio.gather(
                *[self.get_job_status(jid) for jid in job_ids]
            )
            all_done = all(
                BatchJobStatus(s["status"]) in terminal_statuses for s in statuses
            )

            if all_done:
                break

            elapsed = time.monotonic() - start_time
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                raise asyncio.TimeoutError(
                    f"Timed out after {timeout_seconds}s waiting for {len(job_ids)} batch jobs"
                )

            await asyncio.sleep(poll_interval_seconds)

        elapsed_total = time.monotonic() - start_time
        completed_jobs = [
            s for s in statuses if BatchJobStatus(s["status"]) == BatchJobStatus.COMPLETED
        ]
        failed_jobs = [
            s for s in statuses if BatchJobStatus(s["status"]) == BatchJobStatus.FAILED
        ]
        cancelled_jobs = [
            s for s in statuses if BatchJobStatus(s["status"]) == BatchJobStatus.CANCELLED
        ]

        throughput = len(job_ids) / max(elapsed_total, 0.001)

        logger.info(
            "Batch wait complete",
            batch_id=batch_id,
            total=len(job_ids),
            completed=len(completed_jobs),
            failed=len(failed_jobs),
            duration_seconds=round(elapsed_total, 2),
            throughput_jobs_per_second=round(throughput, 2),
        )

        return BatchResult(
            batch_id=batch_id,
            total=len(job_ids),
            completed=len(completed_jobs),
            failed=len(failed_jobs),
            cancelled=len(cancelled_jobs),
            results=[s["result"] for s in completed_jobs if s["result"] is not None],
            errors=[
                {"job_id": s["job_id"], "error": s["error"], "attempt": s["attempt"]}
                for s in failed_jobs
            ],
            duration_seconds=round(elapsed_total, 3),
            throughput_jobs_per_second=round(throughput, 3),
        )

    async def cancel_job(self, job_id: str) -> bool:
        """Attempt to cancel a queued job.

        Jobs already in RUNNING state cannot be cancelled.

        Args:
            job_id: Job ID to cancel.

        Returns:
            True if job was cancelled, False if it was not in a cancellable state.
        """
        async with self._registry_lock:
            job = self._job_registry.get(job_id)
            if job is None:
                return False
            if job.status == BatchJobStatus.QUEUED:
                job.status = BatchJobStatus.CANCELLED
                job.completed_at = time.monotonic()
                logger.info("Batch job cancelled", job_id=job_id)
                return True
        return False

    def get_queue_depth(self) -> int:
        """Return the number of jobs currently waiting in the queue."""
        return self._queue.qsize()

    def get_worker_count(self) -> int:
        """Return the number of active worker tasks."""
        return sum(1 for task in self._worker_tasks if not task.done())

    def get_stats(self) -> dict:
        """Return aggregate runtime statistics.

        Returns:
            Dict with queue depth, worker count, total processed, and total failed.
        """
        return {
            "queue_depth": self.get_queue_depth(),
            "active_workers": self.get_worker_count(),
            "total_processed": self._total_processed,
            "total_failed": self._total_failed,
            "registered_handlers": list(self._handlers.keys()),
            "running": self._running,
        }

    # ── Worker internals ──────────────────────────────────────────────────────

    async def _worker(self, worker_index: int) -> None:
        """Long-running worker coroutine that consumes and processes batch jobs.

        Args:
            worker_index: Worker index for structured logging.
        """
        logger.info("Batch worker started", worker_index=worker_index)

        while self._running:
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Sentinel: processor is shutting down
            if job.handler_key == "__stop__":
                self._queue.task_done()
                break

            # Skip already-cancelled jobs
            async with self._registry_lock:
                registered_job = self._job_registry.get(job.job_id)
                if registered_job and registered_job.status == BatchJobStatus.CANCELLED:
                    self._queue.task_done()
                    continue

            await self._process_job(job, worker_index)
            self._queue.task_done()

        logger.info("Batch worker stopped", worker_index=worker_index)

    async def _process_job(self, job: BatchJob, worker_index: int) -> None:
        """Process a single batch job with retry logic.

        Args:
            job: The BatchJob to execute.
            worker_index: Worker identifier for logging.
        """
        handler = self._handlers.get(job.handler_key)
        if handler is None:
            async with self._registry_lock:
                job.status = BatchJobStatus.FAILED
                job.error = f"No handler registered for key '{job.handler_key}'"
                job.completed_at = time.monotonic()
            self._total_failed += 1
            return

        for attempt in range(1, job.max_attempts + 1):
            job.attempt = attempt

            async with self._registry_lock:
                job.status = BatchJobStatus.RUNNING if attempt == 1 else BatchJobStatus.RETRYING
                if job.started_at is None:
                    job.started_at = time.monotonic()

            logger.info(
                "Batch job starting",
                job_id=job.job_id,
                handler_key=job.handler_key,
                attempt=attempt,
                worker_index=worker_index,
            )

            try:
                # GPU semaphore for GPU-bound jobs
                if job.resource_hint == "gpu":
                    async with self._gpu_semaphore:
                        result = await handler(job.payload)
                else:
                    result = await handler(job.payload)

                async with self._registry_lock:
                    job.status = BatchJobStatus.COMPLETED
                    job.result = result
                    job.error = None
                    job.completed_at = time.monotonic()

                self._total_processed += 1
                duration = (job.completed_at or time.monotonic()) - (job.started_at or time.monotonic())
                logger.info(
                    "Batch job completed",
                    job_id=job.job_id,
                    handler_key=job.handler_key,
                    attempt=attempt,
                    duration_seconds=round(duration, 3),
                )
                return  # Success — exit retry loop

            except Exception as exc:
                error_message = str(exc)
                logger.warning(
                    "Batch job attempt failed",
                    job_id=job.job_id,
                    handler_key=job.handler_key,
                    attempt=attempt,
                    max_attempts=job.max_attempts,
                    error=error_message,
                )

                if attempt < job.max_attempts:
                    # Exponential back-off: 1s, 2s, 4s, ...
                    backoff_seconds = 2 ** (attempt - 1)
                    await asyncio.sleep(backoff_seconds)
                else:
                    # Exhausted all attempts
                    async with self._registry_lock:
                        job.status = BatchJobStatus.FAILED
                        job.error = error_message
                        job.completed_at = time.monotonic()

                    self._total_failed += 1
                    logger.error(
                        "Batch job permanently failed",
                        job_id=job.job_id,
                        handler_key=job.handler_key,
                        total_attempts=job.max_attempts,
                        error=error_message,
                    )

    async def health_check(self) -> bool:
        """Return True if the batch processor is running with at least one active worker."""
        return self._running and self.get_worker_count() > 0
