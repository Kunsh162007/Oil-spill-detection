"""Background jobs for the pipeline API.

Fetching a scene and analysing it take tens of seconds - far longer than an
HTTP request should hold open, and a PaaS router will time out well before
the work finishes. So the pipeline endpoints accept a request, return a job
id immediately, and the client polls.

Deliberately in-process rather than Celery/Redis. A queue would add a broker,
a worker dyno and a whole failure surface for a service that analyses a scene
every few minutes. When that stops being true, JobStore is the seam to
replace - nothing else needs to change.
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

log = logging.getLogger(__name__)

JobState = Literal["queued", "running", "done", "failed"]


@dataclass
class Job:
    job_id: str
    kind: str
    state: JobState = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now(timezone.utc)
        return round((end - self.started_at).total_seconds(), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": self.state,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_s": self.duration_s,
            "params": self.params,
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    """Thread-safe job registry with a bounded history."""

    def __init__(self, max_jobs: int = 200) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def create(self, kind: str, params: dict[str, Any] | None = None) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], kind=kind, params=params or {})
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            # Bound the history so a long-running service does not leak.
            while len(self._order) > self._max_jobs:
                self._jobs.pop(self._order.pop(0), None)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            return [self._jobs[j] for j in reversed(self._order[-limit:]) if j in self._jobs]

    def run(self, job: Job, fn: Callable[[Job], dict[str, Any]]) -> None:
        """Execute fn on a worker thread, recording state transitions."""

        def target() -> None:
            job.state = "running"
            job.started_at = datetime.now(timezone.utc)
            try:
                job.result = fn(job)
                job.state = "done"
            except Exception as exc:
                # The traceback goes to logs; the client gets one clear line.
                log.error("Job %s (%s) failed: %s", job.job_id, job.kind, exc)
                log.debug("%s", traceback.format_exc())
                job.error = f"{type(exc).__name__}: {exc}"
                job.state = "failed"
            finally:
                job.finished_at = datetime.now(timezone.utc)

        threading.Thread(target=target, name=f"job-{job.job_id}", daemon=True).start()
