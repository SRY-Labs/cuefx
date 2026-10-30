"""The render queue: heavy work (stand-in, element video, manual export Make, manual import) runs here, one at a
time on one worker thread, the rest waiting in order. Request threads only submit and (when asked) wait, so the
light routes (health, validate, preview, frames, ...) never wait on a render."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from services.errors import ApiError, internal

log = logging.getLogger("cuefx.server")


@dataclass
class Task:
    """One piece of queued work. `status`: queued | running | done | failed."""
    id: str
    future: Future
    submitted: float = field(default_factory=time.perf_counter)
    started: float | None = None

    @property
    def status(self) -> str:
        if not self.future.done():
            return "running" if self.started is not None else "queued"
        return "failed" if self.future.exception() is not None else "done"

    @property
    def error(self) -> ApiError | None:
        """The contract error of a failed task (an unexpected exception becomes `internal`)."""
        if not self.future.done() or self.future.exception() is None:
            return None
        e = self.future.exception()
        return e if isinstance(e, ApiError) else internal()

    def wait(self) -> Any:
        """Block until the task ends; its result, or its contract error raised."""
        try:
            return self.future.result()
        except ApiError:
            raise
        except Exception as e:  # logged by the worker; the client gets the safe internal error
            raise internal() from e


class RenderQueue:
    """A single worker thread. `submit` with the id of a task still queued or running returns that task, so a
    repeated request joins the render already on its way instead of queueing a second one."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cuefx-render")
        self._tasks: dict[str, Task] = {}
        self._guard = threading.Lock()

    def submit(self, task_id: str, fn: Callable[[], Any]) -> Task:
        with self._guard:
            old = self._tasks.get(task_id)
            if old is not None and not old.future.done():
                return old
            task = Task(task_id, Future())

            def run() -> Any:
                task.started = time.perf_counter()
                try:
                    return fn()
                except ApiError:
                    raise
                except Exception as e:
                    log.error("render task %s failed", task_id, exc_info=e)
                    raise

            task.future = self._pool.submit(run)
            self._tasks[task_id] = task
            return task

    def get(self, task_id: str) -> Task | None:
        with self._guard:
            return self._tasks.get(task_id)

    def busy(self, task_id: str) -> bool:
        """True while the task is queued or running."""
        t = self.get(task_id)
        return t is not None and not t.future.done()

    def cancel(self, task_id: str) -> bool:
        """Cancel a task that has not started; running work must stop cooperatively."""
        task = self.get(task_id)
        return task is not None and task.started is None and task.future.cancel()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
