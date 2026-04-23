from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass(frozen=True)
class OCRPipelineTask:
    image: Image.Image
    task_type: str
    future: Future["OCRTaskResult"]
    enqueued_at: float
    queue_size_at_submit: int


@dataclass(frozen=True)
class OCRTaskResult:
    content: Optional[str]
    status_code: int
    queue_wait_seconds: float
    build_request_seconds: float
    request_seconds: float
    total_seconds: float
    queue_size_at_submit: int


class OCRPipelineDispatcher:
    """Background OCR dispatcher that keeps the backend fed from a bounded queue."""

    def __init__(
        self,
        page_loader,
        ocr_client,
        *,
        max_workers: int,
        queue_size: int,
    ) -> None:
        self._page_loader = page_loader
        self._ocr_client = ocr_client
        self._queue: queue.Queue[OCRPipelineTask | None] = queue.Queue(maxsize=max(1, queue_size))
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        for index in range(max(1, max_workers)):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"ocr-pipeline-{index + 1}",
                daemon=True,
            )
            worker.start()
            self._threads.append(worker)

    def submit(self, image: Image.Image, *, task_type: str) -> Future[OCRTaskResult]:
        future: Future[OCRTaskResult] = Future()
        task = OCRPipelineTask(
            image=image,
            task_type=task_type,
            future=future,
            enqueued_at=time.perf_counter(),
            queue_size_at_submit=self._queue.qsize(),
        )
        self._queue.put(task)
        return future

    def close(self) -> None:
        self._stop_event.set()
        for _ in self._threads:
            self._queue.put(None)
        for worker in self._threads:
            worker.join(timeout=5)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                break

            try:
                request_started_at = time.perf_counter()
                queue_wait_seconds = request_started_at - task.enqueued_at
                build_started_at = request_started_at
                request = self._page_loader.build_request_from_image(task.image, task_type=task.task_type)
                build_request_seconds = time.perf_counter() - build_started_at
                ocr_started_at = time.perf_counter()
                response, status_code = self._ocr_client.process(request)
                request_seconds = time.perf_counter() - ocr_started_at
                total_seconds = time.perf_counter() - task.enqueued_at
                if status_code == 200:
                    content = (
                        response.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    task.future.set_result(
                        OCRTaskResult(
                            content=str(content).strip(),
                            status_code=int(status_code),
                            queue_wait_seconds=queue_wait_seconds,
                            build_request_seconds=build_request_seconds,
                            request_seconds=request_seconds,
                            total_seconds=total_seconds,
                            queue_size_at_submit=task.queue_size_at_submit,
                        )
                    )
                else:
                    task.future.set_result(
                        OCRTaskResult(
                            content=None,
                            status_code=int(status_code),
                            queue_wait_seconds=queue_wait_seconds,
                            build_request_seconds=build_request_seconds,
                            request_seconds=request_seconds,
                            total_seconds=total_seconds,
                            queue_size_at_submit=task.queue_size_at_submit,
                        )
                    )
            except Exception as exc:
                task.future.set_exception(exc)
            finally:
                self._queue.task_done()
