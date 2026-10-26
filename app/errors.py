"""Exceptions that map directly onto an HTTP status.

Every module raises these instead of ``fastapi.HTTPException`` so that the
voice store, the request planner and the worker pool stay importable (and
testable) without FastAPI; ``app.main`` turns them into ``{"detail": ...}``.
"""

from __future__ import annotations


class ApiError(Exception):
    status_code = 500

    def __init__(self, detail: str, *, headers: dict[str, str] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.headers = headers


class BadRequest(ApiError):
    status_code = 400


class NotFound(ApiError):
    status_code = 404


class MethodNotAllowed(ApiError):
    status_code = 405


class Conflict(ApiError):
    status_code = 409


class PayloadTooLarge(ApiError):
    status_code = 413


class QueueFull(ApiError):
    status_code = 429

    def __init__(self, detail: str = "queue full, retry later") -> None:
        super().__init__(detail, headers={"Retry-After": "2"})


class SynthesisError(ApiError):
    status_code = 500


class NotReady(ApiError):
    status_code = 503


class RequestTimeout(ApiError):
    status_code = 504
