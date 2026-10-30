"""The contract error shape {code, message} on every failure, with the HTTP status of its kind."""
from __future__ import annotations

from services.cue import api

STATUS = {  # contract code -> HTTP status; a code not listed here is an internal error (500)
    "input_invalid": 400, "clip_unreadable": 400, "clip_too_long": 400, "no_hand": 400, "track_lost": 400,
    "not_found": 404, "body_point_invalid": 400,
}


class ApiError(Exception):
    """A request that cannot be served: `code` is a key of contract/schemas/error_codes.json."""

    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


def invalid(detail: str, status: int = 400) -> ApiError:
    """input_invalid with the contract's message; status 400 (bad input), 405, 409 (busy) or 413 (too large)."""
    return ApiError("input_invalid", api.message("input_invalid", detail=detail), status)


def not_found(detail: str) -> ApiError:
    """404 not_found (v2.3): an unknown clip, stand-in, job, file or route."""
    return ApiError("not_found", api.message("not_found", detail=detail), 404)


def from_cuefx(e: api.CueFXError) -> ApiError:
    return ApiError(e.code, e.message, STATUS.get(e.code, 500))


def internal() -> ApiError:
    """500 internal (v2.3): an unexpected failure, with the contract's safe message (details in the server log)."""
    return ApiError("internal", api.message("internal"), 500)
