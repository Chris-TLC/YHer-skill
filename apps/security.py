"""Bounded same-origin HTTP guards and streaming image upload validation."""

from __future__ import annotations

import math
import os
import re
import secrets
import tempfile
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from starlette.responses import JSONResponse


_USER_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_IMAGE_MIME_BY_EXTENSION = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class RequestTooLarge(RuntimeError):
    pass


class UploadSecurityError(ValueError):
    def __init__(self, code: str, *, status_code: int = 400):
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class SlidingWindowRateLimiter:
    """Small in-process limiter intended for the single-worker local Demo."""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float,
        max_keys: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests < 1 or window_seconds <= 0 or max_keys < 1:
            raise ValueError("rate limit values must be positive")
        self.max_requests = int(max_requests)
        self.window_seconds = float(window_seconds)
        self.max_keys = int(max_keys)
        self.clock = clock
        self._entries: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def key_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def allow(self, key: str) -> tuple[bool, int]:
        now = float(self.clock())
        cutoff = now - self.window_seconds
        with self._lock:
            timestamps = self._entries.get(key)
            if timestamps is None:
                while len(self._entries) >= self.max_keys:
                    self._entries.popitem(last=False)
                timestamps = deque()
                self._entries[key] = timestamps
            else:
                self._entries.move_to_end(key)
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.max_requests:
                retry_after = max(1, math.ceil(self.window_seconds - (now - timestamps[0])))
                return False, retry_after
            timestamps.append(now)
            return True, 0


class RequestGuardMiddleware:
    """Reject hostile origins, oversized requests and simple request floods."""

    def __init__(
        self,
        app,
        *,
        max_body_bytes: int = 512 * 1024,
        body_limit_overrides: dict[str, int] | None = None,
        max_requests: int = 120,
        window_seconds: float = 60,
        max_rate_keys: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.max_body_bytes = int(max_body_bytes)
        self.body_limit_overrides = tuple(
            sorted(
                (
                    (str(prefix), int(limit))
                    for prefix, limit in (body_limit_overrides or {}).items()
                    if str(prefix) and int(limit) > 0
                ),
                key=lambda entry: len(entry[0]),
                reverse=True,
            )
        )
        self.limiter = SlidingWindowRateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds,
            max_keys=max_rate_keys,
            clock=clock,
        )

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        if not _same_origin(headers.get("origin"), headers.get("host"), scope.get("scheme")):
            await _json_error(403, "cross_origin_forbidden")(scope, receive, send)
            return
        body_limit = self._body_limit(str(scope.get("path") or ""))
        content_length = _content_length(headers.get("content-length"))
        if content_length is not None and content_length > body_limit:
            await _json_error(413, "request_too_large")(scope, receive, send)
            return
        client = scope.get("client") or ("unknown", 0)
        key = f"{client[0]}:{scope.get('method', '')}:{scope.get('path', '')}"
        allowed, retry_after = self.limiter.allow(key)
        if not allowed:
            response = _json_error(429, "rate_limited")
            response.headers["Retry-After"] = str(retry_after)
            await response(scope, receive, send)
            return

        total = 0

        async def guarded_receive():
            nonlocal total
            message = await receive()
            if message.get("type") == "http.request":
                total += len(message.get("body") or b"")
                if total > body_limit:
                    raise RequestTooLarge
            return message

        try:
            await self.app(scope, guarded_receive, send)
        except RequestTooLarge:
            await _json_error(413, "request_too_large")(scope, receive, send)

    def _body_limit(self, path: str) -> int:
        for prefix, limit in self.body_limit_overrides:
            if path.startswith(prefix):
                return limit
        return self.max_body_bytes


def _same_origin(origin: str | None, host: str | None, scheme: str | None) -> bool:
    if origin is None:
        return True
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.scheme == str(scheme or "http")
        and bool(host)
        and parsed.netloc.lower() == str(host).lower()
    )


def _content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return max(0, parsed)


def _json_error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status_code)


async def save_image_upload(
    upload,
    *,
    user_id: str,
    upload_dir: Path,
    max_bytes: int = 8 * 1024 * 1024,
    id_factory: Callable[[], str] | None = None,
    chunk_size: int = 64 * 1024,
) -> dict[str, Any]:
    """Stream one validated image to an atomic destination inside upload_dir."""
    if not _USER_ID.fullmatch(str(user_id)):
        raise UploadSecurityError("invalid_user_id")
    extension = Path(str(upload.filename or "")).suffix.lower()
    expected_mime = _IMAGE_MIME_BY_EXTENSION.get(extension)
    if expected_mime is None:
        raise UploadSecurityError("unsupported_extension")
    content_type = str(getattr(upload, "content_type", "") or "").lower()
    if content_type != expected_mime:
        raise UploadSecurityError("mime_extension_mismatch")
    if max_bytes < 1 or chunk_size < 1:
        raise ValueError("upload limits must be positive")

    token = (id_factory or (lambda: secrets.token_hex(8)))()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(token)):
        raise UploadSecurityError("invalid_upload_id")
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{user_id}_{token}{extension}"
    destination = upload_dir / filename
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".part", dir=upload_dir
    )
    size = 0
    signature = bytearray()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise UploadSecurityError("upload_too_large", status_code=413)
                if len(signature) < 16:
                    signature.extend(chunk[: 16 - len(signature)])
                handle.write(chunk)
            if not _valid_image_signature(expected_mime, bytes(signature)):
                raise UploadSecurityError("invalid_image_signature")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return {"upload_id": filename, "size_bytes": size}


def _valid_image_signature(mime: str, prefix: bytes) -> bool:
    if mime == "image/png":
        return prefix.startswith(b"\x89PNG\r\n\x1a\n")
    if mime == "image/jpeg":
        return prefix.startswith(b"\xff\xd8\xff")
    if mime == "image/webp":
        return len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP"
    return False
