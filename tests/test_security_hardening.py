"""HTTP and upload security contracts for the local Demo applications."""

from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def _guarded_client(clock: Clock, *, max_body_bytes: int = 64, max_requests: int = 20):
    from apps.security import RequestGuardMiddleware

    app = FastAPI()
    app.add_middleware(
        RequestGuardMiddleware,
        max_body_bytes=max_body_bytes,
        max_requests=max_requests,
        window_seconds=60,
        max_rate_keys=4,
        clock=clock,
    )

    @app.post("/echo")
    async def echo(request: Request):
        return {"size": len(await request.body())}

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return TestClient(app)


def test_request_guard_rejects_cross_origin_and_oversized_body() -> None:
    clock = Clock()
    client = _guarded_client(clock, max_body_bytes=16)

    allowed = client.post(
        "/echo",
        content=b"1234",
        headers={"Origin": "http://testserver", "Content-Type": "application/octet-stream"},
    )
    hostile = client.post(
        "/echo",
        content=b"1234",
        headers={"Origin": "https://hostile.example", "Content-Type": "application/octet-stream"},
    )
    oversized = client.post(
        "/echo",
        content=b"x" * 17,
        headers={"Content-Type": "application/octet-stream"},
    )

    assert allowed.status_code == 200
    assert hostile.status_code == 403
    assert hostile.json() == {"detail": "cross_origin_forbidden"}
    assert oversized.status_code == 413
    assert oversized.json() == {"detail": "request_too_large"}


def test_rate_limit_resets_and_limiter_state_is_bounded() -> None:
    from apps.security import SlidingWindowRateLimiter

    clock = Clock()
    client = _guarded_client(clock, max_requests=2)
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    limited = client.get("/ping")
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"

    clock.value += 61
    assert client.get("/ping").status_code == 200

    limiter = SlidingWindowRateLimiter(
        max_requests=1,
        window_seconds=60,
        max_keys=2,
        clock=clock,
    )
    assert limiter.allow("one")[0] is True
    assert limiter.allow("two")[0] is True
    assert limiter.allow("three")[0] is True
    assert limiter.key_count == 2


def test_request_guard_supports_a_larger_streaming_upload_limit() -> None:
    from apps.security import RequestGuardMiddleware

    clock = Clock()
    app = FastAPI()
    app.add_middleware(
        RequestGuardMiddleware,
        max_body_bytes=16,
        body_limit_overrides={"/upload/": 64},
        max_requests=20,
        window_seconds=60,
        max_rate_keys=4,
        clock=clock,
    )

    @app.post("/regular")
    async def regular(request: Request):
        return {"size": len(await request.body())}

    @app.post("/upload/file")
    async def upload(request: Request):
        return {"size": len(await request.body())}

    client = TestClient(app)
    assert client.post("/regular", content=b"x" * 17).status_code == 413
    assert client.post("/upload/file", content=b"x" * 32).status_code == 200


def _upload(filename: str, content_type: str, content: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename, headers={"content-type": content_type})


def test_streaming_upload_accepts_real_image_and_uses_safe_filename(tmp_path: Path) -> None:
    from apps.security import save_image_upload

    payload = b"\x89PNG\r\n\x1a\n" + b"image-data"
    result = asyncio.run(
        save_image_upload(
            _upload("homework.png", "image/png", payload),
            user_id="student_01",
            upload_dir=tmp_path,
            max_bytes=1024,
            id_factory=lambda: "fixedid",
            chunk_size=5,
        )
    )

    assert result == {"upload_id": "student_01_fixedid.png", "size_bytes": len(payload)}
    assert (tmp_path / result["upload_id"]).read_bytes() == payload
    assert not list(tmp_path.glob(".*.part"))


@pytest.mark.parametrize(
    ("user_id", "filename", "content_type", "payload", "error_code"),
    (
        ("../escape", "homework.png", "image/png", b"\x89PNG\r\n\x1a\n", "invalid_user_id"),
        ("student", "homework.exe", "image/png", b"\x89PNG\r\n\x1a\n", "unsupported_extension"),
        ("student", "homework.jpg", "image/png", b"\x89PNG\r\n\x1a\n", "mime_extension_mismatch"),
        ("student", "homework.png", "image/png", b"not-an-image", "invalid_image_signature"),
    ),
)
def test_streaming_upload_rejects_traversal_and_spoofing(
    tmp_path: Path,
    user_id: str,
    filename: str,
    content_type: str,
    payload: bytes,
    error_code: str,
) -> None:
    from apps.security import UploadSecurityError, save_image_upload

    with pytest.raises(UploadSecurityError) as caught:
        asyncio.run(
            save_image_upload(
                _upload(filename, content_type, payload),
                user_id=user_id,
                upload_dir=tmp_path,
                max_bytes=1024,
            )
        )

    assert caught.value.code == error_code
    assert not list(tmp_path.iterdir())


def test_streaming_upload_removes_partial_file_when_limit_is_exceeded(tmp_path: Path) -> None:
    from apps.security import UploadSecurityError, save_image_upload

    payload = b"\xff\xd8\xff" + b"x" * 80
    with pytest.raises(UploadSecurityError) as caught:
        asyncio.run(
            save_image_upload(
                _upload("large.jpg", "image/jpeg", payload),
                user_id="student",
                upload_dir=tmp_path,
                max_bytes=32,
                chunk_size=7,
            )
        )

    assert caught.value.code == "upload_too_large"
    assert not list(tmp_path.iterdir())


def test_legacy_api_uses_same_origin_guard_and_streaming_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps import api_server

    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(api_server, "UPLOAD_MAX_BYTES", 64)
    client = TestClient(api_server.app)

    hostile = client.get("/health", headers={"Origin": "https://hostile.example"})
    traversal = client.post(
        "/upload/homework",
        data={"user_id": "../escape"},
        files={"file": ("work.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    valid = client.post(
        "/upload/homework",
        data={"user_id": "student"},
        files={"file": ("work.png", b"\x89PNG\r\n\x1a\ncontent", "image/png")},
    )

    assert hostile.status_code == 403
    assert traversal.status_code == 400
    assert valid.status_code == 200
    assert valid.json()["size_bytes"] == 15
    assert (tmp_path / valid.json()["upload_id"]).is_file()


def test_legacy_upload_enforces_streaming_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps import api_server

    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(api_server, "UPLOAD_MAX_BYTES", 16)
    client = TestClient(api_server.app)

    response = client.post(
        "/upload/homework",
        data={"user_id": "student"},
        files={"file": ("work.jpg", b"\xff\xd8\xff" + b"x" * 20, "image/jpeg")},
    )

    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("path", "payload"),
    (
        ("/api/render_report", {"message": "layout"}),
        ("/api/v4/study/event", {"event": "session_start"}),
        ("/api/v4/study/report_bad", {"item_id": "fixture"}),
    ),
)
def test_anonymous_append_routes_are_size_and_frequency_bounded(
    path: str, payload: dict
) -> None:
    from apps.security import RequestGuardMiddleware

    app = FastAPI()
    app.add_middleware(
        RequestGuardMiddleware,
        max_body_bytes=128,
        max_requests=2,
        window_seconds=60,
        max_rate_keys=16,
        clock=Clock(),
    )

    async def append_fixture(request: Request):
        return {"ok": True, "payload": await request.json()}

    app.add_api_route(path, append_fixture, methods=["POST"])
    client = TestClient(app)

    assert client.post(path, json=payload).status_code == 200
    assert client.post(path, json=payload).status_code == 200
    assert client.post(path, json=payload).status_code == 429

    oversized_app = FastAPI()
    oversized_app.add_middleware(
        RequestGuardMiddleware,
        max_body_bytes=64,
        max_requests=20,
        window_seconds=60,
        max_rate_keys=16,
        clock=Clock(),
    )

    async def oversized_fixture(request: Request):
        return {"ok": True, "payload": await request.json()}

    oversized_app.add_api_route(path, oversized_fixture, methods=["POST"])
    oversized = TestClient(oversized_app).post(path, json={**payload, "padding": "x" * 128})
    assert oversized.status_code == 413
