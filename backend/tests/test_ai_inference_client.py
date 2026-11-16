from __future__ import annotations

import io
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.infrastructure.ai_clients.video_inference_client import (
    AIInferenceClient,
    AIInferenceHTTPError,
    AIInferenceResponseError,
    AIInferenceUnavailableError,
)

BASE_URL = "http://ai-engine:8001"
TIMEOUT_SECONDS = 3.0
VIDEO_BYTES = b"sample video payload"


def valid_payload() -> dict[str, Any]:
    return {
        "predictionStatus": "UNCONFIRMED",
        "modelVersion": "bilstm-v1",
        "scoreType": "uncalibrated_softmax",
        "frameCount": 96,
        "candidates": [
            {"label": f"candidate-{rank}", "score": (rank - 1) / 4}
            for rank in range(1, 6)
        ],
        "warning": "AI-generated candidates are unconfirmed.",
    }


def predict_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[AIInferenceClient, httpx.AsyncClient]:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    client = AIInferenceClient(
        BASE_URL,
        connection_timeout_seconds=TIMEOUT_SECONDS,
        response_timeout_seconds=TIMEOUT_SECONDS,
        http_client=http_client,
    )
    return client, http_client


@pytest.mark.anyio
async def test_predict_video_posts_multipart_and_preserves_candidate_order() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=valid_payload())

    client, http_client = predict_with_handler(handler)
    async with http_client:
        result = await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4", "video/mp4")

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/api/v1/predictions"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    assert request.extensions["timeout"] == {
        "connect": TIMEOUT_SECONDS,
        "read": TIMEOUT_SECONDS,
        "write": TIMEOUT_SECONDS,
        "pool": TIMEOUT_SECONDS,
    }
    assert b'name="video"' in request.content
    assert b'filename="clip.mp4"' in request.content
    assert VIDEO_BYTES in request.content
    assert result.prediction_status == "UNCONFIRMED"
    assert result.model_version == "bilstm-v1"
    assert result.score_type == "uncalibrated_softmax"
    assert [candidate.rank for candidate in result.candidates] == [1, 2, 3, 4, 5]
    assert len({candidate.rank for candidate in result.candidates}) == 5
    assert [candidate.label for candidate in result.candidates] == [
        f"candidate-{rank}" for rank in range(1, 6)
    ]
    assert [candidate.score for candidate in result.candidates] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]


@pytest.mark.anyio
async def test_predict_video_maps_timeout_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    client, http_client = predict_with_handler(handler)
    async with http_client:
        with pytest.raises(AIInferenceUnavailableError, match="timed out"):
            await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")


@pytest.mark.anyio
async def test_predict_video_maps_connection_failure_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    client, http_client = predict_with_handler(handler)
    async with http_client:
        with pytest.raises(AIInferenceUnavailableError, match="unreachable"):
            await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (422, AIInferenceHTTPError),
        (503, AIInferenceUnavailableError),
    ],
)
async def test_predict_video_maps_http_errors(
    status_code: int,
    error_type: type[AIInferenceHTTPError] | type[AIInferenceUnavailableError],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="response body must not enter the error")

    client, http_client = predict_with_handler(handler)
    async with http_client:
        with pytest.raises(error_type):
            await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")


@pytest.mark.anyio
async def test_predict_video_rejects_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    client, http_client = predict_with_handler(handler)
    async with http_client:
        with pytest.raises(AIInferenceResponseError, match="invalid prediction"):
            await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")


@pytest.mark.anyio
async def test_predict_video_rejects_missing_candidates() -> None:
    payload = valid_payload()
    del payload["candidates"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client, http_client = predict_with_handler(handler)
    async with http_client:
        with pytest.raises(AIInferenceResponseError):
            await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")


@pytest.mark.anyio
@pytest.mark.parametrize("candidate_count", [4, 6])
async def test_predict_video_rejects_candidate_count_other_than_five(
    candidate_count: int,
) -> None:
    payload = valid_payload()
    payload["candidates"] = [
        {"label": f"candidate-{rank}", "score": rank / 10}
        for rank in range(1, candidate_count + 1)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client, http_client = predict_with_handler(handler)
    async with http_client:
        with pytest.raises(AIInferenceResponseError):
            await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("first_rank", "second_rank"),
    [
        (1, 1),
        (0, 2),
    ],
)
async def test_predict_video_rejects_explicit_candidate_rank_fields(
    first_rank: int,
    second_rank: int,
) -> None:
    payload = valid_payload()
    payload["candidates"][0]["rank"] = first_rank
    payload["candidates"][1]["rank"] = second_rank

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client, http_client = predict_with_handler(handler)
    async with http_client:
        with pytest.raises(AIInferenceResponseError):
            await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "score",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        True,
        "0.5",
    ],
)
async def test_predict_video_rejects_invalid_scores(score: object) -> None:
    payload = valid_payload()
    payload["candidates"][0]["score"] = score

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(payload, allow_nan=True).encode(),
            headers={"content-type": "application/json"},
        )

    client, http_client = predict_with_handler(handler)
    async with http_client:
        with pytest.raises(AIInferenceResponseError):
            await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("predictionStatus", "CONFIRMED"),
        ("scoreType", "calibrated_confidence"),
        ("modelVersion", "unexpected-model"),
    ],
)
async def test_predict_video_rejects_unexpected_prediction_metadata(
    field: str,
    value: str,
) -> None:
    payload = valid_payload()
    payload[field] = value

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client, http_client = predict_with_handler(handler)
    async with http_client:
        with pytest.raises(AIInferenceResponseError):
            await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")


@pytest.mark.anyio
async def test_predict_video_rejects_missing_model_version() -> None:
    payload = valid_payload()
    del payload["modelVersion"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client, http_client = predict_with_handler(handler)
    async with http_client:
        with pytest.raises(AIInferenceResponseError):
            await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")


@pytest.mark.anyio
@pytest.mark.parametrize("label", ["", "  "])
async def test_predict_video_rejects_empty_candidate_labels(label: str) -> None:
    payload = valid_payload()
    payload["candidates"][0]["label"] = label

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client, http_client = predict_with_handler(handler)
    async with http_client:
        with pytest.raises(AIInferenceResponseError):
            await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")


@pytest.mark.anyio
async def test_predict_video_rejects_unexpected_response_fields() -> None:
    payload = valid_payload()
    payload["unexpected"] = "not part of the AI contract"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client, http_client = predict_with_handler(handler)
    async with http_client:
        with pytest.raises(AIInferenceResponseError):
            await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")


@pytest.mark.anyio
async def test_predict_video_creates_and_closes_owned_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_async_client = httpx.AsyncClient
    created_clients: list[httpx.AsyncClient] = []

    def create_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        client = real_async_client(
            *args,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=valid_payload())
            ),
            **kwargs,
        )
        created_clients.append(client)
        return client

    monkeypatch.setattr(
        "app.infrastructure.ai_clients.video_inference_client.httpx.AsyncClient",
        create_client,
    )
    client = AIInferenceClient(
        BASE_URL,
        connection_timeout_seconds=TIMEOUT_SECONDS,
        response_timeout_seconds=TIMEOUT_SECONDS,
    )

    await client.predict_video(io.BytesIO(VIDEO_BYTES), "clip.mp4")

    assert len(created_clients) == 1
    assert created_clients[0].is_closed
