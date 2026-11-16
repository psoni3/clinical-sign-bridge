import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal, Protocol

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mov", ".mp4"}
UPLOAD_CHUNK_SIZE = 1024 * 1024
UNCONFIRMED_WARNING = (
    "AI-generated candidates are unconfirmed and are not a clinical interpretation."
)

router = APIRouter()


class InvalidVideoError(ValueError):
    """Raised when an uploaded file is not a usable video."""


class VideoFrameLimitExceededError(ValueError):
    """Raised when an uploaded video exceeds the configured frame limit."""


class UploadTooLargeError(ValueError):
    """Raised when a streamed upload exceeds the configured byte limit."""


class VideoPredictionService(Protocol):
    def predict_video(self, video_path: Path) -> tuple[int, list[tuple[str, float]]]: ...


class PredictionCandidate(BaseModel):
    label: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)


class VideoPredictionResponse(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda name: "".join(
            word.capitalize() if index else word for index, word in enumerate(name.split("_"))
        ),
        populate_by_name=True,
    )

    prediction_status: Literal["UNCONFIRMED"]
    model_version: str
    score_type: Literal["uncalibrated_softmax"]
    frame_count: int = Field(gt=0)
    candidates: list[PredictionCandidate] = Field(min_length=5, max_length=5)
    warning: str


def _copy_upload(video: UploadFile, destination: Path, max_bytes: int) -> int:
    total_bytes = 0
    with destination.open("xb") as output:
        while chunk := video.file.read(
            min(UPLOAD_CHUNK_SIZE, max_bytes - total_bytes + 1)
        ):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise UploadTooLargeError
            output.write(chunk)

    if total_bytes == 0:
        raise InvalidVideoError

    return total_bytes


@router.post(
    "/api/v1/predictions",
    response_model=VideoPredictionResponse,
    tags=["predictions"],
)
async def predict_video(
    request: Request,
    video: Annotated[UploadFile, File(description="MOV or MP4 video upload.")],
) -> VideoPredictionResponse:
    temporary_directory: TemporaryDirectory | None = None
    try:
        service = getattr(request.app.state, "video_prediction_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Video prediction provider is disabled.")

        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Only MOV and MP4 videos are supported.")

        temporary_directory = TemporaryDirectory(
            prefix="clinical-sign-bridge-upload-"
        )
        video_path = Path(temporary_directory.name) / f"upload{suffix}"
        try:
            await run_in_threadpool(
                _copy_upload,
                video,
                video_path,
                request.app.state.max_video_upload_bytes,
            )
        except UploadTooLargeError:
            raise HTTPException(
                status_code=413,
                detail="Video exceeds the configured upload-size limit.",
            ) from None
        except InvalidVideoError:
            raise HTTPException(status_code=422, detail="Uploaded video is empty.") from None

        try:
            frame_count, candidates = await run_in_threadpool(
                service.predict_video,
                video_path,
            )
        except VideoFrameLimitExceededError:
            raise HTTPException(
                status_code=422,
                detail="Video exceeds the configured frame limit.",
            ) from None
        except InvalidVideoError:
            raise HTTPException(status_code=422, detail="Uploaded video is invalid.") from None
        except Exception as error:
            logger.error(
                "Video inference failed; correlation_id=%s error_type=%s",
                request.state.correlation_id,
                type(error).__name__,
            )
            raise HTTPException(
                status_code=500,
                detail="Video inference failed.",
            ) from None

        return VideoPredictionResponse(
            prediction_status="UNCONFIRMED",
            model_version="bilstm-v1",
            score_type="uncalibrated_softmax",
            frame_count=frame_count,
            candidates=[
                PredictionCandidate(label=label, score=score)
                for label, score in candidates
            ],
            warning=UNCONFIRMED_WARNING,
        )
    finally:
        try:
            await video.close()
        finally:
            if temporary_directory is not None:
                await run_in_threadpool(temporary_directory.cleanup)
