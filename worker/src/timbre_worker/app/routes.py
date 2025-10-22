from fastapi import APIRouter, HTTPException
from uuid import uuid4

from .models import (
    GenerationArtifact,
    GenerationMetadata,
    GenerationRequest,
    GenerationStatus,
    JobState,
)

router = APIRouter()

_IN_MEMORY_JOBS: dict[str, GenerationStatus] = {}


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/generate", response_model=GenerationStatus)
async def generate(request: GenerationRequest) -> GenerationStatus:
    job_id = str(uuid4())
    status = GenerationStatus(job_id=job_id, state=JobState.QUEUED)
    _IN_MEMORY_JOBS[job_id] = status
    # TODO: enqueue background worker for actual generation
    return status


@router.get("/status/{job_id}", response_model=GenerationStatus)
async def status(job_id: str) -> GenerationStatus:
    try:
        return _IN_MEMORY_JOBS[job_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


@router.get("/artifact/{job_id}", response_model=GenerationArtifact)
async def artifact(job_id: str) -> GenerationArtifact:
    status = _IN_MEMORY_JOBS.get(job_id)
    if status is None or status.state != JobState.SUCCEEDED:
        raise HTTPException(status_code=404, detail="artifact not available")
    # Stub artifact metadata
    metadata = GenerationMetadata(
        prompt="stub",
        seed=None,
        model_id="riffusion-v1",
        duration_seconds=8,
    )
    return GenerationArtifact(
        job_id=job_id,
        artifact_path=f"/tmp/{job_id}.wav",
        metadata=metadata,
    )
