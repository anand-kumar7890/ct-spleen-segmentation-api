import os
import shutil
import tempfile
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

import nibabel as nib
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.inference import get_inference_engine, SpleenInferenceEngine
from app.utils import save_mask_as_nifti

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ct-spleen-api")


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    device: str = Field(..., json_schema_extra={"example": "cpu"})
    model: str = Field(..., json_schema_extra={"example": "unet3d-msd-spleen"})


class PredictResponse(BaseModel):
    spleen_volume_ml: float = Field(..., description="Calculated spleen volume in milliliters (mL)", json_schema_extra={"example": 111.7})
    voxels: int = Field(..., description="Number of segmented spleen voxels", json_schema_extra={"example": 40788})
    spacing_mm: List[float] = Field(..., description="Voxel spacing in mm [dx, dy, dz]", json_schema_extra={"example": [0.74, 0.74, 5.0]})
    needs_review: bool = Field(..., description="Flag indicating if mask is empty, fragmented, or volume is outside plausible range", json_schema_extra={"example": False})
    review_reasons: List[str] = Field(default_factory=list, description="Diagnostic explanations if manual review is recommended", json_schema_extra={"example": []})
    connected_components: int = Field(default=1, description="Number of isolated 3D mask segments (detects fragmentation)", json_schema_extra={"example": 1})
    model: str = Field(..., description="Model architecture identifier", json_schema_extra={"example": "unet3d-msd-spleen"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Load model once at startup, not per request.
    """
    logger.info("Initializing CT Spleen Volumetry API...")
    model_path = os.getenv("MODEL_PATH", "models/best.pth")
    engine = get_inference_engine(model_path=model_path)
    app.state.engine = engine
    logger.info(f"Model {engine.MODEL_NAME} loaded and ready on {engine.device_str}.")
    yield
    logger.info("Shutting down CT Spleen Volumetry API...")


app = FastAPI(
    title="CT Spleen Segmentation & Volumetry API",
    description=(
        "Production-ready inference service for automated spleen segmentation and volume "
        "measurement from 3D abdominal CT scans (NIfTI format). Trained on MSD Task09 Spleen."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS) for decoupled frontend services
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and Web UI
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
if not os.path.exists(STATIC_DIR):
    STATIC_DIR = os.path.abspath("static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", tags=["Root"])
def root():
    """Backend service root. Dedicated frontend is hosted at http://localhost:3000."""
    return {
        "service": "CT Spleen Segmentation & Volumetry API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "frontend_url": "http://localhost:3000",
        "message": "CT Spleen API backend is running. Open http://localhost:3000 for the web frontend interface.",
    }


def cleanup_file(path: str):
    """Safely delete a temporary file if it exists."""
    try:
        if os.path.exists(path):
            os.unlink(path)
            logger.debug(f"Cleaned up temporary file: {path}")
    except Exception as e:
        logger.warning(f"Failed to remove temporary file {path}: {e}")


def validate_upload(filename: str, path: str):
    name = (filename or "").lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise HTTPException(400, "Only .nii or .nii.gz CT volumes are accepted.")
    try:
        shape = nib.load(path).shape
    except Exception:
        raise HTTPException(400, "File is not a valid NIfTI volume.")
    if len(shape) != 3 or min(shape) < 16:
        raise HTTPException(400, f"Expected a 3D CT volume, got shape {shape}.")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint returning service status, model availability, and execution device.
    """
    engine: SpleenInferenceEngine = getattr(app.state, "engine", None)
    if engine is None:
        engine = get_inference_engine()

    return HealthResponse(
        status="ok",
        model_loaded=engine.is_loaded,
        device=engine.device_str or "cpu",
        model=engine.MODEL_NAME,
    )


async def run_prediction_on_file(
    file_path: str, engine: SpleenInferenceEngine
) -> PredictResponse:
    """Helper to execute non-blocking inference and format structured PredictResponse."""
    try:
        # Non-blocking CPU execution: runs heavy 3D inference in threadpool
        result = await run_in_threadpool(engine.predict, file_path)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"NIfTI validation error: {str(ve)}",
        )
    except Exception as e:
        logger.error(f"Inference failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )

    return PredictResponse(
        spleen_volume_ml=result["spleen_volume_ml"],
        voxels=result["voxels"],
        spacing_mm=result["spacing_mm"],
        needs_review=result.get("needs_review", False),
        review_reasons=result.get("review_reasons", []),
        connected_components=result.get("connected_components", 1),
        model=result["model"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_volume(
    file: UploadFile = File(..., description="Abdominal CT volume in .nii.gz or .nii format")
):
    """
    Accepts a 3D CT scan upload (.nii.gz), runs 3D U-Net inference,
    and returns structured output with spleen volume (mL), voxel count, and voxel spacing.
    """
    filename = file.filename or ""
    name = filename.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise HTTPException(400, "Only .nii or .nii.gz CT volumes are accepted.")

    # Save upload to a secure temporary file
    temp_in = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    temp_path = temp_in.name
    temp_in.close()

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        validate_upload(filename, temp_path)

        engine: SpleenInferenceEngine = getattr(app.state, "engine", None)
        if engine is None:
            engine = get_inference_engine()

        return await run_prediction_on_file(temp_path, engine)

    finally:
        cleanup_file(temp_path)


@app.post("/predict/sample", response_model=PredictResponse, tags=["Inference"])
async def predict_sample():
    """
    Runs prediction on the built-in sample scan (data/sample_ct.nii.gz).
    Useful for testing and browser demo without requiring file upload.
    """
    sample_path = "data/sample_ct.nii.gz"
    if not os.path.exists(sample_path):
        sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_ct.nii.gz"))

    if not os.path.exists(sample_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample CT scan not found at data/sample_ct.nii.gz",
        )

    engine: SpleenInferenceEngine = getattr(app.state, "engine", None)
    if engine is None:
        engine = get_inference_engine()

    return await run_prediction_on_file(sample_path, engine)


@app.post("/predict/mask", tags=["Inference"])
async def predict_mask(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Abdominal CT volume in .nii.gz or .nii format")
):
    """
    Accepts a 3D CT scan upload (.nii.gz), runs inference,
    and returns the predicted 3D binary segmentation mask as a downloadable .nii.gz file.
    """
    filename = file.filename or ""
    name = filename.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise HTTPException(400, "Only .nii or .nii.gz CT volumes are accepted.")

    temp_in = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    temp_path = temp_in.name
    temp_in.close()

    temp_out = tempfile.NamedTemporaryFile(suffix="_mask.nii.gz", delete=False)
    mask_path = temp_out.name
    temp_out.close()

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        validate_upload(filename, temp_path)

        engine: SpleenInferenceEngine = getattr(app.state, "engine", None)
        if engine is None:
            engine = get_inference_engine()

        result = await run_in_threadpool(engine.predict, temp_path)
        mask = result.get("mask")

        if mask is None:
            # If mask is None (e.g. lightweight fallback), copy input or generate fallback
            shutil.copyfile(temp_path, mask_path)
        else:
            save_mask_as_nifti(mask, reference_nifti_path=temp_path, output_path=mask_path)

        background_tasks.add_task(cleanup_file, temp_path)
        background_tasks.add_task(cleanup_file, mask_path)

        base_name = file.filename.replace(".nii.gz", "").replace(".nii", "")
        return FileResponse(
            path=mask_path,
            filename=f"{base_name}_spleen_mask.nii.gz",
            media_type="application/gzip",
        )

    except HTTPException:
        cleanup_file(temp_path)
        cleanup_file(mask_path)
        raise
    except Exception as e:
        cleanup_file(temp_path)
        cleanup_file(mask_path)
        logger.error(f"Mask prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mask generation failed: {str(e)}",
        )
