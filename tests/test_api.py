import os
import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.utils import calculate_spleen_volume, get_nifti_metadata

client = TestClient(app)

SAMPLE_CT_PATH = os.path.join(os.path.dirname(__file__), "sample_ct.nii.gz")


def test_health():
    """Verify that GET /health returns 200 and expected status indicators."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "device" in data
    assert data["model"] == "unet3d-msd-spleen"


def test_predict_bad_extension():
    """Verify that uploading a non-NIfTI file (e.g., .txt) returns 400 Bad Request."""
    file_content = b"this is not a nifti volume"
    files = {"file": ("report.txt", file_content, "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_predict_bad_upload_markdown_returns_400():
    """Verify that uploading a markdown file like README.md returns 400, not 500."""
    readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    with open(readme_path, "rb") as f:
        file_bytes = f.read()
    files = {"file": ("README.md", file_bytes, "text/markdown")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_predict_empty_file():
    """Verify that uploading an empty .nii.gz file returns 400 Bad Request."""
    files = {"file": ("empty_scan.nii.gz", b"", "application/gzip")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_corrupt_nifti():
    """Verify that uploading a corrupt .nii.gz file returns 400 Bad Request."""
    corrupt_content = b"\x1f\x8b\x08\x00\x00\x00\x00\x00some_corrupted_garbage_bytes"
    files = {"file": ("corrupt_scan.nii.gz", corrupt_content, "application/gzip")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "validation error" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


def test_predict_valid_sample_volume():
    """Verify that POST /predict with a valid sample CT scan returns structured output."""
    assert os.path.exists(SAMPLE_CT_PATH), f"Sample CT not found at {SAMPLE_CT_PATH}"

    with open(SAMPLE_CT_PATH, "rb") as f:
        file_bytes = f.read()

    files = {"file": ("spleen_test_sample.nii.gz", file_bytes, "application/gzip")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200

    data = response.json()
    assert "spleen_volume_ml" in data
    assert isinstance(data["spleen_volume_ml"], (int, float))
    assert data["spleen_volume_ml"] > 0

    assert "voxels" in data
    assert isinstance(data["voxels"], int)
    assert data["voxels"] > 0

    assert "spacing_mm" in data
    assert len(data["spacing_mm"]) == 3
    for s in data["spacing_mm"]:
        assert s > 0

    assert "needs_review" in data
    assert isinstance(data["needs_review"], bool)

    assert "review_reasons" in data
    assert isinstance(data["review_reasons"], list)

    assert "connected_components" in data
    assert isinstance(data["connected_components"], int)

    assert data["model"] == "unet3d-msd-spleen"


def test_predict_different_spacing_accuracy():
    """
    Test on a real scan whose native spacing is NOT 1.5 x 1.5 x 2.0 mm
    (spleen_15 with spacing [0.74, 0.74, 5.0] mm) and verify that the predicted
    spleen volume matches the Colab reference prediction from spleen_volumes.json (111.7 mL).
    This confirms the API matches the notebook pipeline in native space.
    If spleen_15.nii.gz is not present, this test skips automatically so cloners
    can run pytest without needing the full dataset.
    """
    spleen_15_path = os.path.join(os.path.dirname(__file__), "..", "data", "spleen_15.nii.gz")
    if not os.path.exists(spleen_15_path):
        pytest.skip("spleen_15.nii.gz not present in data/ (skipped to maintain small repo footprint)")

    with open(spleen_15_path, "rb") as f:
        file_bytes = f.read()

    files = {"file": ("spleen_15.nii.gz", file_bytes, "application/gzip")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200

    data = response.json()
    # Native spacing verification
    assert pytest.approx(data["spacing_mm"][0], rel=1e-2) == 0.74
    assert pytest.approx(data["spacing_mm"][1], rel=1e-2) == 0.74
    assert pytest.approx(data["spacing_mm"][2], rel=1e-2) == 5.0

    # Volume must match spleen_volumes.json: 111.7 mL
    assert data["spleen_volume_ml"] == 111.7
    assert data["voxels"] == 40788
    assert data["needs_review"] is False
    assert data["connected_components"] == 1


def test_predict_mask_endpoint():
    """Verify that POST /predict/mask returns a 200 OK downloadable NIfTI stream."""
    assert os.path.exists(SAMPLE_CT_PATH)

    with open(SAMPLE_CT_PATH, "rb") as f:
        file_bytes = f.read()

    files = {"file": ("spleen_test_sample.nii.gz", file_bytes, "application/gzip")}
    response = client.post("/predict/mask", files=files)
    assert response.status_code == 200
    assert len(response.content) > 0
    assert "spleen_test_sample_spleen_mask.nii.gz" in response.headers.get("content-disposition", "")


def test_volume_calculation():
    """Unit test for mathematical correctness of spleen volume calculation."""
    # 1000 voxels with 1mm x 1mm x 1mm spacing = 1.0 mL
    vol1 = calculate_spleen_volume(1000, (1.0, 1.0, 1.0))
    assert vol1 == 1.0

    # 245300 voxels with [0.8, 0.8, 1.5] spacing
    # Voxel volume = 0.8 * 0.8 * 1.5 = 0.96 mm3
    # Total mm3 = 245300 * 0.96 = 235488 mm3 -> 235.5 mL
    vol2 = calculate_spleen_volume(245300, (0.8, 0.8, 1.5))
    assert vol2 == 235.5

    # 0 voxels
    vol_zero = calculate_spleen_volume(0, (1.5, 1.5, 2.0))
    assert vol_zero == 0.0

    # Negative voxel count raises ValueError
    with pytest.raises(ValueError):
        calculate_spleen_volume(-10, (1.0, 1.0, 1.0))

    # Degenerate spacing raises ValueError
    with pytest.raises(ValueError):
        calculate_spleen_volume(100, (0.0, 1.0, 1.0))
