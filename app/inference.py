import os
import logging
from typing import Dict, Any, Tuple, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    nib = None

from app.utils import (
    calculate_spleen_volume,
    get_nifti_metadata,
    validate_ct_scan_quality,
    analyze_spleen_mask,
    is_volume_suspicious,
)

logger = logging.getLogger(__name__)

# Check for PyTorch and MONAI availability
try:
    import torch
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        ScaleIntensityRanged,
        CropForegroundd,
        Invertd,
        AsDiscreted,
    )
    from monai.data import decollate_batch, DataLoader, Dataset
    HAS_TORCH_MONAI = True
except ImportError:
    HAS_TORCH_MONAI = False


class SpleenInferenceEngine:
    """
    3D U-Net inference engine for Spleen segmentation from CT scans.
    Loads checkpoint once at startup and reuses exact training preprocessing.
    """

    MODEL_NAME = "unet3d-msd-spleen"

    def __init__(self, model_path: Optional[str] = None, device_str: Optional[str] = None):
        self.model_path = model_path or os.getenv("MODEL_PATH", "models/best.pth")
        self.device_str = device_str
        self.device = None
        self.model = None
        self.pre_transforms = None
        self.post_transforms = None
        self.is_loaded = False

        self._initialize()

    def _initialize(self):
        """Build model architecture, load weights, and prepare transforms."""
        if not HAS_TORCH_MONAI or os.getenv("MOCK_INFERENCE", "0") == "1":
            logger.warning(
                "Running in MOCK / lightweight mode (torch/monai not available or MOCK_INFERENCE=1)."
            )
            self.device_str = "cpu"
            self.is_loaded = True
            return

        # Determine target device
        if self.device_str:
            self.device = torch.device(self.device_str)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_str = str(self.device)

        logger.info(f"Initializing {self.MODEL_NAME} on device: {self.device}")

        # Build 3D U-Net exactly matching notebook training architecture
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(self.device)

        # Load weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.model_path}. "
                "Please place best.pth in models/ directory."
            )

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        # Handle state_dict if checkpoint has top-level wrapper
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Loaded checkpoint from {self.model_path} successfully.")

        # Preprocessing transforms matching model training and validation
        self.pre_transforms = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image"),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
            ]
        )

        # Post-transforms for inverting prediction back to original NIfTI image space
        self.post_transforms = Compose(
            [
                Invertd(
                    keys="pred",
                    transform=self.pre_transforms,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                ),
                AsDiscreted(keys="pred", argmax=True),
            ]
        )

        self.is_loaded = True

    def predict(self, file_path: str) -> Dict[str, Any]:
        """
        Run spleen segmentation on a .nii.gz file and return structured volume prediction
        with morphologic quality diagnostics.
        """
        # 1. CT Scan Validation (OOD check, dimensions, slice thickness)
        scan_qc = validate_ct_scan_quality(file_path)
        pre_scan_warnings = scan_qc["warnings"]

        # Native spacing strictly from file header zooms
        if HAS_NIBABEL:
            try:
                zooms = nib.load(file_path).header.get_zooms()[:3]
                spacing_mm = tuple(float(s) for s in zooms)
            except Exception:
                spacing_mm = tuple(float(s) for s in scan_qc["spacing_mm"])
        else:
            spacing_mm = tuple(float(s) for s in scan_qc["spacing_mm"])

        if not HAS_TORCH_MONAI or os.getenv("MOCK_INFERENCE", "0") == "1" or self.model is None:
            # Fallback mock prediction for fast testing environments
            shape = scan_qc["shape"]
            if HAS_NUMPY and np is not None:
                mask = np.zeros(shape, dtype=np.uint8)
                cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
                r = min(shape) // 6
                for z in range(max(0, cz - r), min(shape[2], cz + r)):
                    for y in range(max(0, cy - r), min(shape[1], cy + r)):
                        for x in range(max(0, cx - r), min(shape[0], cx + r)):
                            if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r ** 2:
                                mask[x, y, z] = 1
            else:
                mask = None

            analysis = analyze_spleen_mask(
                mask=mask,
                spacing_mm=spacing_mm,
                pre_scan_warnings=pre_scan_warnings,
            )

            return {
                "spleen_volume_ml": analysis["spleen_volume_ml"],
                "voxels": analysis["voxels"],
                "spacing_mm": list(spacing_mm),
                "needs_review": analysis["needs_review"],
                "review_reasons": analysis["review_reasons"],
                "connected_components": analysis["connected_components"],
                "model": self.MODEL_NAME,
                "mask": mask,
            }

        # Real inference with PyTorch and MONAI using DataLoader
        ds = Dataset(data=[{"image": file_path}], transform=self.pre_transforms)
        loader = DataLoader(ds, batch_size=1)
        batch_data = next(iter(loader))

        inputs = batch_data["image"].to(self.device)
        roi_size = (160, 160, 160)
        sw_batch_size = 4 if self.device.type == "cuda" else 1

        with torch.no_grad():
            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=self.model,
                overlap=0.25,
            )

        raw_voxels = int((outputs.argmax(1) == 1).sum().item())
        logger.info(f"[Inference Audit] Raw model voxels before inversion (pred.argmax(1) == 1).sum(): {raw_voxels}")

        batch_data["pred"] = outputs
        # Apply inverse transforms to map segmentation back to original CT grid
        decollated = [self.post_transforms(i) for i in decollate_batch(batch_data)]
        inverted_pred = decollated[0]["pred"]

        # Extract binary mask
        mask_np = inverted_pred.squeeze().cpu().numpy().astype(np.uint8)
        binary_mask = (mask_np > 0).astype(np.uint8)
        inv_voxels = int(binary_mask.sum())
        logger.info(f"[Inference Audit] Voxel count after Invertd: {inv_voxels}")

        # Comprehensive morphologic and clinical analysis
        analysis = analyze_spleen_mask(
            mask=binary_mask,
            spacing_mm=spacing_mm,
            pre_scan_warnings=pre_scan_warnings,
        )

        return {
            "spleen_volume_ml": analysis["spleen_volume_ml"],
            "voxels": analysis["voxels"],
            "spacing_mm": list(spacing_mm),
            "needs_review": analysis["needs_review"],
            "review_reasons": analysis["review_reasons"],
            "connected_components": analysis["connected_components"],
            "model": self.MODEL_NAME,
            "mask": binary_mask,
        }


# Global engine instance holder
_engine_instance: Optional[SpleenInferenceEngine] = None


def get_inference_engine(model_path: Optional[str] = None) -> SpleenInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SpleenInferenceEngine(model_path=model_path)
    return _engine_instance
