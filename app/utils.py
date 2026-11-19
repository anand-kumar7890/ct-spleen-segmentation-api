import os
import gzip
import struct
import logging
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

try:
    import nibabel as nib
    import numpy as np
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    try:
        import numpy as np
    except ImportError:
        np = None

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def calculate_spleen_volume(
    mask_voxels: int, spacing_mm: Tuple[float, float, float]
) -> float:
    """
    Calculate spleen volume in milliliters (mL) from voxel count and spacing.

    Formula:
        Voxel volume (mm³) = spacing_x * spacing_y * spacing_z
        Total volume (mm³) = mask_voxels * Voxel volume
        Total volume (mL)  = Total volume (mm³) / 1000.0

    Args:
        mask_voxels: Number of positive spleen voxels (mask == 1)
        spacing_mm: Tuple of (dx, dy, dz) voxel dimensions in millimeters

    Returns:
        float: Spleen volume in mL, rounded to 1 decimal place.
    """
    if mask_voxels < 0:
        raise ValueError("Voxel count cannot be negative")
    if any(s <= 0 for s in spacing_mm):
        raise ValueError("Voxel spacings must be positive numbers")

    voxel_volume_mm3 = float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2])
    volume_ml = (mask_voxels * voxel_volume_mm3) / 1000.0
    return round(volume_ml, 1)


def validate_ct_scan_quality(file_path: str) -> Dict[str, Any]:
    """
    Inspect raw CT volume before inference to detect Out-of-Distribution (OOD)
    data, non-CT modalities (e.g. MRI), pre-normalized inputs, or extreme slice thicknesses.
    """
    meta = get_nifti_metadata(file_path)
    shape = meta["shape"]
    spacing = meta["spacing_mm"]
    warnings = []

    if len(shape) < 3:
        raise ValueError(f"Expected 3D volume, got dimensions: {shape}")

    if any(dim < 16 for dim in shape[:3]):
        raise ValueError(f"Scan dimensions {shape} too small for 3D abdominal CT segmentation")

    # Check slice thickness
    slice_thickness = spacing[2] if len(spacing) >= 3 else 1.0
    if slice_thickness > 5.5:
        warnings.append(
            f"Thick slice acquisition ({slice_thickness:.1f} mm > 5.0 mm). Volumetric accuracy may be degraded."
        )

    # Check intensity distribution if nibabel is available
    if HAS_NIBABEL:
        try:
            img = nib.load(file_path)
            data = img.get_fdata(dtype=np.float32)
            val_min = float(np.min(data))
            val_max = float(np.max(data))
            val_mean = float(np.mean(data))

            # CT scans have distinctive Hounsfield Units: Air ~ -1000 HU, Bone > +300 HU
            if val_min >= -10.0 and val_max <= 1.0:
                warnings.append(
                    f"Possible pre-normalized volume [min: {val_min:.2f}, max: {val_max:.2f}]. Model expects unnormalized Hounsfield Units."
                )
            elif val_min >= 0.0 and val_max <= 255.0:
                warnings.append(
                    f"Possible 8-bit windowed or non-CT image [0, 255]. Standard calibrated CT HU expected."
                )
            elif val_max < 50.0:
                warnings.append(
                    f"Abnormally low maximum intensity ({val_max:.1f} HU). Lack of contrast or incomplete scan."
                )
        except Exception as e:
            logger.warning(f"Could not compute CT intensity profile: {e}")

    return {
        "is_valid": True,
        "warnings": warnings,
        "spacing_mm": spacing,
        "shape": shape,
    }


def analyze_spleen_mask(
    mask: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    min_plausible_ml: float = 30.0,
    max_plausible_ml: float = 650.0,
    pre_scan_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive morphologic analysis of the predicted spleen mask:
    - Calculates volume in mL and voxel count
    - Performs 3D connected-component analysis to detect false-positive fragmentation
    - Aggregates actionable clinical review reasons
    """
    review_reasons = list(pre_scan_warnings or [])

    if mask is None:
        voxels = 0
        volume_ml = 0.0
        components = 0
        review_reasons.append("Inference returned null mask.")
        return {
            "spleen_volume_ml": 0.0,
            "voxels": 0,
            "connected_components": 0,
            "needs_review": True,
            "review_reasons": review_reasons,
        }

    binary_mask = (mask > 0).astype(np.uint8)
    voxels = int(binary_mask.sum())
    volume_ml = calculate_spleen_volume(voxels, spacing_mm)

    # Empty mask guard
    if voxels == 0 or volume_ml <= 0.0:
        review_reasons.append(
            "Empty segmentation mask (0 spleen voxels detected). Verify organ presence, prior splenectomy, or contrast phase."
        )
        return {
            "spleen_volume_ml": 0.0,
            "voxels": 0,
            "connected_components": 0,
            "needs_review": True,
            "review_reasons": review_reasons,
        }

    # Volume physiological sanity bounds
    if volume_ml < min_plausible_ml:
        review_reasons.append(
            f"Spleen volume ({volume_ml:.1f} mL) is below physiological adult minimum ({min_plausible_ml:.1f} mL). Possible hypoplasia, partial volume loss, or segmentation under-call."
        )
    elif volume_ml > max_plausible_ml:
        review_reasons.append(
            f"Spleen volume ({volume_ml:.1f} mL) exceeds normal upper threshold ({max_plausible_ml:.1f} mL). Clinical indication of splenomegaly or potential false-positive bleed."
        )

    # Connected component analysis for fragmentation
    connected_components = 1
    if HAS_SCIPY and binary_mask.ndim == 3:
        labeled, num_features = ndimage.label(binary_mask)
        connected_components = int(num_features)
        if num_features > 1:
            # Measure sizes of components
            sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
            sizes = np.sort(sizes)[::-1]
            primary_pct = (sizes[0] / voxels) * 100.0

            if primary_pct < 90.0:
                review_reasons.append(
                    f"Segmentation is fragmented across {num_features} disconnected islands (primary component is only {primary_pct:.1f}% of total). Possible accessory spleen or out-of-organ artifact."
                )

    needs_review = len(review_reasons) > 0

    return {
        "spleen_volume_ml": volume_ml,
        "voxels": voxels,
        "connected_components": connected_components,
        "needs_review": needs_review,
        "review_reasons": review_reasons,
    }


def is_volume_suspicious(
    volume_ml: float,
    voxels: int,
    min_plausible_ml: float = 30.0,
    max_plausible_ml: float = 650.0,
) -> bool:
    """Backward-compatible convenience wrapper."""
    if voxels <= 0 or volume_ml <= 0.0:
        return True
    if volume_ml < min_plausible_ml or volume_ml > max_plausible_ml:
        return True
    return False


def parse_nifti_header_raw(file_path: str) -> Dict[str, Any]:
    """
    Pure-Python parser for NIfTI-1 (.nii or .nii.gz) headers.
    Enables reading spacing and dimensions without requiring heavy external dependencies.
    """
    is_gz = file_path.endswith(".gz")
    open_fn = gzip.open if is_gz else open

    try:
        with open_fn(file_path, "rb") as f:
            hdr = f.read(348)
            if len(hdr) < 348:
                raise ValueError("File is smaller than a 348-byte NIfTI-1 header")

            sizeof_hdr = struct.unpack("<i", hdr[0:4])[0]
            if sizeof_hdr != 348:
                # Try big-endian
                sizeof_hdr = struct.unpack(">i", hdr[0:4])[0]
                endian = ">"
                if sizeof_hdr != 348:
                    raise ValueError(f"Invalid NIfTI header size: {sizeof_hdr} (expected 348)")
            else:
                endian = "<"

            magic = hdr[344:348]
            if magic not in (b"n+1\0", b"ni1\0"):
                raise ValueError(f"Invalid NIfTI magic string: {magic}")

            dim = struct.unpack(f"{endian}8h", hdr[40:56])
            ndim = dim[0]
            shape = tuple(dim[1 : ndim + 1])

            pixdim = struct.unpack(f"{endian}8f", hdr[76:108])
            spacing = tuple(abs(float(pixdim[i])) for i in range(1, min(ndim + 1, 4)))

            # If spacing is missing or degenerate, fallback to default (1.0, 1.0, 1.0)
            if not spacing or any(s <= 0 for s in spacing):
                spacing = (1.0, 1.0, 1.0)

            return {
                "shape": shape,
                "spacing_mm": spacing,
                "ndim": ndim,
            }
    except Exception as e:
        raise ValueError(f"Corrupt or invalid NIfTI file: {str(e)}") from e


def get_nifti_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract spatial metadata (spacing, dimensions, affine) from a NIfTI volume.
    Uses nibabel when available, or falls back to pure-Python header parsing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise ValueError("File is empty (0 bytes)")

    if HAS_NIBABEL:
        try:
            img = nib.load(file_path)
            header = img.header
            zooms = header.get_zooms()[:3]
            spacing = tuple(float(s) for s in zooms)
            return {
                "shape": tuple(int(s) for s in img.shape[:3]),
                "spacing_mm": spacing,
                "affine": img.affine.tolist(),
            }
        except Exception as e:
            raise ValueError(f"Failed to load NIfTI with nibabel: {str(e)}") from e

    # Fallback to pure-Python header inspection
    raw_meta = parse_nifti_header_raw(file_path)
    return {
        "shape": raw_meta["shape"],
        "spacing_mm": raw_meta["spacing_mm"],
        "affine": None,
    }


def save_mask_as_nifti(
    mask_array: Any, reference_nifti_path: str, output_path: str
) -> str:
    """
    Save predicted segmentation mask to a .nii.gz file, preserving
    the exact affine and orientation header of the input scan.
    """
    if not HAS_NIBABEL:
        raise RuntimeError("nibabel is required to save NIfTI masks")

    ref_img = nib.load(reference_nifti_path)
    mask_uint8 = np.asarray(mask_array, dtype=np.uint8)

    # Ensure shape matches reference
    if mask_uint8.shape != ref_img.shape[:3]:
        # Handle cases where channel dimension may be present
        if mask_uint8.ndim == 4:
            mask_uint8 = mask_uint8.squeeze()

    mask_nii = nib.Nifti1Image(mask_uint8, affine=ref_img.affine, header=ref_img.header)
    # Set datatype to uint8
    mask_nii.set_data_dtype(np.uint8)
    nib.save(mask_nii, output_path)
    return output_path
