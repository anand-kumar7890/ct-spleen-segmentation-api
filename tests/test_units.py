import os
import unittest
import sys

# Ensure ct-spleen-api is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils import (
    calculate_spleen_volume,
    parse_nifti_header_raw,
    get_nifti_metadata,
    is_volume_suspicious,
    validate_ct_scan_quality,
    analyze_spleen_mask,
)
from app.inference import SpleenInferenceEngine

SAMPLE_CT_PATH = os.path.join(os.path.dirname(__file__), "sample_ct.nii.gz")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pth")


class TestSpleenVolumetry(unittest.TestCase):
    def test_volume_calculation_basic(self):
        # 1000 voxels with 1.0mm isotropic resolution = 1.0 mL
        vol = calculate_spleen_volume(1000, (1.0, 1.0, 1.0))
        self.assertEqual(vol, 1.0)

    def test_volume_calculation_clinical_case(self):
        # 245,300 voxels with [0.8, 0.8, 1.5] mm spacing = 235.5 mL
        vol = calculate_spleen_volume(245300, (0.8, 0.8, 1.5))
        self.assertEqual(vol, 235.5)

    def test_volume_zero_voxels(self):
        vol = calculate_spleen_volume(0, (1.5, 1.5, 2.0))
        self.assertEqual(vol, 0.0)

    def test_negative_voxels_raises(self):
        with self.assertRaises(ValueError):
            calculate_spleen_volume(-10, (1.0, 1.0, 1.0))

    def test_invalid_spacing_raises(self):
        with self.assertRaises(ValueError):
            calculate_spleen_volume(500, (0.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            calculate_spleen_volume(500, (-1.0, 1.0, 1.0))

    def test_sample_nifti_metadata(self):
        self.assertTrue(os.path.exists(SAMPLE_CT_PATH))
        meta = get_nifti_metadata(SAMPLE_CT_PATH)
        self.assertEqual(meta["spacing_mm"], (1.5, 1.5, 2.0))
        self.assertEqual(meta["shape"], (64, 64, 32))

    def test_ct_quality_validation(self):
        qc = validate_ct_scan_quality(SAMPLE_CT_PATH)
        self.assertTrue(qc["is_valid"])
        self.assertEqual(qc["shape"], (64, 64, 32))
        self.assertEqual(qc["spacing_mm"], (1.5, 1.5, 2.0))

    def test_analyze_spleen_mask_diagnostics(self):
        import numpy as np
        # Single solid mask
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        mask[10:20, 10:20, 10:20] = 1 # 1000 voxels
        analysis = analyze_spleen_mask(mask, (1.0, 1.0, 1.0))
        self.assertEqual(analysis["spleen_volume_ml"], 1.0)
        self.assertEqual(analysis["connected_components"], 1)
        self.assertTrue(analysis["needs_review"]) # Volume < 30 mL
        self.assertTrue(any("Volume below plausible range" in r for r in analysis["review_reasons"]))

        # Fragmented mask (2 separate components)
        mask2 = np.zeros((50, 50, 50), dtype=np.uint8)
        mask2[5:15, 5:15, 5:15] = 1 # 1000 voxels
        mask2[35:45, 35:45, 35:45] = 1 # 1000 voxels
        analysis2 = analyze_spleen_mask(mask2, (1.0, 1.0, 1.0))
        self.assertEqual(analysis2["connected_components"], 2)
        self.assertTrue(any("fragmented" in r for r in analysis2["review_reasons"]))

    def test_suspicious_volume_flagging(self):
        # Normal volume should not need review
        self.assertFalse(is_volume_suspicious(180.0, 150000))
        # Zero voxels / empty mask
        self.assertTrue(is_volume_suspicious(0.0, 0))
        # Unusually low volume (< 30 mL)
        self.assertTrue(is_volume_suspicious(15.0, 12000))
        # Extreme splenomegaly / spurious segmentation (> 650 mL)
        self.assertTrue(is_volume_suspicious(800.0, 600000))

    def test_inference_engine_prediction(self):
        engine = SpleenInferenceEngine(model_path=MODEL_PATH)
        self.assertTrue(engine.is_loaded)
        result = engine.predict(SAMPLE_CT_PATH)
        self.assertIn("spleen_volume_ml", result)
        self.assertIn("voxels", result)
        self.assertIn("spacing_mm", result)
        self.assertIn("needs_review", result)
        self.assertEqual(result["model"], "unet3d-msd-spleen")
        self.assertGreater(result["spleen_volume_ml"], 0)


if __name__ == "__main__":
    unittest.main()
