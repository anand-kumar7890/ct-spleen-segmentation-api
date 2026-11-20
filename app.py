"""
app.py — Hugging Face Spaces (Gradio) entry point for CT Spleen Volumetry.

This file adds a Gradio UI on top of the existing inference engine.
The Docker stack (docker-compose.yml, frontend/, backend/) is unchanged.

Upload a NIfTI CT scan (.nii or .nii.gz) → get back:
  • Spleen volume in mL
  • Voxel count and spacing
  • Automated sanity check status
  • Downloadable segmentation mask (.nii.gz)

NOT clinical software. Research demonstration only.
"""

import os
import io
import tempfile
import time
import logging

import gradio as gr
import numpy as np

from app.inference import get_inference_engine
from app.utils import save_mask_as_nifti

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf-spleen")

# ---------------------------------------------------------------------------
# Load model once at module import (Spaces starts the process once)
# ---------------------------------------------------------------------------
logger.info("Loading SpleenInferenceEngine …")
ENGINE = get_inference_engine(model_path=os.getenv("MODEL_PATH", "models/best.pth"))
logger.info(f"Engine ready — device: {ENGINE.device_str}")

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_ct.nii.gz")

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run(file_path: str):
    """Run inference and return (result_dict, mask_path, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = ENGINE.predict(file_path)
    elapsed = time.perf_counter() - t0

    # Save mask to a temp file so Gradio can serve it as a download
    mask_path = None
    if result.get("mask") is not None:
        try:
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix="_spleen_mask.nii.gz"
            )
            tmp.close()
            save_mask_as_nifti(
                mask_array=result["mask"],
                reference_nifti_path=file_path,
                output_path=tmp.name,
            )
            mask_path = tmp.name
        except Exception as e:
            logger.warning(f"Could not save mask: {e}")

    return result, mask_path, elapsed


def _format_review(needs_review: bool, reasons: list) -> str:
    if not needs_review:
        return "✅ Passed all automated sanity checks"
    lines = ["⚠️ Review recommended:"]
    for r in reasons:
        lines.append(f"  • {r}")
    return "\n".join(lines)


def _format_spacing(spacing) -> str:
    return "  ×  ".join(f"{v:.3f}" for v in spacing) + " mm"


# ---------------------------------------------------------------------------
# Gradio predict functions
# ---------------------------------------------------------------------------

def predict_upload(nifti_file):
    """Called when user uploads a file."""
    if nifti_file is None:
        return (
            "No file uploaded.",
            "", "", "", "", "", None
        )

    file_path = nifti_file.name if hasattr(nifti_file, "name") else nifti_file

    if not (file_path.endswith(".nii") or file_path.endswith(".nii.gz")):
        return (
            "❌ Please upload a NIfTI file (.nii or .nii.gz). "
            "Single-slice images (JPG, PNG, DICOM) cannot give a volume.",
            "", "", "", "", "", None
        )

    try:
        result, mask_path, elapsed = _run(file_path)
    except Exception as e:
        return (f"❌ Inference error: {e}", "", "", "", "", "", None)

    return _build_outputs(result, mask_path, elapsed)


def predict_sample():
    """Called when user clicks 'Try sample scan'."""
    if not os.path.isfile(SAMPLE_PATH):
        return (
            "❌ Sample scan not found. Make sure data/sample_ct.nii.gz exists.",
            "", "", "", "", "", None
        )
    try:
        result, mask_path, elapsed = _run(SAMPLE_PATH)
    except Exception as e:
        return (f"❌ Inference error: {e}", "", "", "", "", "", None)

    return _build_outputs(result, mask_path, elapsed)


def _build_outputs(result, mask_path, elapsed):
    volume    = f"{result['spleen_volume_ml']:.1f} mL"
    voxels    = f"{result['voxels']:,}"
    spacing   = _format_spacing(result["spacing_mm"])
    components = str(result["connected_components"])
    review    = _format_review(result["needs_review"], result["review_reasons"])
    timing    = f"Inference completed in {elapsed:.1f} s  |  Model: {result['model']}"
    return volume, voxels, spacing, components, review, timing, mask_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="CT Spleen Volumetry",
    theme=gr.themes.Base(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    css="""
        #header { text-align: center; margin-bottom: 8px; }
        #disclaimer {
            background: #fff8e1; border: 1px solid #f9a825;
            border-radius: 8px; padding: 10px 14px;
            color: #5c3d00; font-size: 0.85rem; margin-bottom: 12px;
        }
        .result-box textarea { font-size: 1.1rem !important; font-weight: 600; }
    """,
) as demo:

    gr.Markdown(
        """
        # 🫁 CT Spleen Volumetry
        **3D U-Net (MONAI) · MSD Task09 Spleen · Research demo**

        Upload a NIfTI CT scan (`.nii` or `.nii.gz`) to measure spleen volume in mL.
        """,
        elem_id="header",
    )

    gr.HTML(
        """<div id="disclaimer">
        ⚠️ <strong>Not clinical software.</strong>
        This is a portfolio project trained on a small public dataset.
        It has not been validated on hospital data and must not be used
        for any medical decision.
        </div>"""
    )

    with gr.Row():
        # ── Left column: input ──────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Upload scan")
            upload = gr.File(
                label="NIfTI CT volume (.nii / .nii.gz)",
                file_types=[".nii", ".gz"],
                type="filepath",
            )

            with gr.Row():
                run_btn    = gr.Button("▶ Run inference",   variant="primary")
                sample_btn = gr.Button("🧪 Try sample scan", variant="secondary")

            gr.Markdown(
                """
                **Notes**
                - Full 3D abdominal volumes: **20–30 s** on CPU
                - Built-in smoke-test scan: ~2–3 s (output is not a real measurement)
                - Accepts single-volume NIfTI only — no DICOM, no PNG/JPG
                """,
                elem_id="notes",
            )

        # ── Right column: results ───────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Results")

            volume_out = gr.Textbox(
                label="Spleen volume",
                interactive=False,
                elem_classes=["result-box"],
            )
            with gr.Row():
                voxels_out     = gr.Textbox(label="Voxel count",          interactive=False)
                components_out = gr.Textbox(label="Connected components",  interactive=False)

            spacing_out = gr.Textbox(label="Voxel spacing (from NIfTI header)", interactive=False)
            review_out  = gr.Textbox(label="Automated sanity checks", interactive=False, lines=3)
            timing_out  = gr.Textbox(label="Timing / model", interactive=False)

            mask_dl = gr.File(
                label="Download segmentation mask (.nii.gz)",
                interactive=False,
            )

    # ── Wire up buttons ─────────────────────────────────────────────────
    outputs = [volume_out, voxels_out, spacing_out,
               components_out, review_out, timing_out, mask_dl]

    run_btn.click(
        fn=predict_upload,
        inputs=[upload],
        outputs=outputs,
    )
    sample_btn.click(
        fn=predict_sample,
        inputs=[],
        outputs=outputs,
    )

    gr.Markdown(
        """
        ---
        **Source**: [github.com/anand-kumar7890/ct-spleen-segmentation-api](https://github.com/anand-kumar7890/ct-spleen-segmentation-api)
        · Dataset: [MSD Task09 Spleen](http://medicaldecathlon.com/)
        · Training: MONAI tutorial on Google Colab T4
        · Mean absolute error vs MSD labels: **2.8%** (8 scans, not a generalization measure)
        """,
        elem_id="footer",
    )

if __name__ == "__main__":
    demo.launch()
