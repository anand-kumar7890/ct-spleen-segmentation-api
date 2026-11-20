#!/usr/bin/env python3
"""
eval_api.py — Evaluate the CT Spleen API against MSD label volumes.

For each NIfTI scan in --scan-dir, it:
  1. POSTs the file to POST /predict (default: http://localhost:8000/predict).
  2. Reads the matching label from --label-dir and computes the reference volume
     from the label voxel count and header spacing.
  3. Prints a table of label volume, API volume, and percentage error.
  4. Reports mean absolute error (MAE) across all scans.

Usage:
  python eval_api.py --scan-dir data/imagesTr --label-dir data/labelsTr

Requirements:
  pip install requests nibabel numpy

The script skips any scan that does not have a matching label file, and
skips any scan for which the API returns an error. All volumes are in mL.
"""

import argparse
import os
import sys

import requests
import nibabel as nib
import numpy as np


def label_volume_ml(label_path: str) -> float:
    """Return the spleen volume in mL from a binary label NIfTI."""
    img = nib.load(label_path)
    data = np.asarray(img.dataobj)
    # MSD labels: spleen = voxel value 1
    voxel_count = int((data == 1).sum())
    zooms = img.header.get_zooms()
    voxel_vol_mm3 = float(zooms[0]) * float(zooms[1]) * float(zooms[2])
    return round(voxel_count * voxel_vol_mm3 / 1000.0, 1)


def api_volume_ml(scan_path: str, endpoint: str) -> float:
    """POST a scan to the API and return the predicted spleen volume in mL."""
    with open(scan_path, "rb") as fh:
        resp = requests.post(endpoint, files={"file": fh}, timeout=300)
    resp.raise_for_status()
    return float(resp.json()["spleen_volume_ml"])


def find_label(scan_path: str, label_dir: str) -> str | None:
    """
    Given a scan path like imagesTr/spleen_10.nii.gz, return the matching
    label path in label_dir (labelsTr/spleen_10.nii.gz), or None.
    """
    basename = os.path.basename(scan_path)
    candidate = os.path.join(label_dir, basename)
    return candidate if os.path.isfile(candidate) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CT Spleen API against MSD labels.")
    parser.add_argument("--scan-dir", default="data/imagesTr", help="Directory of input scans (.nii.gz)")
    parser.add_argument("--label-dir", default="data/labelsTr", help="Directory of matching label NIfTIs")
    parser.add_argument("--endpoint", default="http://localhost:8000/predict", help="API predict endpoint")
    parser.add_argument("--pattern", default=".nii.gz", help="File extension to match (default: .nii.gz)")
    args = parser.parse_args()

    if not os.path.isdir(args.scan_dir):
        print(f"ERROR: scan directory not found: {args.scan_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.label_dir):
        print(f"ERROR: label directory not found: {args.label_dir}", file=sys.stderr)
        sys.exit(1)

    scans = sorted(
        f for f in os.listdir(args.scan_dir) if f.endswith(args.pattern)
    )

    if not scans:
        print(f"No files matching *{args.pattern} found in {args.scan_dir}", file=sys.stderr)
        sys.exit(1)

    header = f"{'Scan':<18} {'Label (mL)':>12} {'API (mL)':>10} {'Error':>8}"
    print(header)
    print("-" * len(header))

    errors = []
    for scan_name in scans:
        scan_path = os.path.join(args.scan_dir, scan_name)
        label_path = find_label(scan_path, args.label_dir)
        if label_path is None:
            print(f"{'  ' + scan_name:<18}  (no label — skipped)")
            continue

        try:
            ref_vol = label_volume_ml(label_path)
        except Exception as exc:
            print(f"{'  ' + scan_name:<18}  ERROR reading label: {exc}")
            continue

        try:
            pred_vol = api_volume_ml(scan_path, args.endpoint)
        except Exception as exc:
            print(f"{'  ' + scan_name:<18}  ERROR calling API: {exc}")
            continue

        pct_err = (pred_vol - ref_vol) / ref_vol * 100.0
        sign = "+" if pct_err >= 0 else ""
        errors.append(abs(pct_err))
        name = scan_name.replace(".nii.gz", "").replace(".nii", "")
        print(f"{name:<18} {ref_vol:>12.1f} {pred_vol:>10.1f} {sign}{pct_err:>6.1f}%")

    print("-" * len(header))
    if errors:
        mae = sum(errors) / len(errors)
        print(f"\nMean absolute error: {mae:.1f}%  ({len(errors)} scans evaluated)")
        print("\nNote: training and validation scans are not separated.")
        print("This measures API consistency against MSD labels, not generalization.")
    else:
        print("\nNo scans were successfully evaluated.")


if __name__ == "__main__":
    main()
