#!/usr/bin/env python3
"""
Merge per-structure NIfTI masks inside each case folder into one labeled volume.

Usage:
  python merge_masks.py --input ./CAS/ImageCAS_heart_masks --output ./CAS/ImageCAS_heart_compact_masks
  # Optional:
  #   --overwrite   : replace existing outputs
  #   --workers N   : parallel workers (default: CPU-1)

Edit LABEL_MAP below if your filenames/labels change.
"""

import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np

# ---------- EDIT ME IF NEEDED ----------
# Map: filename in each case folder -> integer label
LABEL_MAP = {
    "aorta.nii.gz": 1,
    "heart_atrium_left.nii.gz": 2,
    "heart_atrium_right.nii.gz": 3,
    "heart_myocardium.nii.gz": 4,
    "heart_ventricle_left.nii.gz": 5,
    "heart_ventricle_right.nii.gz": 6,
    "pulmonary_artery.nii.gz": 7,
}
OUT_NAME_TEMPLATE = "{case}_merge.nii.gz"
MASK_THRESHOLD = 0.0          # voxel is foreground if > this
OUTPUT_DTYPE   = np.int16     # output label dtype
# ---------------------------------------


def _find_file(case_dir: Path, name: str) -> Path | None:
    """Find name as-is, or swap .nii<->.nii.gz."""
    p = case_dir / name
    if p.exists():
        return p
    if name.endswith(".nii.gz"):
        q = case_dir / name[:-3]  # .nii
        return q if q.exists() else None
    if name.endswith(".nii"):
        q = case_dir / (name + ".gz")  # .nii.gz
        return q if q.exists() else None
    return None


def _combine_one_case(case_dir: Path, out_root: Path, overwrite: bool) -> tuple[str, bool, str]:
    logs = [f"Processing: {case_dir}"]
    case_name = case_dir.name
    out_path = out_root / OUT_NAME_TEMPLATE.format(case=case_name)

    if out_path.exists() and not overwrite:
        logs.append(f"  ⏭️  Skipped (already exists): {out_path}")
        return case_name, True, "\n".join(logs)

    # pick the first existing file as reference (to get shape/affine)
    ref_path = None
    ordered_files = []
    for fname, lbl in LABEL_MAP.items():
        fpath = _find_file(case_dir, fname)
        ordered_files.append((fname, lbl, fpath))
        if ref_path is None and fpath is not None:
            ref_path = fpath

    if ref_path is None:
        logs.append("  ⚠️  No expected mask files found — skipped.")
        return case_name, False, "\n".join(logs)

    try:
        ref_img = nib.load(str(ref_path))
        shape, affine = ref_img.shape, ref_img.affine
        combined = np.zeros(shape, dtype=OUTPUT_DTYPE)

        applied = 0
        for fname, label_val, fpath in ordered_files:
            if fpath is None:
                logs.append(f"  ⚠️ Missing: {fname}")
                continue

            try:
                img = nib.load(str(fpath))
                seg = img.get_fdata()

                if seg.shape != shape:
                    logs.append(f"  ❌ Shape mismatch for {fname}: {seg.shape} vs {shape} (skipped)")
                    continue

                mask = seg > MASK_THRESHOLD
                combined[mask] = label_val
                applied += 1
            except Exception as e:
                logs.append(f"  ❌ Failed to read/apply {fname}: {e}")

        if applied == 0:
            logs.append("  ⚠️  No masks applied — skipped.")
            return case_name, False, "\n".join(logs)

        out_img = nib.Nifti1Image(combined, affine)
        out_img.set_qform(affine, code=1)
        out_img.set_sform(affine, code=1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_img, str(out_path))
        logs.append(f"  ✅ Saved: {out_path}")
        return case_name, True, "\n".join(logs)

    except Exception as e:
        logs.append(f"  💥 Error: {e}")
        return case_name, False, "\n".join(logs)


def _list_cases(root: Path) -> list[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def main():
    parser = argparse.ArgumentParser(description="Merge structure masks per case into one labeled NIfTI.")
    parser.add_argument("--input", "-i", required=True, type=Path, help="Root folder containing case subfolders.")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Output folder.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="Parallel workers (default: CPU-1).")
    args = parser.parse_args()

    in_root: Path = args.input
    out_root: Path = args.output
    out_root.mkdir(parents=True, exist_ok=True)

    cases = _list_cases(in_root)
    if not cases:
        print(f"No case folders found under: {in_root}")
        return

    print(f"Found {len(cases)} case(s). Writing to: {out_root}")

    ok_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_combine_one_case, c, out_root, args.overwrite) for c in cases]
        for fut in as_completed(futures):
            case_name, ok, log = fut.result()
            print(log, flush=True)
            if ok:
                ok_count += 1

    print(f"\nDone. {ok_count}/{len(cases)} case(s) processed or skipped successfully.")


if __name__ == "__main__":
    main()
