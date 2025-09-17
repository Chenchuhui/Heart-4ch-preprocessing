from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------------- PATHS ----------------
SEG_DIR = Path("./CAS/ImageCAS_heart_output_new")   # resampled segmentations (cardiac space)
CT_DIR  = Path("./CAS/ImageCAS")                        # original CT intensities (e.g., 12.img.nii.gz)
OUT_DIR = Path("./CAS/newImageCAS")                     # where to save resampled CT
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Extract leading numeric case id from FILENAME only (e.g., "12.img_cardiac..." -> "12")
CASE_ID_RX = re.compile(r"^(\d+)")

# ------------- HELPERS -------------

def _strip_nii_suffix(name: str) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", name)


def find_ct_for_seg(seg_path: Path) -> Optional[Path]:
    """Map a segmentation filename to its source CT image.

    Examples
    --------
    seg: '.../100_merge_cardiac_coordinate_space_new_affine.nii.gz' ->
         CT:  '<CT_DIR>/100.img.nii.gz' (or .nii)

    IMPORTANT: extract the ID from the FILE NAME ONLY (seg_path.name), not directories.
    """
    fname = seg_path.name
    base  = _strip_nii_suffix(fname)
    m = CASE_ID_RX.match(base)
    if not m:
        return None
    case_id = m.group(1)

    # Try exact filenames first
    for cand in (CT_DIR / f"{case_id}.img.nii.gz", CT_DIR / f"{case_id}.img.nii"):
        if cand.is_file():
            return cand

    # Fallback: allow any suffix after .nii (gz or not); search recursively as last resort
    cands = sorted(CT_DIR.glob(f"{case_id}.img.nii*"))
    if not cands:
        cands = sorted(CT_DIR.rglob(f"{case_id}.img.nii*"))
    return cands[0] if cands else None


def make_output_name(seg_path: Path) -> Path:
    base = re.sub(r"\.nii(\.gz)?$", "", seg_path.name)
    core = re.sub(r"cardiac_coordinate_space.*$", "ct_in_cardiac_space", base)
    if core == base:
        core = base + "_ct_in_cardiac_space"
    return OUT_DIR / (core + ".nii.gz")


# ------------- CORE RESAMPLING -------------

def resample_intensity_from_seg(
    seg_resampled_file: Path,
    ct_original_file: Path,
    out_file: Path,
    fill_value: float = 0.0,
    max_chunk_voxels: int = 600_000,
) -> None:
    """Resample the original CT intensity into the same space as the resampled segmentation.

    Chunked mapping avoids huge temporary arrays and plays nicely with threading.
    """
    # Target (resampled seg) defines affine & shape
    seg_resampled = nib.load(str(seg_resampled_file))
    new_affine = seg_resampled.affine.astype(np.float64)
    Sx, Sy, Sz = map(int, seg_resampled.shape)

    # Source CT and source -> index transform
    ct_img = nib.load(str(ct_original_file))
    ct = ct_img.get_fdata().astype(np.float32, copy=False)
    A_src = ct_img.affine.astype(np.float64)

    nx, ny, nz = ct.shape
    interp = RegularGridInterpolator(
        (np.arange(nx), np.arange(ny), np.arange(nz)),
        ct,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )

    # Precompute direct transform: src_idx = inv(A_src) @ new_affine @ out_idx_h
    T = np.linalg.inv(A_src) @ new_affine
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64)

    # Prepare output
    out = np.empty((Sx, Sy, Sz), dtype=np.float32)

    # Process along K in chunks to bound memory
    # Estimate slices per chunk from max_chunk_voxels (~6e5 default)
    vox_per_slice = Sx * Sy
    slabs = max(1, int(max_chunk_voxels // max(1, vox_per_slice)))

    k0 = 0
    while k0 < Sz:
        k1 = min(Sz, k0 + slabs)
        kk = np.arange(k0, k1, dtype=np.float64)[None, None, :]
        ii = np.arange(Sx, dtype=np.float64)[:, None, None]
        jj = np.arange(Sy, dtype=np.float64)[None, :, None]

        # Broadcast to (Sx, Sy, k1-k0)
        # src = R@[i,j,k] + t  (vectorized)
        x_src = R[0, 0] * ii + R[0, 1] * jj + R[0, 2] * kk + t[0]
        y_src = R[1, 0] * ii + R[1, 1] * jj + R[1, 2] * kk + t[1]
        z_src = R[2, 0] * ii + R[2, 1] * jj + R[2, 2] * kk + t[2]

        pts = np.stack([x_src, y_src, z_src], axis=-1).reshape(-1, 3).astype(np.float32, copy=False)
        vals = interp(pts).astype(np.float32, copy=False).reshape(Sx, Sy, k1 - k0)
        out[:, :, k0:k1] = vals
        k0 = k1

    # Save with target affine
    out_img = nib.Nifti1Image(out, new_affine)
    out_img.set_qform(new_affine, code=1)
    out_img.set_sform(new_affine, code=1)
    nib.save(out_img, str(out_file))


# ------------- PER-FILE WRAPPER -------------

def _process_one(seg_path: Path) -> Tuple[str, Optional[str]]:
    """Return (case_key, error_msg). error_msg=None on success."""
    try:
        ct_path = find_ct_for_seg(seg_path)
        if ct_path is None:
            return (seg_path.name, f"No matching CT in {CT_DIR}")

        out_path = make_output_name(seg_path)
        if out_path.exists():
            return (seg_path.name, None)  # already done

        resample_intensity_from_seg(seg_path, ct_path, out_path)
        return (seg_path.name, None)
    except Exception as e:  # noqa: BLE001
        return (seg_path.name, f"{type(e).__name__}: {e}")


# ------------- MAIN (THREADED) -------------

def main(num_workers: Optional[int] = None) -> None:
    seg_files = sorted([p for p in SEG_DIR.glob("*.nii*") if p.is_file()])
    if not seg_files:
        print(f"No NIfTI files found under: {SEG_DIR}")
        return

    if num_workers is None or num_workers <= 0:
        # Threads: conservative default to avoid memory spikes from concurrent interpolation
        cpu = os.cpu_count() or 4
        num_workers = min(max(2, cpu), 8)

    futures = []
    errors = []
    done_ct = 0

    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="ct-resample") as ex:
        for seg_path in seg_files:
            futures.append(ex.submit(_process_one, seg_path))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Resampling CT to cardiac space"):
            name, err = fut.result()
            if err is None:
                done_ct += 1
            else:
                # Skip exists silently, but log real errors
                if not err.startswith("No matching CT"):
                    errors.append((name, err))

    print(f"\nCompleted: {done_ct}/{len(seg_files)} files.")
    if errors:
        print("Errors:")
        for n, e in errors:
            print(f"  - {n}: {e}")


if __name__ == "__main__":
    # Adjust the worker count if you have ample RAM & CPU; default is conservative.
    main(num_workers=None)
