from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- PATHS (edit as needed) ----------------
SEG_DIR = Path("./4ch_out/labels")        # resampled segmentations (cardiac space)
CT_CUBE_DIR  = Path("./4ch_out/images")   # CT already resampled to the same cardiac space
OUT_IMG_DIR = Path("./4ch_out/4ch_previews")
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- OPTIONS ----------------
# Choose which long-axis plane your 4CH is in:
#   "xz" -> XZ plane (normal is +Y). We take the mid-Y slice, shape (X,Z) then transpose -> (Z,X)
#   "yz" -> YZ plane (normal is +X). We take the mid-X slice, shape (Y,Z) then transpose -> (Z,Y)
PREVIEW_PLANE = "xz"     # change to "yz" if your 4CH view is the YZ plane

WINDOW_CENTER = 40.0     # HU window center (soft tissue)
WINDOW_WIDTH  = 400.0    # HU window width
ALPHA = 0.35             # overlay opacity for labels
DRAW_CONTOURS = False    # if True, draw 1px contours instead of filled regions
NUM_WORKERS = min(8, (os.cpu_count() or 4))

# Label → RGB color mapping (0-1 floats). Adjust to your schema as needed.
# Common heart labels seen in your datasets:
# 1=Aorta, 2=LA, 3=RA, 4=Myocardium, 5=LV, 6=RV, 7=Pulmonary Artery
LABEL_COLORS: Dict[int, Tuple[float, float, float]] = {
    1: (1.0, 0.0, 0.0),            # Aorta - red
    2: (0.0, 1.0, 0.0),            # LA - green
    3: (0.0, 0.0, 1.0),            # RA - blue
    4: (1.0, 1.0, 0.0),            # Myocardium - yellow
    5: (0.0, 1.0, 1.0),            # LV - cyan
    6: (1.0, 0.0, 1.0),            # RV - magenta
    7: (1.0, 0.937255, 0.835294),  # Pulmonary artery - bisque
}

# ---------------- HELPERS ----------------
CASE_ID_RX = re.compile(r"^(\d+)")

def _strip_nii_suffix(name: str) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", name)

def make_ct_cube_name(seg_filename: str) -> str:
    """
    Map segmentation filename -> expected CT-cube filename.
    Example:
      '12.img_cardiac_coordinate_space_new_affine.nii.gz'
        -> '12.img_ct_in_cardiac_space.nii.gz'
    """
    base = re.sub(r"\.nii(\.gz)?$", "", seg_filename)
    core = re.sub(r"cardiac_coordinate_space.*$", "ct_in_cardiac_space", base)
    if core == base:
        core = base + "_ct_in_cardiac_space"
    return core + ".nii.gz"

def window_ct(img: np.ndarray, wc: float, ww: float) -> np.ndarray:
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    img = np.clip(img, lo, hi)
    return (img - lo) / max(1e-6, (hi - lo))  # -> [0,1]

def label_to_rgba(seg2d: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Convert a 2D label map to an RGBA image using LABEL_COLORS."""
    h, w = seg2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    # fallback palette for unseen labels
    fallback = [
        (0.00, 0.80, 0.80),  # cyan
        (0.80, 0.40, 0.00),  # orange
        (0.60, 0.20, 0.80),  # purple
        (0.20, 0.80, 0.20),  # green
        (0.80, 0.20, 0.20),  # red
        (0.20, 0.20, 0.80),  # blue
    ]

    labels = np.unique(seg2d)
    labels = labels[labels != 0]  # skip background
    if labels.size == 0:
        return rgba

    for i, lbl in enumerate(labels):
        col = LABEL_COLORS.get(int(lbl))
        if col is None:
            col = fallback[i % len(fallback)]
        mask = (seg2d == lbl)
        rgba[mask, 0] = col[0]
        rgba[mask, 1] = col[1]
        rgba[mask, 2] = col[2]
        rgba[mask, 3] = alpha

    return rgba

def contours_from_labels(seg2d: np.ndarray) -> np.ndarray:
    """Return an RGBA image with 1-pixel contours for each label."""
    from scipy.ndimage import binary_erosion
    h, w = seg2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    labels = np.unique(seg2d)
    labels = labels[labels != 0]
    if labels.size == 0:
        return rgba

    for i, lbl in enumerate(labels):
        col = LABEL_COLORS.get(int(lbl), (0.9, 0.1, 0.1))
        mask = (seg2d == lbl)
        er = binary_erosion(mask, iterations=1, border_value=0)
        edge = mask ^ er
        rgba[edge, 0] = col[0]
        rgba[edge, 1] = col[1]
        rgba[edge, 2] = col[2]
        rgba[edge, 3] = 1.0
    return rgba

def save_overlay_png(ct2d_norm: np.ndarray, seg2d: np.ndarray, out_path: Path) -> None:
    """Compose and save an overlay PNG: grayscale CT + colored labels."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # background grayscale as RGB
    bg = np.stack([ct2d_norm, ct2d_norm, ct2d_norm], axis=-1)
    if DRAW_CONTOURS:
        ov = contours_from_labels(seg2d)
    else:
        ov = label_to_rgba(seg2d, ALPHA)
    a = ov[..., 3:4]
    comp = a * ov[..., :3] + (1.0 - a) * bg
    plt.imsave(str(out_path), comp)

def save_segcolor_png(seg2d: np.ndarray, out_path: Path) -> None:
    """Save a standalone colored segmentation (no CT), useful for QA."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if DRAW_CONTOURS:
        rgba = contours_from_labels(seg2d)
        plt.imsave(str(out_path), rgba)
    else:
        rgba = label_to_rgba(seg2d, alpha=1.0)
        plt.imsave(str(out_path), rgba)

# ---------------- LONG-AXIS (4CH) SLICE HELPERS ----------------
def extract_4ch_slice(vol: np.ndarray, plane: str = "xz") -> np.ndarray:
    """
    Return a 2D slice in the 4CH long-axis plane with Z as the vertical axis.
      plane="xz": take mid-Y slice -> vol[:, ymid, :] has axes (X, Z) -> transpose to (Z, X)
      plane="yz": take mid-X slice -> vol[xmid, :, :] has axes (Y, Z) -> transpose to (Z, Y)
    """
    assert vol.ndim == 3, "Expected a 3D volume"
    p = plane.lower()
    if p == "xz":
        ymid = vol.shape[1] // 2
        sl = vol[:, ymid, :]          # (X, Z)
    elif p == "yz":
        xmid = vol.shape[0] // 2
        sl = vol[xmid, :, :]          # (Y, Z)
    else:
        raise ValueError("plane must be 'xz' or 'yz'")
    return sl.T                        # (Z, X) or (Z, Y) -> Z is vertical

# ---------------- PER-CASE PIPELINE ----------------
def process_one(seg_path: Path) -> Tuple[str, Optional[str]]:
    try:
        # Load segmentation and extract a long-axis (4CH) slice
        seg_img = nib.load(str(seg_path))
        seg_vol = seg_img.get_fdata()
        seg2d = extract_4ch_slice(seg_vol, PREVIEW_PLANE)

        # Try to load the matching CT cube and slice it the same way
        ct_name = make_ct_cube_name(seg_path.name)
        ct_path = CT_CUBE_DIR / ct_name
        if ct_path.is_file():
            ct_img = nib.load(str(ct_path))
            ct = ct_img.get_fdata().astype(np.float32, copy=False)
            ct2d = extract_4ch_slice(ct, PREVIEW_PLANE)
            ct2d = window_ct(ct2d, WINDOW_CENTER, WINDOW_WIDTH)

            out_overlay = OUT_IMG_DIR / (ct_name.replace(".nii.gz", f"_4ch_{PREVIEW_PLANE}_overlay.png"))
            save_overlay_png(ct2d, seg2d, out_overlay)

            out_segcolor = OUT_IMG_DIR / (ct_name.replace(".nii.gz", f"_4ch_{PREVIEW_PLANE}_segcolor.png"))
            save_segcolor_png(seg2d, out_segcolor)
            return seg_path.name, None

        # If CT cube is missing, still save the colored segmentation
        out_seg_only = OUT_IMG_DIR / (seg_path.name.replace(".nii.gz", f"_4ch_{PREVIEW_PLANE}_segcolor.png"))
        save_segcolor_png(seg2d, out_seg_only)
        return seg_path.name, None

    except Exception as e:  # noqa: BLE001
        return seg_path.name, f"{type(e).__name__}: {e}"

# ---------------- MAIN ----------------
def main() -> None:
    seg_files = sorted([p for p in SEG_DIR.glob("*.nii*") if p.is_file()])
    if not seg_files:
        print(f"No NIfTI files found under: {SEG_DIR}")
        return

    futures = []
    errors = []
    done_ct = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS, thread_name_prefix="save-4ch-overlay") as ex:
        for seg_path in seg_files:
            futures.append(ex.submit(process_one, seg_path))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Saving 4CH overlays"):
            name, err = fut.result()
            if err is None:
                done_ct += 1
            else:
                errors.append((name, err))

    print(f"\nSaved overlays for {done_ct}/{len(seg_files)} cases -> {OUT_IMG_DIR}")
    if errors:
        print("Errors:")
        for n, e in errors:
            print(f"  - {n}: {e}")

if __name__ == "__main__":
    main()
