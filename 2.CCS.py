from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import distance
from skimage.measure import label as cc_label
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Required label indices in the input segmentation (edit if your dataset differs)
# Aorta=1, LA=2, RA=3, LV=5, RV=6 are assumed below.
LABEL_AORTA = 1
LABEL_LA = 2
LABEL_RA = 3
LABEL_LV = 5
LABEL_RV = 6

# Central-ROI radius (in voxels) used to decide if aorta is visible in 4CH plane
AORTA_ROI_RADIUS = 20

# Roll search parameters (degrees)
ROLL_STEP_DEG = 0.2
ROLL_MAX_DEG = 30.0

# Output volume size (isotropic cube)
OUTPUT_SIZE = 160

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_base_name(p: Path) -> str:
    """Return filename without .nii or .nii.gz (or other) suffixes.
    Examples: foo.nii.gz -> foo ; foo.nii -> foo ; foo.img.nii.gz -> foo.img
    """
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def largest_cc(mask: np.ndarray) -> np.ndarray:
    """Return the largest connected component of a binary mask.
    If mask has no positive voxels, returns the original (all-zeros) mask.
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if not mask.any():
        return mask.astype(np.uint8)
    lbl = cc_label(mask, connectivity=1)
    if lbl.max() == 0:
        return mask.astype(np.uint8)
    # bincount ignores 0; add 1 index offset
    largest = 1 + np.argmax(np.bincount(lbl.flat)[1:])
    return (lbl == largest).astype(np.uint8)


# -----------------------------------------------------------------------------
# Landmark detection & cardiac coordinate construction
# -----------------------------------------------------------------------------

def compute_landmarks_and_affine(
    seg: np.ndarray,
    affine_mtx: np.ndarray,
    output_size: int = OUTPUT_SIZE,
    aorta_label: int = LABEL_AORTA,
    lv_label: int = LABEL_LV,
    rv_label: int = LABEL_RV,
    la_label: int = LABEL_LA,
    ra_label: int = LABEL_RA,
    do_simple_roll: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray, float, float, np.ndarray]:
    """Compute key landmarks and the resampling affine for a 4CH-oriented cube.

    Returns
    -------
    landmarks_vox : list of np.ndarray
        [apex, mvc, tvc, lvc, rvc, coh] in voxel coordinates (i,j,k)
    landmarks_ps : (3,6) array
        Same landmarks mapped to patient/world space using `affine_mtx`.
    new_space : float
        Isotropic voxel spacing (mm) for output cube.
    max_distance : float
        MVC→apex distance used to set spacing.
    new_affine_mtx : (4,4) array
        Affine for the resampled cube in cardiac coordinates.
    """
    # --- Extract primary structures and clean up islands ---
    seg = np.asarray(seg)
    seg_bin = (seg != 0)

    seg_lv = largest_cc(seg == lv_label)
    seg_rv = largest_cc(seg == rv_label)
    seg_la = largest_cc(seg == la_label)
    seg_ra = largest_cc(seg == ra_label)
    seg_fgd = largest_cc(seg_bin)

    # --- Mitral valve center (MVC): boundary between LA (dilated) and LV ---
    la_dil = ndimage.binary_dilation(seg_la).astype(np.uint8)
    mvc_idx = np.where((seg_lv > 0) & (la_dil > 0))
    if mvc_idx[0].size == 0:
        # Fallback: use LV centroid as MVC if overlap not found
        mvc = np.mean(np.where(seg_lv > 0), axis=1)
    else:
        mvc = np.mean(mvc_idx, axis=1)

    # --- Apex: farthest LV voxel from MVC (in voxel space, then distance in mm) ---
    lv_pts = np.array(np.where(seg_lv > 0)).T  # (N,3)
    if lv_pts.size == 0:
        raise ValueError("LV not found in segmentation; cannot compute apex.")
    dists = np.linalg.norm(lv_pts - mvc[None, :], axis=1)
    apex = lv_pts[np.argmax(dists)].astype(float)

    # --- Ventricle centroids ---
    lvc = np.mean(np.where(seg_lv > 0), axis=1)
    rvc = np.mean(np.where(seg_rv > 0), axis=1)

    # --- Tricuspid valve center (TVC): boundary between RA (dilated) and RV ---
    ra_dil = ndimage.binary_dilation(seg_ra).astype(np.uint8)
    tvc_idx = np.where((seg_rv > 0) & (ra_dil > 0))
    tvc = np.mean(tvc_idx, axis=1) if tvc_idx[0].size > 0 else np.mean(np.where(seg_rv > 0), axis=1)

    # --- Center of heart ---
    coh = np.mean(np.where(seg_fgd > 0), axis=1)

    # --- Landmarks to world (patient) space ---
    landmarks_vox_list = [apex, mvc, tvc, lvc, rvc, coh]  # voxel ijk
    L = np.ones((4, 6), dtype=float)
    L[:3, :] = np.array(landmarks_vox_list).T
    landmarks_ps = (affine_mtx @ L)[:3, :]  # (3,6)

    apex_ps, mvc_ps, tvc_ps, lvc_ps, rvc_ps, coh_ps = [landmarks_ps[:, i] for i in range(6)]

    # --- Determine spacing from farthest foreground point opposite apex ---
    vec_apex_mvc_vox = apex - mvc

    # foreground surface (one-pixel shell)
    surface = ndimage.binary_dilation(seg_fgd) & (~seg_fgd)
    surf_pts = np.array(np.where(surface)).T  # (M,3)
    if surf_pts.size == 0:
        raise ValueError("No foreground surface found; cannot set spacing.")

    vecs = surf_pts - mvc[None, :]
    dots = vecs @ vec_apex_mvc_vox
    far_idx = np.argmin(dots)  # most opposite direction to apex
    far_pt_vox = surf_pts[far_idx]

    # measure distance in mm in world space
    far_ps = (affine_mtx @ np.r_[far_pt_vox, 1.0])[:3]
    max_distance = float(distance.euclidean(far_ps, apex_ps))
    new_space = (np.round(max_distance) * 1.3) / float(output_size)  # +30% margin

    # --- Build orthonormal cardiac axes ---
    # Long axis (x): apex → MVC (world space)
    x_vec = apex_ps - mvc_ps
    x_norm = np.linalg.norm(x_vec)
    if x_norm < 1e-8:
        raise ValueError("Degenerate long axis (apex≈MVC) detected.")
    x_hat = x_vec / x_norm

    # Base vector toward right heart to define plane (MVC→TVC)
    base = tvc_ps - mvc_ps

    # Normal to 4CH plane: n = x × base
    n_tmp = np.cross(x_hat, base)
    n_norm = np.linalg.norm(n_tmp)
    if n_norm < 1e-6:
        # Fallback to MVC→RVC or MVC→LVC if colinear
        alt = (rvc_ps - mvc_ps) if np.linalg.norm(rvc_ps - mvc_ps) > np.linalg.norm(lvc_ps - mvc_ps) else (lvc_ps - mvc_ps)
        n_tmp = np.cross(x_hat, alt)
        n_norm = np.linalg.norm(n_tmp)
        if n_norm < 1e-6:
            raise ValueError("Cannot construct a valid 4CH plane (degenerate geometry).")
    z_hat = n_tmp / n_norm  # plane normal

    # In-plane second axis: y = z × x  (ensures orthonormal basis)
    y_tmp = np.cross(z_hat, x_hat)
    y_hat = y_tmp / np.linalg.norm(y_tmp)

    def build_affine(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        R = np.column_stack([x * new_space, y * new_space, z * new_space])  # (3,3)
        # Project COH to the 4CH plane through MVC for a stable IPP
        dist_plane = (coh_ps - mvc_ps) @ z
        ipp = coh_ps - dist_plane * z  # point on plane near COH
        center = np.array([output_size / 2.0, output_size / 2.0, output_size / 2.0])
        t = ipp - R @ center
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = t
        return A

    new_affine = build_affine(x_hat, y_hat, z_hat)

    # --- Optional small roll to suppress central aorta visibility on mid-slice ---
    if do_simple_roll and (seg == aorta_label).any():
        # Prepare nearest-neighbor interpolator in original *index* space
        nx, ny, nz = seg.shape
        grid_x = np.arange(nx)
        grid_y = np.arange(ny)
        grid_z = np.arange(nz)
        interp = RegularGridInterpolator((grid_x, grid_y, grid_z), seg, method="nearest", bounds_error=False, fill_value=0)

        inv_aff = np.linalg.inv(affine_mtx)

        def central_aorta_fraction(A: np.ndarray) -> float:
            z0 = output_size // 2
            xs, ys = np.meshgrid(np.arange(output_size), np.arange(output_size), indexing="ij")
            out_ijk = np.stack([xs, ys, np.full_like(xs, z0)], axis=-1).reshape(-1, 3)  # (N,3)
            homog = np.c_[out_ijk, np.ones(out_ijk.shape[0])].T  # (4,N)
            world = (A @ homog)[:3, :].T  # (N,3)
            in_idx = (inv_aff @ np.c_[world, np.ones(world.shape[0])].T)[:3, :].T  # (N,3)
            vals = interp(in_idx).reshape(output_size, output_size)
            # Circular ROI at image center
            cx = cy = output_size // 2
            yy, xx = np.ogrid[:output_size, :output_size]
            mask = (xx - cy) ** 2 + (yy - cx) ** 2 <= (AORTA_ROI_RADIUS ** 2)
            roi = vals[mask]
            if roi.size == 0:
                return 0.0
            return float(np.count_nonzero(roi == aorta_label)) / float(roi.size)

        frac = central_aorta_fraction(new_affine)
        if frac > 1e-6:
            step = np.deg2rad(ROLL_STEP_DEG)
            max_steps = int(ROLL_MAX_DEG / ROLL_STEP_DEG)
            kx, ky, kz = x_hat
            K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=float)

            best_A = new_affine
            best_frac = frac

            # Try rolling + and - around x_hat
            for sign in (+1, -1):
                theta = 0.0
                for _ in range(max_steps):
                    theta += sign * step
                    Rroll = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
                    y_r = (Rroll @ y_hat); y_r /= np.linalg.norm(y_r)
                    z_r = np.cross(x_hat, y_r); z_r /= np.linalg.norm(z_r)
                    A = build_affine(x_hat, y_r, z_r)
                    f = central_aorta_fraction(A)
                    if f < best_frac:
                        best_frac, best_A = f, A
                        if best_frac <= 1e-6:
                            break
                if best_frac <= 1e-6:
                    break
            new_affine = best_A

    return landmarks_vox_list, landmarks_ps, new_space, max_distance, new_affine


# -----------------------------------------------------------------------------
# Resampling per case
# -----------------------------------------------------------------------------

def resample_to_cardiac_cube(
    seg_path: Path,
    out_dir_new_aff: Path,
    out_dir_id_aff: Path,
    output_size: int = OUTPUT_SIZE,
) -> Tuple[str, Optional[str]]:
    """Process a single case: compute cardiac affine, resample, and save outputs.

    Returns
    -------
    (case_name, error_msg)
    """
    try:
        img = nib.load(str(seg_path))
        seg = img.get_fdata()
        affine = img.affine  # robust vs get_qform()

        # Compute landmarks and affine
        _, _, _, _, new_aff = compute_landmarks_and_affine(seg, affine, output_size=output_size)

        # Build target ijk grid (output index space)
        I, J, K = np.meshgrid(
            np.arange(output_size), np.arange(output_size), np.arange(output_size), indexing="ij"
        )
        out_ijk = np.stack([I, J, K], axis=-1).reshape(-1, 3)  # (N,3)

        # Map to world then to original index space
        homog = np.c_[out_ijk, np.ones(out_ijk.shape[0])]
        world = (new_aff @ homog.T)[:3, :].T
        inv_affine = np.linalg.inv(affine)
        src_idx = (inv_affine @ np.c_[world, np.ones(world.shape[0])].T)[:3, :].T  # (N,3)

        # Nearest-neighbor sampling over full grid (chunk to reduce memory peak)
        interp = RegularGridInterpolator(
            (np.arange(seg.shape[0]), np.arange(seg.shape[1]), np.arange(seg.shape[2])),
            seg,
            method="nearest",
            bounds_error=False,
            fill_value=0,
        )

        # Chunked evaluation to keep memory moderate
        N = src_idx.shape[0]
        chunk = 500_000
        out_vals = np.empty(N, dtype=seg.dtype)
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            out_vals[s:e] = interp(src_idx[s:e])

        vol = out_vals.reshape(output_size, output_size, output_size).astype(seg.dtype, copy=False)

        case_base = get_base_name(seg_path)
        out_new = out_dir_new_aff / f"{case_base}_cardiac_coordinate_space_new_affine.nii.gz"
        out_id = out_dir_id_aff / f"{case_base}_cardiac_coordinate_space_id_affine.nii.gz"

        # Save with cardiac affine
        nib.save(nib.Nifti1Image(vol, affine=new_aff), str(out_new))
        # Save with identity affine for e.g., 3D Slicer visualization
        nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), str(out_id))

        return case_base, None

    except Exception as e:  # noqa: BLE001 (propagate error info)
        return seg_path.name, f"{type(e).__name__}: {e}"


# -----------------------------------------------------------------------------
# Parallel driver
# -----------------------------------------------------------------------------

def main(
    data_path,
    output_path_new_aff,
    output_path_id_aff,
    output_size: int = OUTPUT_SIZE,
    num_workers: Optional[int] = None,
) -> None:
    data_dir = Path(data_path)
    out_new = Path(output_path_new_aff)
    out_id = Path(output_path_id_aff)
    out_new.mkdir(parents=True, exist_ok=True)
    out_id.mkdir(parents=True, exist_ok=True)

    seg_files = sorted([p for p in data_dir.iterdir() if p.suffix in {".nii", ".gz"} or str(p).endswith(".nii.gz")])
    if not seg_files:
        raise FileNotFoundError(f"No NIfTI files found under: {data_dir}")

    if num_workers is None or num_workers <= 0:
        # Leave one core free by default
        cpu = int(os.cpu_count()/2) or 1
        num_workers = max(1, cpu - 1)

    tasks = {}
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for p in seg_files:
            fut = ex.submit(resample_to_cardiac_cube, p, out_new, out_id, output_size)
            tasks[fut] = p

        errors = []
        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            case_name, err = fut.result()
            if err:
                errors.append((case_name, err))

    if errors:
        print("\nCompleted with errors on the following cases:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
    else:
        print("\nAll cases processed successfully.")


if __name__ == "__main__":
    # Adjust paths and workers as needed
    main(
        data_path="./CAS/ImageCAS_heart_compact_masks",
        output_path_new_aff="./CAS/ImageCAS_heart_output_new",
        output_path_id_aff="./CAS/ImageCAS_heart_output_id",
        output_size=OUTPUT_SIZE,
        num_workers=None,  # auto (CPU-1)
    )
