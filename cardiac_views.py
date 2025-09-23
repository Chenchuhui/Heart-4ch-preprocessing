#!/usr/bin/env python3
import os, re, glob
import argparse
import math
import numpy as np
import nibabel as nib
import scipy.ndimage as nd
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import label as cc_label
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ------------------------
# Label ids (edit if needed)
# ------------------------
LBL_LV, LBL_RV, LBL_RA, LBL_LA = 5, 6, 3, 2

# ------------------------
# Utilities
# ------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def is_nii(name: str) -> bool:
    n = name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")

def largest_cc(binary: np.ndarray) -> np.ndarray:
    if binary.max() == 0:
        return binary
    lab = cc_label(binary)
    idx = np.argmax(np.bincount(lab.flat)[1:]) + 1
    return (lab == idx).astype(binary.dtype)

def unit(v: np.ndarray, eps: float = 1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps or not np.isfinite(n):
        return None
    return v / n

def pick_perp(z_hat: np.ndarray):
    # choose a canonical axis least aligned with z, then orthogonalize
    cands = np.eye(3)
    dots = np.abs(cands @ z_hat)
    x0 = cands[np.argmin(dots)]
    x = x0 - np.dot(x0, z_hat) * z_hat
    return unit(x)

def best_affine(img: nib.Nifti1Image) -> np.ndarray:
    A = img.get_qform()
    if A is None or not np.any(A):
        A = img.affine
    return A

def save_nifti(data: np.ndarray, affine: np.ndarray, out_path: str):
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, out_path)

def resample_volume(vol: np.ndarray, A_src: np.ndarray, A_dst: np.ndarray,
                    out_size: int, method: str = "nearest", chunk_pts: int = 4_000_000) -> np.ndarray:
    nx, ny, nz = vol.shape
    xi, yi, zi = np.arange(nx), np.arange(ny), np.arange(nz)
    interp = RegularGridInterpolator((xi, yi, zi), vol, method=method,
                                     bounds_error=False, fill_value=0)

    I, J, K = np.meshgrid(np.arange(out_size),
                          np.arange(out_size),
                          np.arange(out_size), indexing='ij')
    ijk_h = np.stack([I, J, K, np.ones_like(I)], axis=-1).reshape(-1, 4)
    del I, J, K

    pts_src = (np.linalg.inv(A_src) @ (A_dst @ ijk_h.T))[:3, :].T
    del ijk_h

    N = pts_src.shape[0]
    out = np.empty(N, dtype=np.float32)
    for s in range(0, N, chunk_pts):
        e = min(N, s + chunk_pts)
        out[s:e] = interp(pts_src[s:e])
    return out.reshape(out_size, out_size, out_size)

# ------------------------
# Landmark finding (in source index space)
# ------------------------
def find_landmarks(seg: np.ndarray):
    LV = largest_cc((seg == LBL_LV).astype(np.uint8))
    RV = largest_cc((seg == LBL_RV).astype(np.uint8))
    RA = largest_cc((seg == LBL_RA).astype(np.uint8))
    LA = largest_cc((seg == LBL_LA).astype(np.uint8))
    FG = largest_cc((seg != 0).astype(np.uint8))

    # MVC: LV ∩ dilated LA; fallback centroid midpoint
    LA_d = nd.binary_dilation(LA).astype(np.uint8)
    inter = (LV * LA_d) == 1
    if inter.any():
        mvc = np.mean(np.where(inter), axis=1)
    else:
        mvc = 0.5*(np.mean(np.where(LV==1),axis=1) + np.mean(np.where(LA==1),axis=1))

    # Apex: farthest LV voxel from MVC
    LV_pts = np.column_stack(np.where(LV == 1))
    if LV_pts.size == 0:
        raise RuntimeError("LV not found")
    apex = LV_pts[np.argmax(np.linalg.norm(LV_pts - mvc[None, :], axis=1))].astype(float)

    # TVC: RV ∩ dilated RA; fallback centroid midpoint
    RA_d = nd.binary_dilation(RA).astype(np.uint8)
    inter = (RV * RA_d) == 1
    if inter.any():
        tvc = np.mean(np.where(inter), axis=1)
    else:
        tvc = 0.5*(np.mean(np.where(RV==1),axis=1) + np.mean(np.where(RA==1),axis=1))

    # Center of heart
    if FG.max() == 0:
        raise RuntimeError("Foreground mask empty")
    coh = np.mean(np.where(FG == 1), axis=1)

    return apex, mvc, tvc, coh

# ------------------------
# Build SAX (COH-centered) affine from raw label
# ------------------------
def build_sax_affine_from_label(seg_img: nib.Nifti1Image, out_size: int, margin_scale: float):
    seg = seg_img.get_fdata()
    A_src = best_affine(seg_img)

    apex_i, mvc_i, tvc_i, coh_i = find_landmarks(seg)

    def i2w(p): return (A_src @ np.r_[p, 1.0])[:3]
    apex_w, mvc_w, tvc_w, coh_w = map(i2w, [apex_i, mvc_i, tvc_i, coh_i])

    # z: long axis
    z = unit(mvc_w - apex_w)
    if z is None:
        raise RuntimeError("Long axis ill-defined")

    # x: TVC projected to SAX plane
    ref = tvc_w - mvc_w
    ref_in_plane = ref - np.dot(ref, z) * z
    x0 = unit(ref_in_plane) if np.linalg.norm(ref_in_plane) > 1e-6 else pick_perp(z)

    # y, then re-orthogonalize x
    y = unit(np.cross(z, x0))
    x = unit(np.cross(y, z))
    if x is None or y is None:
        raise RuntimeError("Failed to build orthonormal basis")

    # --- spacing / FOV (no rounding) ---
    fg = (seg != 0).astype(np.uint8)
    surf = nd.binary_dilation(fg).astype(np.uint8) - fg
    surf_pts = np.column_stack(np.where(surf == 1))
    if surf_pts.size == 0:
        # fallback: use apex↔MVC distance
        max_dist = float(np.linalg.norm(mvc_w - apex_w))
    else:
        vec_apex_mvc_i = apex_i - mvc_i
        dot_res = (surf_pts - mvc_i) @ vec_apex_mvc_i
        far_i = surf_pts[np.argmin(dot_res)]
        far_w = i2w(far_i)
        max_dist = float(np.linalg.norm(far_w - apex_w))

    new_space = (max_dist * margin_scale) / float(out_size)
    if not np.isfinite(new_space) or new_space <= 0:
        raise RuntimeError(f"Bad voxel size: {new_space}")
    new_space = max(new_space, 1e-3)  # clamp to something positive

    R = np.column_stack([x*new_space, y*new_space, z*new_space])

    # Ensure right-handed (det>0); if negative, flip x
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0

    if not np.isfinite(R).all():
        raise RuntimeError("Non-finite entries in rotation/scale matrix")

    center = np.array([out_size/2, out_size/2, out_size/2], dtype=float)
    t = coh_w - R @ center
    if not np.isfinite(t).all():
        raise RuntimeError("Non-finite translation vector")

    A_sax = np.eye(4, dtype=float)
    A_sax[:3, :3] = R
    A_sax[:3, 3]  = t

    return A_sax, A_src, (apex_i, mvc_i, tvc_i)


def build_view_from_sax(A_sax: np.ndarray, A_src: np.ndarray, apex_i, mvc_i, tvc_i,
                         out_size: int, align_to: str = "x", roll_deg: float = 0.0):
    """
    4CH: roll_deg = 0
    2CH: roll_deg = +120
    3CH: roll_deg = -120

    Align the IN-PLANE 4CH short-axis direction to +X (align_to='x') or +Y ('y').
    MVC is placed on the mid-slice.
    """
    R = A_sax[:3, :3]; t = A_sax[:3, 3]
    def i2w_src(p): return (A_src @ np.r_[p, 1.0])[:3]
    apex_w, mvc_w, tvc_w = map(i2w_src, [apex_i, mvc_i, tvc_i])

    # Long axis and 4CH plane normal in world
    z_world = unit(apex_w - mvc_w)
    n_world = unit(np.cross(z_world, tvc_w - mvc_w))  # normal to 4CH plane
    if n_world is None or z_world is None:
        raise RuntimeError("View axes ill-defined")

    # Bring to local (SAX) coords
    n_local = unit(np.linalg.inv(R) @ n_world)
    z_local = np.array([0.0, 0.0, 1.0], dtype=float)

    # In-plane 4CH direction (lies in XY): s_local = z × n
    s_local = unit(np.cross(z_local, n_local))
    if s_local is None:
        s_local = np.array([1.0, 0.0, 0.0], dtype=float)

    # Current angle of s_local from +X, target 0 (X) or +pi/2 (Y)
    beta = math.atan2(s_local[1], s_local[0])
    target = 0.0 if align_to.lower() == "x" else (math.pi / 2.0)
    phi = (target - beta) + math.radians(roll_deg)

    c, s = math.cos(phi), math.sin(phi)
    Bz = np.array([[ c,-s, 0],
                   [ s, c, 0],
                   [ 0, 0, 1]], dtype=float)
    R_new = R @ Bz

    # center-preserving translation
    center = np.array([out_size/2, out_size/2, out_size/2], dtype=float)
    t0 = (R @ center + t) - (R_new @ center)

    # Put MVC on the mid-slice: if align_to='x' -> mid XZ (y=center), else -> mid YZ (x=center)
    invRnew = np.linalg.inv(R_new)
    mvc_local = invRnew @ (mvc_w - t0)

    if align_to.lower() == "x":
        # XZ plane is the view -> force y=center
        delta = center[1] - mvc_local[1]
        shift_idx = np.array([0.0, delta, 0.0])
    else:
        # YZ plane is the view -> force x=center
        delta = center[0] - mvc_local[0]
        shift_idx = np.array([delta, 0.0, 0.0])

    t_new = t0 - (R_new @ shift_idx)

    A_view = np.eye(4, dtype=float)
    A_view[:3, :3] = R_new
    A_view[:3, 3]  = t_new
    return A_view, phi


# ------------------------
# Worker to process a single case
# ------------------------
from typing import Optional

def process_one(stem: str, label_path: str, image_path: Optional[str],
                out_dir_labels: str, out_dir_images: str,
                out_dir_labels_id: str, out_dir_images_id: str,
                views: list[str], out_size: int, margin_scale: float,
                align_to: str, lbl_interp: str, img_interp: str,
                chunk_pts: int):

    # Load label and build SAX (COH-centered)
    lbl_img = nib.load(label_path)
    seg = lbl_img.get_fdata()
    A_sax, A_src, (apex_i, mvc_i, tvc_i) = build_sax_affine_from_label(lbl_img, out_size, margin_scale)

    # Helper to resample & save with suffix
    def emit(A_dst: np.ndarray, view_tag: str):
        seg_r = resample_volume(seg, A_src, A_dst, out_size, method=lbl_interp, chunk_pts=chunk_pts)
        save_nifti(seg_r, A_dst, os.path.join(out_dir_labels, f"{stem}_{view_tag}_stack_new_affine.nii.gz"))
        save_nifti(seg_r, np.eye(4), os.path.join(out_dir_labels_id, f"{stem}_{view_tag}_stack_id_affine.nii.gz"))

        if image_path and os.path.exists(image_path):
            im_img = nib.load(image_path)
            vol = im_img.get_fdata()
            A_img_src = best_affine(im_img)
            im_r = resample_volume(vol, A_img_src, A_dst, out_size, method=img_interp, chunk_pts=chunk_pts)
            save_nifti(im_r, A_dst, os.path.join(out_dir_images, f"{stem}_{view_tag}_stack_new_affine.nii.gz"))
            save_nifti(im_r, np.eye(4), os.path.join(out_dir_images_id, f"{stem}_{view_tag}_stack_id_affine.nii.gz"))

    # 1) SAX (COH-centered)
    if "sax" in views or "all" in views:
        emit(A_sax, "sax")

    # 2) 4CH / 2CH / 3CH (from SAX)
    if any(v in views for v in ("4ch","2ch","3ch","all")):
        mapping = [("4ch", 0.0), ("2ch", +120.0), ("3ch", -120.0)]
        for tag, roll in mapping:
            if tag in views or "all" in views:
                A_view, phi = build_view_from_sax(A_sax, A_src, apex_i, mvc_i, tvc_i,
                                  out_size, align_to=align_to, roll_deg=roll)
                emit(A_view, tag)

    return f"[OK] {stem}"

# ------------------------
# CLI
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate SAX/4CH/2CH/3CH stacks in one pass (parallel).")
    ap.add_argument("--labels", required=True, help="Folder with label NIfTI files")
    ap.add_argument("--images", default="", help="Folder with paired image NIfTI files (optional)")
    ap.add_argument("--out", required=True, help="Output base folder")
    ap.add_argument("--views", default="sax,4ch",
                    help="Comma list from {sax,4ch,2ch,3ch,all}")
    ap.add_argument("--align-to", choices=["x","y"], default="x",
  help="Align the IN-PLANE 4CH short-axis direction to this axis: 'x' -> 4CH=XZ (mid XZ slice), 'y' -> 4CH=YZ (mid YZ slice)")
    ap.add_argument("--output-size", type=int, default= 160)
    ap.add_argument("--margin-scale", type=float, default=1.3)
    ap.add_argument("--image-interp", choices=["nearest","linear"], default="linear")
    ap.add_argument("--label-interp", choices=["nearest"], default="nearest")
    ap.add_argument("--chunk-points", type=int, default=4_000_000)
    ap.add_argument("--workers", type=int, default=max(1, (int(os.cpu_count()/2) or 4) - 1))
    args = ap.parse_args()

    labels_dir = args.labels
    images_dir = args.images if args.images else None
    out_base   = args.out
    views = [v.strip().lower() for v in args.views.split(",")]

    out_labels = os.path.join(out_base, "labels")
    out_images = os.path.join(out_base, "images")
    out_labels_id = os.path.join(out_base, "labels_id")
    out_images_id = os.path.join(out_base, "images_id")

    ensure_dir(out_labels); ensure_dir(out_images); ensure_dir(out_labels_id); ensure_dir(out_images_id)

    files = sorted([f for f in os.listdir(labels_dir) if is_nii(f)])
    if not files:
        print("No NIfTI labels found.")
        return

    def paired_image_path(label_name: str) -> str | None:
        """
        Map '1_merge.nii.gz' -> '<images_dir>/1.img.nii' (or .nii.gz).
        Uses the *first* integer found in the label filename.
        """
        if not images_dir:
            return None

        # 1) Extract the first integer from the label filename
        fname = os.path.basename(label_name)
        m = re.search(r'(\d+)', fname)
        if not m:
            return None
        num = m.group(1)
        num_nozeros = str(int(num))  # handle cases like '001' -> '1'

        # 2) Try exact candidates
        candidates = [
            f"{num}.img.nii",
            f"{num}.img.nii.gz",
            f"{num_nozeros}.img.nii",
            f"{num_nozeros}.img.nii.gz",
        ]
        for rel in candidates:
            p = os.path.join(images_dir, rel)
            if os.path.exists(p):
                return p

        # 3) Fallback: glob for anything like '<num>*.img.nii*'
        pats = [f"{num}*.img.nii*", f"{num_nozeros}*.img.nii*"]
        for pat in pats:
            hits = sorted(glob.glob(os.path.join(images_dir, pat)))
            if hits:
                return hits[0]

        return None


    tasks = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fn in files:
            stem = fn[:-7] if fn.lower().endswith(".nii.gz") else os.path.splitext(fn)[0]
            label_path = os.path.join(labels_dir, fn)
            image_path = paired_image_path(fn)
            tasks.append(ex.submit(process_one, stem, label_path, image_path,
                                   out_labels, out_images, out_labels_id, out_images_id,
                                   views,args.output_size, args.margin_scale,
                                   args.align_to, args.label_interp, args.image_interp,
                                   args.chunk_points))
        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            try:
                msg = fut.result()
                print(msg)
            except Exception as e:
                print(f"[SKIP] {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
