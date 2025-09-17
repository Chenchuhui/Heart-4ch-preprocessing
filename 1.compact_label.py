import os
import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ---------- CONFIG ----------
BASE_DIR = r"./CAS/ImageCAS_heart_masks"
OUT_ROOT = r"./CAS/ImageCAS_heart_compact_masks"
os.makedirs(OUT_ROOT, exist_ok=True)

# Use processes (bypass GIL, faster if CPU-bound) or threads (lower RAM)
USE_PROCESSES = True
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)  # leave one core free

# Map filenames (inside each case folder) -> desired label value
LABEL_MAP = {
    "aorta.nii.gz": 1,
    "heart_atrium_left.nii.gz": 2,
    "heart_atrium_right.nii.gz": 3,
    "heart_myocardium.nii.gz": 4,
    "heart_ventricle_left.nii.gz": 5,
    "heart_ventricle_right.nii.gz": 6,
    "pulmonary_artery.nii.gz": 7,
}

def combine_folder(case_dir: str, case_name: str, label_map: dict, out_root: str):
    """
    Combine individual structure masks in one folder into a single label NIfTI.
    Returns a (case_name, success_bool, log_text).
    """
    logs = [f"Processing: {case_dir}"]

    out_name = f"{case_name.split('.', 1)[0]}_merge.nii.gz"
    out_path = os.path.join(out_root, out_name)
    if os.path.exists(out_path):
        logs.append(f"  ⏭️ Skipped (already exists): {out_path}")
        return case_name, True, "\n".join(logs)

    # Find a reference file to get shape & affine
    ref_path = None
    for fname in label_map.keys():
        p = os.path.join(case_dir, fname)
        if os.path.exists(p):
            ref_path = p
            break
        if fname.endswith(".nii.gz"):
            p2 = os.path.join(case_dir, fname[:-3])  # try ".nii"
            if os.path.exists(p2):
                ref_path = p2
                break

    if ref_path is None:
        logs.append(f"⚠️  No expected mask files found in: {case_dir} — skipped.")
        return case_name, False, "\n".join(logs)

    try:
        ref_img = nib.load(ref_path)
        shape   = ref_img.shape
        affine  = ref_img.affine
        combined = np.zeros(shape, dtype=np.int16)

        # Fill the combined array
        for fname, label_val in label_map.items():
            candidates = [fname]
            if fname.endswith(".nii.gz"):
                candidates.append(fname[:-3])   # also try ".nii"
            elif fname.endswith(".nii"):
                candidates.append(fname + ".gz")

            found = None
            for cand in candidates:
                fpath = os.path.join(case_dir, cand)
                if os.path.exists(fpath):
                    found = fpath
                    break

            if not found:
                logs.append(f"  ⚠️ Missing file: {fname} in {case_dir} (skipped this label)")
                continue

            seg_img = nib.load(found)
            seg = seg_img.get_fdata()

            if seg.shape != shape:
                logs.append(f"  ❌ Shape mismatch for {found}: got {seg.shape}, expected {shape}. Skipping this file.")
                continue

            seg_bin = (seg > 0).astype(np.int16)
            combined[seg_bin == 1] = label_val

        out_img = nib.Nifti1Image(combined, affine)
        out_img.set_qform(affine, code=1)
        out_img.set_sform(affine, code=1)
        nib.save(out_img, out_path)
        logs.append(f"  ✅ Saved: {out_path}")
        return case_name, True, "\n".join(logs)
    except Exception as e:
        logs.append(f"  💥 Error: {e}")
        return case_name, False, "\n".join(logs)

def list_cases(base_dir: str):
    cases = []
    for name in sorted(os.listdir(base_dir)):
        case_dir = os.path.join(base_dir, name)
        if os.path.isdir(case_dir):
            cases.append((case_dir, name))
    return cases

def main():
    cases = list_cases(BASE_DIR)
    if not cases:
        print(f"No case folders found under: {BASE_DIR}")
        return

    print(f"Found {len(cases)} case(s). Writing to: {OUT_ROOT}")
    Executor = ProcessPoolExecutor if USE_PROCESSES else ThreadPoolExecutor

    futures = []
    with Executor(max_workers=MAX_WORKERS) as ex:
        for case_dir, name in cases:
            futures.append(
                ex.submit(combine_folder, case_dir, name, LABEL_MAP, OUT_ROOT)
            )

        succeeded = 0
        for fut in as_completed(futures):
            case_name, ok, log_text = fut.result()
            print(log_text)
            if ok:
                succeeded += 1

    print(f"\nDone. {succeeded}/{len(cases)} case(s) processed or skipped successfully.")

if __name__ == "__main__":
    main()
