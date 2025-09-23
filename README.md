# Cardiac Mask Processing Toolkit

This repository contains two Python command-line tools for processing cardiac medical imaging data (NIfTI).  

1. **Mask Merger** → merges per-structure binary masks into a single labeled NIfTI.  
2. **Cardiac Coordinate Transformer** → converts raw cardiac labels/images into a standardized cardiac coordinate system and generates standard chamber views (SAX, 4CH, 2CH, 3CH).  

Both scripts use [nibabel](https://nipy.org/nibabel/), [numpy](https://numpy.org/), and [scipy](https://scipy.org/) as core dependencies.

---

## 📦 Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/yourname/cardiac-processing.git
cd cardiac-processing
pip install -r requirements.txt
```

Example `requirements.txt`:

```
numpy
scipy
nibabel
tqdm
scikit-image
```

---

## 🩺 Part 1 – Mask Merger

**File:** `merge_masks.py`
**Purpose:** Combine per-structure masks (e.g., aorta, ventricles, atria) into a single labeled volume.

### 🔧 Edit Label Map

At the top of the script:

```python
LABEL_MAP = {
    "aorta.nii.gz": 1,
    "heart_atrium_left.nii.gz": 2,
    "heart_atrium_right.nii.gz": 3,
    "heart_myocardium.nii.gz": 4,
    "heart_ventricle_left.nii.gz": 5,
    "heart_ventricle_right.nii.gz": 6,
    "pulmonary_artery.nii.gz": 7,
}
```

Change these keys/values to match your dataset filenames and desired label IDs.

### ▶️ Run

```bash
python merge_masks.py \
    --input ./CAS/ImageCAS_heart_masks \
    --output ./CAS/ImageCAS_heart_compact_masks
```

**Options:**

* `--overwrite` → overwrite existing outputs
* `--workers N` → number of parallel workers (default = CPU-1)

### 📂 Input / Output

```
input_root/
  case1/
    aorta.nii.gz
    heart_atrium_left.nii.gz
    ...
  case2/
    ...
```

Output:

```
output_root/
  case1_merge.nii.gz
  case2_merge.nii.gz
  ...
```

---

## 🫀 Part 2 – Cardiac Coordinate Transformer

**File:** `cardiac_views.py`
**Purpose:** Align raw cardiac labels/images into a standardized cardiac coordinate system (SAX) and generate chamber views (SAX, 4CH, 2CH, 3CH).

### ▶️ Run

```bash
python cardiac_views.py \
    --labels ./CAS/heart_labels \
    --images ./CAS/heart_images \
    --out ./CAS/heart_views \
    --views all \
    --output-size 160 \
    --margin-scale 1.3 \
    --workers 8
```

### ⚙️ Key Options

* `--labels` (required) → folder of merged label NIfTI files
* `--images` (optional) → folder of original CT/MRI volumes (paired automatically)
* `--out` (required) → base output folder
* `--views` → which views to generate: `sax,4ch,2ch,3ch,all`
* `--align-to` → axis for in-plane alignment (`x` → mid XZ slice, `y` → mid YZ slice)
* `--output-size` → cubic volume size (default 160³ voxels)
* `--margin-scale` → margin scaling factor for field of view
* `--image-interp` → interpolation for images (`nearest`, `linear`)
* `--label-interp` → interpolation for labels (`nearest`)
* `--workers` → number of parallel processes

### 📂 Output

Inside `--out`, you’ll get four subfolders:

```
out/
  labels/       → resampled label volumes (world affine)
  images/       → resampled image volumes (world affine)
  labels_id/    → resampled labels (identity affine)
  images_id/    → resampled images (identity affine)
```

File naming:

* `<case>_<view>_stack_new_affine.nii.gz` (world)
* `<case>_<view>_stack_id_affine.nii.gz` (identity)

### 🧭 Views

* **SAX** – Short-axis stack
* **4CH** – Four-chamber view
* **2CH** – Two-chamber view (rotated +120°)
* **3CH** – Three-chamber view (rotated –120°)

---

## 🚀 Example Workflow

1. **Merge raw masks** per case:

   ```bash
   python merge_masks.py -i ./raw_masks -o ./merged_masks
   ```

2. **Generate views**:

   ```bash
   python cardiac_views.py \
       --labels ./merged_masks \
       --images ./raw_images \
       --out ./views --views all
   ```

---

## 🔍 Notes

* Landmark detection is based on LV, RV, LA, RA labels.
* Largest connected component is automatically chosen to reduce noise.
* If paired images exist, they will be resampled to the same space as labels.
* Ensure labels follow consistent IDs (`LBL_LV=5, LBL_RV=6, ...`) — edit the script header if your dataset differs.

```