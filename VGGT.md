## Why VGGT

VGGT provides geometry-grounded representations and fast multi-view token aggregation. We only need patch-level embeddings for Hummingbird’s memory-based segmentation. We leverage the VGGT aggregator to produce per-patch tokens.

---

## Loading the Model

Two methods are supported:
- Hugging Face from_pretrained (recommended):
  - Model id: `facebook/VGGT-1B` (currently the only one available on HF for public access)
  - CLI: `--vggt-hf-id facebook/VGGT-1B`
- Local checkpoint:
  - CLI: `--vggt-ckpt /path/to/checkpoint.pth`

Loader location:
- `hbird/encoders/vggt_encoder.py`
  - `load_vggt(backbone, ckpt_path, hf_model_id, device)` returns a ready, eval-mode VGGT model.
  - Defaults any non-1B shorthand to `facebook/VGGT-1B` on HF for resilience.

Key design choice:
- Prefer HF `from_pretrained` to avoid manual checkpoint management. Fall back to local checkpoints when needed.

---

## Feature Extraction Pipeline

Where:
- Wrapper: `VGGTFeatureExtractor` in `hbird/encoders/vggt_encoder.py`

Input and shape handling:
- Hummingbird provides images as `(B, C, H, W)`.
- VGGT aggregator expects `(B, S, C, H, W)` with `S` = frames/sequence length.
- We add `S=1` for single-view evaluation (unsqueeze at dim=1).

Tokens:
- We call `aggregator(images)` and take the last-level tokens: `(B, P, D)`.
  - `B`: batch size
  - `P`: number of patch tokens (spatial grid flattened)
  - `D`: embedding dimension
- We return `(tokens, None)` to align with the Hummingbird `forward_features` interface.

Spatial resolution coherence:
- Hummingbird uses `input_size` and `patch_size` to set an expected grid: `eval_spatial_resolution = input_size // patch_size`.
- We verify that `P` is a square grid (`sqrt(P)` is integer). If different from `eval_spatial_resolution`, we log the mismatch (user should align input/patch sizes).

Normalization:
- We reuse existing image transforms in `hbird.utils.transforms`. Optionally, `--vggt-normalize` can be used if a specific VGGT normalization is desired.

---

## Datasets and Data Paths (MVImgNet specifics)

MVImgNet dataset structure:
1) Angle-bin structure (preprocessed):
   - `datasets/split_angles_mvimagenet/<class_id>/<angle_bin>/{img, mask}/...`
2) Raw MVImgNet structure with separate masks:
   - Images: `datasets/mvimgnet/<class_id>/<scene_id>/images/*.jpg`
   - Masks: `datasets/mvimgnet/masks/<class_id>/<scene_id>/*.jpg.png`

What we do:
- Auto-detect angle bins directory:
  - If `split_angles_mvimagenet` is present next to `mvimgnet`, we use it for bins.
  - CLI allows explicit `--train-bins 0,15,30` and `--val-bins 45`.
- Auto-detect masks directory:
  - If `mvimgnet/masks` exists, we use it for raw scene-id structures.
- Flexible matching of file/folder names:
  - Image folder variants: `img/`, `images/`, `Images/`, direct images under the scene folder, or a child folder with images.
  - Mask folder variants: `mask/`, `masks/`, or separate masks tree.
  - Mask naming patterns supported:
    - `image.jpg.png` (original)
    - `image.png` (if image is `image.jpg`)
    - `<scene_id>_image.jpg.png` (angle-bin export style)

By making the loader robust to structure differences, we avoid brittle preconditions while preserving correctness.

---

## Hummingbird Flow with VGGT

1) Build memory over training set:
   - Extract `(B, P, D)` tokens for each image; optionally sub-sample if `--memory-size` is given.
   - Compute one-hot label distributions per patch by patchifying masks and averaging within patches.
2) Nearest neighbors:
   - KNN search over memory features (FAISS/ScaNN) for each validation patch.
3) Cross-attention aggregation:
   - Aggregate neighbor labels using normalized similarity (softmax over neighbors).
4) Upsampling and scoring:
   - Reshape predicted patch labels into a grid; upsample to input size.
   - Compute mIoU (Jaccard) per class; return mean across classes if requested.

Outputs:
- For MVImgNet:
  - Returns a list (one per `val_bin`) of arrays with per-class mIoU.
  - Logs include mean across classes for each bin and full per-class values.

---

## VGGT vs prior encoders

Keep everything identical except the encoder:
- Same dataset split (bins for MVImgNet or file sets for VOC).
- Same `input_size`, `patch_size`, `batch_size` (if memory allows).
- Same `n_neighbours`, NN method (FAISS/ScaNN), and `memory_size` (if set).
- Same transforms and ignore indices.

Compare:
- Per-class mIoU arrays and the mean across classes.
- For MVImgNet across angle bins, compare the mean mIoU per bin and optionally the per-class breakdown.

Rationale:
- Controls confounding variables. Any difference is explained by the encoder change.

---

## Example Commands

MVImgNet with angle bins (recommended):
```bash
python eval.py \
  --model vggt-1b \
  --vggt-hf-id facebook/VGGT-1B \
  --dataset-name mvimgnet \
  --train-bins 0,15,30 \
  --val-bins 45 \
  --batch-size 4 \
  --input-size 392 \
  --patch-size 14 \
  --data-dir datasets/mvimgnet
```

If OOM persists:
```bash
python eval.py \
  --model vggt-1b \
  --vggt-hf-id facebook/VGGT-1B \
  --dataset-name mvimgnet \
  --train-bins 0,15 \
  --val-bins 45 \
  --batch-size 2 \
  --input-size 336 \
  --patch-size 16 \
  --memory-size 1024000 \
  --data-dir datasets/mvimgnet
```