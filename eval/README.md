# Evaluation suite (for chapter 4 of report)

Lightweight, reproducible benchmarking for inpainting, outpainting (panoramic) and geometric editing. The suite builds fixed 1k-sample benchmarks from Flickr30k + NVIDIA masks (inpainting) or GeoBench 2D (geometric editing), runs methods through unified adapter interfaces, and reports PSNR, LPIPS, FID, and CLIP prompt adherence where applicable.

Dataset fetching and adapter code for baselines written with help from Cursor Composer 2.5.

## Folder structure

```
eval/
  config.py                    # Defaults (dataset IDs, sizes, metric settings)
  utils.py                     # Download, IO, mask/image helpers
  data/                        # Dataset loaders and manifest builders
  masks/                       # Pairing and shared spatial transforms
  metrics/                     # PSNR, LPIPS, CLIP, FID + aggregators
  methods/                     # InpaintingMethod + GeometricEditMethod adapters
  runners/
    run_single.py              # Inpainting evaluation CLI
    run_geometric_edit.py      # GeoBench geometric editing CLI
    run_pano_eval.py           # Panoramic generation evaluation CLI
  output/                      # Generated manifests, images, results
```

## Install

From the repository root:

```bash
pip install -r eval/requirements.txt
```

For inpainting runs, continue using the original wonderworld backbone environment

Notes:

- Metrics are computed via **pyiqa** (`psnr`, `lpips`, `fid`, `clipscore`) when installed.
- `datasets<4.0` is recommended because `nlphuji/flickr30k` uses a Hugging Face loading script.
- Metric model weights download on first evaluation run.

## Workflow

1. Create Flickr30k 1k subset (Hugging Face, no git clone)
2. Download NVIDIA testing masks subset (single archive, selective extraction)
3. Build benchmark manifest CSV
4. Run evaluation for one method

## 1. Flickr30k subset (Hugging Face)

Default dataset: `nlphuji/flickr30k`.

```bash
python -m eval.data.flickr30k --out eval/output/flickr30k_1k --count 1000 --seed 42
```

This will:

1. Load Flickr30k from Hugging Face
2. Shuffle indices with `--seed`
3. Select the first `--count` items
4. Save only selected images under `eval/output/flickr30k_1k/images/`
5. Write `eval/output/flickr30k_1k/flickr30k_1k.jsonl`

Each JSONL row includes `sample_id`, `split`, `dataset_index`, `img_id`, `filename`, `chosen_prompt`, and `all_captions`.

### Streaming alternative (less local materialization)

`nlphuji/flickr30k` downloads `flickr30k-images.zip` (~4 GB) before subsetting. For a lighter path, stream parquet shards and save only selected rows:

```bash
python -m eval.data.flickr30k \
  --dataset lmms-lab/flickr30k \
  --split test \
  --streaming \
  --out eval/output/flickr30k_1k \
  --count 1000 \
  --seed 42
```

Caveat: streaming still iterates the remote dataset once to reach selected indices; it avoids storing all 31k images locally.

Caption selection:

- Default: first caption (`--caption-strategy first`)
- Optional: deterministic per-sample pick (`--caption-strategy seeded_random`)

## 2. NVIDIA irregular testing masks

Testing set only (no training set, no git clone). Downloads one zip/tar archive and extracts only the sampled members.

```bash
python -m eval.data.nvidia_masks --out eval/output/nvidia_masks_test --count 1000 --seed 42
```

If the default URL is unavailable, pass a direct archive link from the [official NVIDIA page](https://nv-adlr.github.io/publication/partialconv-inpainting):

```bash
python -m eval.data.nvidia_masks \
  --out eval/output/nvidia_masks_test \
  --count 1000 \
  --seed 42 \
  --url "https://example.com/path/to/testing_masks.zip"
```

Or set `eval.config.NVIDIA_TEST_MASK_ZIP_URL`.

Output:

- `eval/output/nvidia_masks_test/masks/*.png` (binarized, white = inpaint region)
- `eval/output/nvidia_masks_test/masks.jsonl`

Caveats:

- Official links may redirect to Google Drive; direct HTTP download may require a manual `--url`.
- Masks are stored at original resolution; resizing happens during manifest build.

## 3. Build benchmark manifest

Pairs 1k images with 1k masks deterministically and preprocesses both to 512x512 (for our backbone) using **center crop.**

```bash
python -m eval.data.sample_subset \
  --images-manifest eval/output/flickr30k_1k/flickr30k_1k.jsonl \
  --masks-manifest eval/output/nvidia_masks_test/masks.jsonl \
  --out eval/output/manifests/benchmark_1k.csv \
  --seed 42 \
  --size 512
```

Pairing modes:

- `--pairing fixed` (default): image *i* ↔ mask *i*
- `--pairing random`: seeded random mask permutation

Manifest columns: `sample_id`, `image_path`, `prompt`, `all_captions_json`, `mask_id`, `mask_path`, `target_width`, `target_height`.

Processed images/masks are written under `eval/output/manifests/processed/`.

## 4. Run evaluation

```bash
python -m eval.runners.run_single \
  --method dummy \
  --manifest eval/output/manifests/benchmark_1k.csv \
  --output-dir eval/output/dummy_test \
  --limit 32 \
  --seed 42 \
  --device cpu
```

Outputs under `--output-dir`:


| File           | Contents                    |
| -------------- | --------------------------- |
| `results.csv`  | Per-sample metrics          |
| `summary.json` | Mean/std/min/max per metric |
| `config.json`  | Run configuration snapshot  |
| `images/`      | Generated outputs           |


### CSV columns

`sample_id`, `method`, `prompt`, `image_path`, `mask_path`, `output_path`, `psnr`, `fid`, `lpips`, `clip_score`

### Metrics


| Metric | Region                               | Notes            |
| ------ | ------------------------------------ | ---------------- |
| PSNR   | Masked (default)                     | Higher is better |
| FID    | Batch-level on masked composited set | Lower is better  |
| LPIPS  | Masked composite                     | Lower is better  |
| CLIP   | Full image + prompt                  | Higher is better |


Use `--no-masked-metrics` for full-image PSNR/LPIPS.

FID is computed once over all evaluated samples in a run (requires at least 2 samples). The same batch FID value is recorded in each CSV row.

## Adding a new method

1. Subclass `InpaintingMethod` in `eval/methods/`
2. Implement `load()`, `infer(...)`, optional `unload()`
3. Register in `eval/methods/__init__.py` `METHOD_REGISTRY`

Supported adapters: FLUX.2 klein 4B (our method), FLUX.1 Fill dev, SD1.5 inpainting, BCDM, etc.

## Reproducibility

All sampling, pairing, and caption selection honor `--seed`. Re-running with the same seed and manifests yields identical benchmark rows and metric inputs.

## Caveats

- **Hugging Face**: `nlphuji/flickr30k` may download the full image zip; prefer streaming parquet when disk/bandwidth is limited.
- **NVIDIA hosting**: archive URLs can change; use `--url` when auto-discovery fails.
- **datasets version**: pin `datasets<4.0` for script-based Flickr30k loading.
- **Metric models**: first run downloads LPIPS AlexNet and OpenCLIP ViT-B-32 weights.
- **Dummy method**: returns the input unchanged; masked PSNR/LPIPS should be near-perfect, LPIPS near 0.

---

## GeoBench 2D geometric editing

GeoBench (`[CIawevy/GeoBench](https://huggingface.co/datasets/CIawevy/GeoBench)`) benchmarks **spatially-aware image editing**: moving, resizing, or repositioning objects using source and target geometry masks plus a text description. This integration uses config `**2d`**, split `**data`** (5677 examples), and a reproducible random subset of **1000** rows.

### Model inputs (per example)


| Field        | Role                                                          |
| ------------ | ------------------------------------------------------------- |
| `ori_img`    | Original input image                                          |
| `ori_mask`   | Source object/location mask (white = active region)           |
| `tgt_mask`   | Desired target geometry/location mask (white = active region) |
| `4v_caption` | Textual description used as the primary prompt for editing    |


`edit_prompt` and `edit_param` are kept in manifest metadata but are not used as the default adapter prompt.

### Workflow

1. Build GeoBench 2D subset (Hugging Face, no FreeFine clone)
2. Build geometric-editing benchmark manifest
3. Run evaluation with `run_geometric_edit`

### 1. Build GeoBench 2D subset

```bash
python -m eval.data.geobench_2d --out eval/output/geobench_2d_1k --count 1000 --seed 42 --force
```

This loads only `CIawevy/GeoBench` config `2d` (not `3d` or `sc`), shuffles with `--seed`, selects exactly `--count` rows, and saves:

- `eval/output/geobench_2d_1k/images/`
- `eval/output/geobench_2d_1k/ori_masks/` (aligned to `ori_img` size)
- `eval/output/geobench_2d_1k/tgt_masks/`
- `eval/output/geobench_2d_1k/coarse_inputs/` (HF reference edits)
- `eval/output/geobench_2d_1k/geobench_2d_1k.jsonl`
- Parsed `edit_dx`...`edit_sz` columns plus `edit_param_json`
- `eval/output/geobench_2d_1k/geobench_2d_1k.csv`

### 2. Build benchmark manifest

Resizes image and both masks to 512x512 with aligned center-crop geometry:

```bash
python -m eval.data.manifest_geobench \
  --input eval/output/geobench_2d_1k/geobench_2d_1k.jsonl \
  --out eval/output/manifests/geobench_2d_1k.csv \
  --size 512
```

Manifest columns include `coarse_input_path`, `edit_param_iou`, and processed assets under `eval/output/manifests/processed/` (including `coarse_inputs/`).

At manifest build time, `edit_param` is used to predict `tgt_mask` from `ori_mask` (FreeFine-style affine) and log IoU against the dataset mask. **Inference does not apply `edit_param`** — the FLUX adapter feeds resized masks as-is via a socket-style `GeometrySpec`.

### 3. FLUX geometric editing test (single sample)

Uses `BackbonePipeline` from `backbone/pipeline.py` with identical source/target prompts (`4v_caption`) and socket-style masks (no `edit_param` warp at inference):

```bash
python -m eval.runners.run_geometric_edit \
  --method flux_geom \
  --manifest eval/output/manifests/geobench_2d_1k.csv \
  --output-dir eval/output/geobench_flux_smoke \
  --sample-id 000000 \
  --seed 42 \
  --device cuda \
  --offload \
  --verify
```

Verify panel manually or via:

```bash
python -m eval.metrics.geometric_verify \
  --manifest eval/output/manifests/geobench_2d_1k.csv \
  --output eval/output/geobench_flux_smoke/images/000000.png \
  --sample-id 000000 \
  --out eval/output/geobench_flux_smoke/verify
```

### GeoBench metrics

When `coarse_input_path` is present in the manifest:

- **PSNR / LPIPS** use `coarse_input` as reference (masked on `tgt_mask` when applicable)
- **CLIP score** uses generated image + `4v_caption`

Without `coarse_input_path`:

- **Computes CLIP score** (generated image + `4v_caption`)
- **Skips PSNR, LPIPS, and FID** (recorded as `NaN` in `results.csv`)

Registered geometric methods include `bcdm_flux_geom` (our method), `design_edit` (DesignEdit / SDXL), and `freefine` / `freefine_geom` (FreeFine / SD1.5).

### External baselines (DesignEdit, FreeFine)

Clone upstream repos and use thin GeoBench launchers.

Checkouts live under `eval/external/` (gitignored). DesignEdit’s `model.py` is patched from FreeFine’s evaluation fork to add `infer_2d_edit`.

**Environments** (separate from `wonderworld_eval` / `wonderworld_bw`):


| Method     | Conda env    | Install                                                    |
| ---------- | ------------ | ---------------------------------------------------------- |
| DesignEdit | `DesignEdit` | `pip install -r eval/external/DesignEdit/requirements.txt` |
| FreeFine   | `FreeFine`   | `pip install -r eval/external/FreeFine/requirements.txt`   |


Model paths (override via env vars):

- `DESIGNEDIT_SDXL_PATH` (default: `stabilityai/stable-diffusion-xl-base-1.0`)
- `FREEFINE_SD15_PATH` (default: `runwayml/stable-diffusion-v1-5`)

**Example: DesignEdit smoke test** (single sample):

```bash
cd eval/external/DesignEdit
python run_geometric_edit.py \
  --manifest ../../../output/manifests/geobench_2d_1k.csv \
  --output-dir ../../../output/geobench_designedit_smoke \
  --sample-id 000000 --seed 42 --device cuda
```

## Panoramic Novel View Benchmark

### Overview

Evaluates 3D scene generation quality using a horizontal panoramic camera sweep. Generates 9 views spanning a configurable yaw range (default +-60°) and computes:

- **CS** — CLIP Score: cosine similarity between scene prompt and each rendered view
- **CC** — CLIP Consistency: cosine similarity between each view and the central (yaw=0°) view
- **CIQA** — CLIP-IQA+: no-reference perceptual quality score
- **CAS** — CLIP Aesthetic Score: LAION aesthetic predictor score (0-10)

### Input format

Place input images and a manifest CSV into:

```
eval/outputs/manifests/processed_pano/
```

`manifest.csv` columns: `scene_id`, `image_path`, `prompt`, `notes`

### Commands

1. Generate panorama + evaluate metrics for all scenes:

```bash
python -m eval.runners.run_pano_eval \
  --manifest eval/outputs/manifests/processed_pano/manifest.csv \
  --output-dir /path/to/outputs \
  --batch-name my_run \
  --seed 42 --device cuda
```

1. Run metrics only (views already generated):

```bash
python -m eval.runners.run_pano_eval \
  --manifest eval/outputs/manifests/processed_pano/manifest.csv \
  --output-dir /path/to/outputs \
  --batch-name my_run \
  --skip-generate
```

1. Generate panorama for a single scene:

```bash
python generate_pano.py \
  --input-image eval/outputs/manifests/processed_pano/images/scene_001.png \
  --prompt "a lush forest at golden hour" \
  --output-dir /path/to/outputs \
  --scene-id scene_001 \
  --seed 42 --device cuda
```

Scene expansion follows `config/base-config.yaml` (`num_scenes`, `rotation_path`, etc.).
Full 16-scene runs are slow; use `--limit 1` during development or `--skip-generate`
to iterate on metrics.