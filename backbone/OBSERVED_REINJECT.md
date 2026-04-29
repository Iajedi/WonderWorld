# Observed-Region Token Reinjection

## Motivation

During the denoising (edit) phase of UniEdit-Flow, the observed (known)
region can drift colorimetrically due to the velocity fusion and
stride-correction updates applied by `UniEditEulerScheduler`.
Reinjection prevents this by anchoring the observed region to its
inversion-consistent state at every step.

## How It Works

### Inversion trajectory capture

During the T-step inversion phase, a `callback_on_step_end` hook reads
`pipe.scheduler.sample` (the *true* state, not the look-ahead
prediction returned by `UniInvEulerScheduler`) after each scheduler
step.  This produces a trajectory of `T+1` packed latents:

```
inv_trajectory[i]  →  z at noise level σ_inv[i]
```

where `σ_inv = [0, σ_{T-1}, σ_{T-2}, …, σ_1, σ_0]` (flipped schedule).

### Edit-phase reinjection

At each edit step `j` (with noise level `σ_edit[j]`), before the
transformer forward pass:

```
reinject_idx = T − K − j
z_inv = inv_trajectory[reinject_idx]          # [1, N, C]
z_in  = mask · z_cur  +  (1 − mask) · z_inv  # [2, N, C]
```

The transformer then predicts velocity from `z_in`, and the scheduler
advances the latent as usual.

## Tensor shapes

| Tensor           | Shape         | Description                       |
|------------------|---------------|-----------------------------------|
| `z_packed`       | `[2, N, C]`   | Edit latent (src + tgt branches)  |
| `inv_trajectory` | list of `[1, N, C]` | Per-step inversion states   |
| `mask_token`     | `[1, N, 1]`   | 1 = unknown, 0 = observed        |
| `z_inv_batch`    | `[2, N, C]`   | Trajectory entry expanded to batch|

## Mask convention

- **1 = unknown** (hole to fill)
- **0 = observed** (keep from original image)
- The original input mask is used as-is: no feathering, dilation,
  erosion, or smoothing is applied.

## Config

```yaml
observed_reinject: true   # enable/disable in configs/*.yaml
# Reinjection-only: widen unknown in *image* space (pixels) before token resize.
reinject_unknown_expand_px: 12   # 0 = use the same soft mask as warm-start / UniEdit
reinject_mask_dilate: 0          # optional extra unknown dilation in *token* units
```

When `false`, Phase C falls back to the original `pipe(…)` black-box
call.  When `true`, a manual denoising loop replaces it.

`reinject_unknown_expand_px` addresses blur-band artefacts: the image
fed to the VAE has black composited into the hole using a soft mask, so
latent tokens just inside the nominal “known” region can still carry
darkened inversion states.  Expanding unknown for reinjection only
leaves those tokens on the denoising trajectory instead of overwriting
them from `inv_trajectory`.

## Files

- `edit/observed_reinject.py` — `reinject_observed_tokens()` helper
- `edit/controller.py` — `_edit_with_reinject()` manual loop +
  `_inv_callback` trajectory capture
