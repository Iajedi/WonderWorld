# BCOT-HVE: Boundary-Conditioned Optimal Transport with Harmonic Velocity Extension

Training-free warm-start method for inpainting / outpainting with UniEdit-Flow.

## Method overview

When an input image has an unknown (masked) region filled with black pixels,
the source-branch velocity predicted by the DiT in that region is meaningless.
BCOT-HVE inserts a **K-step warm-start phase** before the standard T-step
UniEdit-Flow editing loop:

1. **Optimal transport** transfers known-region source velocity statistics to
   unknown tokens via an entropic Sinkhorn coupling.
2. **Harmonic extension** smooths the transported velocity field and enforces
   Dirichlet boundary continuity from neighbouring known tokens.
3. **Masked Euler update** advances only the unknown-region latent while
   freezing the known region.

After the warm start, the standard UniEdit-Flow masked-velocity-injection
schedule takes over for the remaining T-K steps.

## Four-phase pipeline

```
Phase A  (setup)        VAE encode -> T-step inversion -> mask -> noisy init
Phase B  (warm start)   K steps: OT + harmonic + masked Euler update
Phase C  (UniEdit)      T-K steps: standard masked velocity fusion
Phase D  (decode)       VAE decode -> output image
```

## Code path

```
run_inpaint.py / run_ablation.py
  └─ edit/controller.py :: BCOTHVEPipeline.run()
       ├─ flux_pipeline.py :: image2latent(), invert (UniInvEulerScheduler)
       ├─ edit/warm_start.py :: warm_start_loop()
       │    ├─ WarmStartHookManager  (hooks on transformer blocks)
       │    ├─ edit/mask_transport.py :: build_transport_cost, sinkhorn_transport, barycentric_velocity
       │    ├─ edit/harmonic_extension.py :: harmonic_extend
       │    └─ edit/baselines.py :: kv_replace / masked_attention (ablation)
       ├─ UniEditEulerScheduler (remaining T-K steps via pipe.__call__)
       └─ viz/debug_transport.py  (optional debug outputs)
```

## Tensor shapes

Assuming 512x512 input with Flux2-Klein (VAE 8x downsample, 2x2 patchify):

| Tensor | Shape | Description |
|--------|-------|-------------|
| Image latent (VAE output) | `[1, 16, 64, 64]` | Raw VAE latent |
| Patchified latent | `[1, 64, 32, 32]` | After `_patchify_latents` (channels * 4) |
| Packed latent (token format) | `[B, 1024, 64]` | `B=2` for source + target branch |
| Token mask | `[1, 1024, 1]` | 1 = unknown, 0 = known |
| Mask 2-D | `[32, 32]` | Spatial view of token mask |
| Prompt embeddings | `[2, S, D]` | `S` = text seq len, `D` = model dim |
| Text IDs | `[2, S, 4]` | Positional IDs for text tokens |
| Latent image IDs | `[2, 1024, 4]` | Positional IDs for image tokens |
| Velocity (model output) | `[2, 1024, 64]` | Chunked into `v_src [1,N,C]` + `v_trg [1,N,C]` |
| Hook hidden states | `[2, 1024, D_inner]` | Per-layer; double blocks return image-only |
| OT cost matrix | `[1, U, K]` | `U` = unknown count, `K` = known count |
| Transport plan | `[1, U, K]` | Row-normalised Sinkhorn coupling |
| Laplacian | `[U, U]` | Dense graph Laplacian on masked sub-grid |
| Boundary adjacency | `[U, B_n]` | Unknown-to-boundary adjacency |
| Harmonic output | `[1, U, C]` | Smoothed velocity for unknown tokens |

## Warm-start methods (ablation)

| Flag | Description |
|------|-------------|
| `none` | No warm start; raw target velocity for unknown tokens |
| `kv_replace` | Mean of k-nearest known-token source velocities |
| `masked_attention` | Soft attention from unknown to known tokens |
| `ot_only` | Sinkhorn OT transport + linear alpha blend (no harmonic) |
| `ot_harmonic` | Full BCOT-HVE: OT + screened harmonic extension |

## Key formulas

### OT cost matrix

```
C_ij = ||phi(h_i^U) - phi(h_j^K)||^2
     + lambda_pos * ||p_i - p_j||^2
     + lambda_bdry * d_bdry(j)
```

### Sinkhorn (log-domain)

```
log K = -C / tau
repeat: log u = -logsumexp(log K + log v, dim=-1)
        log v = -logsumexp(log K + log u, dim=-2)
pi = exp(log u + log K + log v),  row-normalised
```

### Harmonic extension

```
(L_M + lambda_s I) v_tilde = lambda_s (alpha v_transport + (1-alpha) v_tgt)
                              + B_adj @ v_bdry_src
```

### Masked Euler update

```
z_{t-1} = (1 - m) * z_t + m * (z_t + dt * v_tilde)
```

Known region is frozen; only unknown tokens are updated.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T` | 50 | Total denoising steps |
| `K` | 10 | Warm-start steps |
| `lambda_pos` | 0.1 | Positional weight in OT cost |
| `lambda_bdry` | 1.0 | Boundary penalty in OT cost |
| `tau` | 0.05 | Sinkhorn entropy regularisation |
| `lambda_s` | 0.5 | Harmonic screening strength |
| `alpha_start` | 0.9 | Initial transport-vs-target blend |
| `alpha_end` | 0.3 | Final transport-vs-target blend |
| `boundary_band_width` | 2 | Known-side boundary width |
| `omega` | 5.0 | UniEdit guidance correction strength |
| `sinkhorn_iters` | 100 | Sinkhorn iteration count |
| `connectivity` | 4 | Grid adjacency (4 or 8) |

## Running

```bash
cd backbone

# Single inpainting example
python run_inpaint.py --config configs/inpaint.yaml

# Ablation (all 5 warm methods)
python run_ablation.py --config configs/inpaint.yaml --outdir outputs/ablation

# Outpainting
python run_inpaint.py --config configs/outpaint.yaml --outdir outputs/bcot_hve_outpaint
```

## Metrics

- **Preservation error**: MSE between original and decoded known-region pixels.
- **Boundary seam score**: Mean gradient magnitude in a thin band around the
  mask boundary (lower = smoother transition).
