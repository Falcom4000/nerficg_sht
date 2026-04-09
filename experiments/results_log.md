# FasterGSDash Experiment Results Log

Scene: bicycle (MipNeRF360, IMAGE_SCALE_FACTOR=0.25)
Hardware: single GPU
Framework: NeRFICG

## Metrics

| ID | Description | Time | PSNR | SSIM | LPIPS | #Gaussians | VRAM (alloc/reserved) | Config delta |
|----|-------------|------|------|------|-------|------------|----------------------|--------------|
| B-FasterGSFused | FasterGSFused baseline (user run) | 7:05 | 25.26 | 0.767 | 0.232 | ? | ? | — |
| B-FasterGS | FasterGS non-fused baseline | 7:46 | 25.30 | 0.767 | 0.231 | 4,807,821 | 5.34 GiB / 7.07 GiB | — |
| B-FasterGSDash-v1 | FasterGSDash Phase 1 (D2 scale<4, factor=5) | ~2:50 | 24.27 | 0.683 | 0.358 | 2,152,007 | ? | INITIAL_MOMENTUM_FACTOR=5 |

## Phase 2 Experiments

| ID | Description | Time | PSNR | SSIM | LPIPS | #Gaussians | VRAM (alloc/reserved) | Config delta |
|----|-------------|------|------|------|-------|------------|----------------------|--------------|
| E2-A | Remove r² threshold scaling (root cause fix) + factor=15 | ~4:30 | **25.12** | **0.748** | **0.284** | **3,319,051** | 3.83/4.83 GiB | Model.py: effective_threshold=grad_threshold |
| E2-B | E2-A + LR decay from lr_decay_from_iter (Conflict J) | — | — | — | — | — | — | Trainer.py LR offset |
| E2-C | E2-A + MAX_N_GAUSSIANS=5M (isolate Gaussian count effect) | — | — | — | — | — | — | MAX_N_GAUSSIANS: 5000000 |

## Analysis Notes

### Phase 1 Root Cause (Discovered via CUDA code analysis)
- The `effective_threshold = grad_threshold × render_scale²` in `dash_density_control_topk` was WRONG
- CUDA backward (kernels_backward.cuh): stored gradient = 0.5×|[dL_pixel.x×W/r, dL_pixel.y×H/r]|
- With L1 mean loss: dL_pixel ×r², footprint derivative ×r, affected pixels ×(1/r) → dL_pixel ×r²
  Combined with ×(W/r): stored NDC gradient scales as **r (linear)**, NOT r²
- With r² scaling at render_scale=7: threshold 49× higher but gradients only 7× higher
  → virtually zero Gaussians pass threshold → momentum_add≈0 → P_fin stuck → 2.15M Gaussians

### E2-A Findings
- Removing r² scaling: **+0.85 dB PSNR** (24.27 → 25.12), Gaussians 2.15M → 3.32M
- Still -0.18 dB below FasterGS (25.30). Gaussian gap: 3.32M vs 4.8M (69%)
- VRAM lower than FasterGS (3.83 vs 5.34 GiB) due to fewer Gaussians
- Speed still ~4:30 vs 7:46 FasterGS ≈ 1.7× faster

### Phase 1 Other Findings
- SH unlock threshold change (scale<2 → scale<4) only improved by +0.09 dB
