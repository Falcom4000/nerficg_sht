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
| E2-B | E2-A + LR decay from lr_decay_from_iter (Conflict J) | ~4:30 | **25.22** | **0.773** | **0.216** | **5,094,230** | 5.87/7.92 GiB | Trainer.py LR offset |
| E2-C | E2-A+E2-B + INITIAL_MOMENTUM_FACTOR 15→5 (DashGaussian default) | ~5:44 | 25.18 | 0.772 | 0.217 | 5,037,397 | 5.80/7.70 GiB | INITIAL_MOMENTUM_FACTOR: 5 |

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

### E2-B Findings (E2-A + LR decay fix, factor=15)
- LR fix adds another +0.10 dB PSNR, +0.025 SSIM, **-0.068 LPIPS** (huge LPIPS improvement!)
- Gaussians: 3.32M → **5.09M** (now exceeds FasterGS 4.8M by 6%)
- SSIM **0.773 > FasterGS 0.767** ✓, LPIPS **0.216 < FasterGS 0.231** ✓
- Only PSNR still slightly below: 25.22 vs 25.30 (-0.08 dB)

### E2-C Findings (same fixes, factor reverted to 5)
- factor=5 vs factor=15: PSNR 25.18 vs 25.22, SSIM 0.772 vs 0.773, LPIPS 0.217 vs 0.216
- **Essentially identical** — INITIAL_MOMENTUM_FACTOR doesn't matter once bugs are fixed
- factor=5 (DashGaussian original default) is sufficient; keeping it as the config default
- **Final config**: no r² threshold scaling + LR decay from lr_decay_from_iter() + factor=5

### Phase 1 Other Findings
- SH unlock threshold change (scale<2 → scale<4) only improved by +0.09 dB
