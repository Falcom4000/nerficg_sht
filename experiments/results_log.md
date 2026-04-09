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

## Multi-Scene Validation (Phase 3)

Scenes: bonsai, garden, counter (IMAGE_SCALE_FACTOR=0.25). bicycle results included for reference.

### FasterGS Baseline

| Scene | Time | PSNR | SSIM | LPIPS | #Gaussians |
|-------|------|------|------|-------|------------|
| bicycle | 7:46 | 25.30 | 0.767 | 0.231 | 4,807,821 |
| bonsai | 2:50 | 32.84 | 0.954 | 0.136 | 1,060,601 |
| garden | 7:54 | 27.43 | 0.866 | 0.121 | 4,129,558 |
| counter | 3:01 | 29.49 | 0.921 | 0.149 | 990,482 |

### FasterGSDash (Phase 2 fixes: no r² scaling + LR decay fix, factor=5)

| Scene | Time | PSNR | SSIM | LPIPS | #Gaussians | PSNR Δ | Speed |
|-------|------|------|------|-------|------------|--------|-------|
| bicycle | 5:44 | 25.18 | 0.772 | 0.217 | 5,037,397 | -0.12 | 1.35× |
| bonsai | 2:18 | 31.87 | 0.950 | 0.138 | 904,761 | -0.97 | 1.23× |
| garden | 5:21 | 27.39 | 0.864 | 0.122 | 3,797,280 | -0.04 | 1.48× |
| counter | 2:16 | 28.97 | 0.913 | 0.159 | 964,085 | -0.52 | 1.33× |

### Phase 3 Analysis

**Outdoor scenes** (bicycle, garden): near-parity with FasterGS.
- garden: -0.04 dB (essentially identical quality, 1.48× faster)
- bicycle: -0.12 dB (minimal gap, 1.35× faster)

**Indoor scenes** (bonsai, counter): larger quality gaps.
- counter: -0.52 dB
- bonsai: -0.97 dB

Likely cause: indoor scenes have smaller spatial extent, so the FFT resolution schedule
allocates less time at full resolution than outdoor scenes need to form fine details.
Indoor scenes converge more in the full-res phase, which FasterGSDash shortens.
The Gaussian count gap is also more pronounced: bonsai 905K vs 1.06M (85%), counter 964K vs 990K (97%).

Speed improvement is consistent: **1.23×–1.48× faster** across all scenes.

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
