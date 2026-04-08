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
| E2-1 | Higher momentum factor (factor=15) | — | — | — | — | — | — | INITIAL_MOMENTUM_FACTOR: 5→15 |
| E2-2 | Very high momentum factor (factor=40) | — | — | — | — | — | — | INITIAL_MOMENTUM_FACTOR: 5→40 |
| E2-3 | Fixed Gaussian cap (MAX_N=5M) | — | — | — | — | — | — | MAX_N_GAUSSIANS: -1→5000000 |
| E2-4 | LR decay from lr_decay_from_iter() (Conflict J fix) | — | — | — | — | — | — | Trainer.py LR offset |
| E2-5 | Combined: best count fix + LR fix | — | — | — | — | — | — | TBD after E2-1..E2-3 |

## Analysis Notes

### Phase 1 Findings
- Speed: 2.75× faster than FasterGS non-fused (2:50 vs 7:46)
- Quality gap: -1.03 dB PSNR (24.27 vs 25.30)
- Root cause hypothesis: P_fin underestimation (Conflict L)
  - Only 2.15M Gaussians vs 4.8M (45% of FasterGS)
  - DashGaussian momentum γ=0.98 calibrated for standard 3DGS convergence speed
  - FasterGS converges faster → P_add drops earlier → momentum saturates too low
- SH unlock threshold change (scale<2 → scale<4) only improved by +0.09 dB
  → SH contamination is NOT the primary cause
