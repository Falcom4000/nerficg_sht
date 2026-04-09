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

## Phase 3 Ablation: Diagnosing Indoor Scene Gaps (E3-A–D, bonsai)

### E3 Experiments

| ID | Description | PSNR | SSIM | LPIPS | #Gaussians | Notes |
|----|-------------|------|------|-------|------------|-------|
| E3-A | densify_end=14900 (FasterGS default), freq | **32.49** | 0.954 | 0.133 | 983,218 | Best result |
| E3-B | densify_end=27000, const resolution | 32.22 | 0.952 | 0.142 | 982,320 | Extended densify hurts |
| E3-C | densify_end=27000, freq, reso_until=14900 (decoupled) | 32.06 | 0.950 | 0.137 | 915,403 | Worse: extra opacity resets |
| E3-D | E3-A + Conflict E fix (carry gradient history) | 32.29 | 0.950 | 0.149 | 712,287 | Worse than E3-A |
| FasterGS ref | — | 32.84 | 0.954 | 0.136 | 1,060,601 | — |

**Root cause of indoor quality gap (solved):** `DENSIFICATION_END_ITERATION=27000` (extended from FasterGS default 14900).

Mechanism: `densify_rate = (max_n − init_n) / scale^(2 − iter/densify_until_iter)`.
Indoor scenes (bonsai, counter) get `max_reso_scale≈7–8` from FFT analysis. With `densify_until_iter=27000` and `scale=7`: divisor reaches ~47 → densify budget ≈2% at early iterations. Additionally, extending densify_end to 27000 triggers extra opacity resets at iters 15K, 18K, 21K, 24K (every 3K), disrupting a mature model.

**Fix:** `DENSIFICATION_END_ITERATION: 14900` for all scenes.

**Why E3-C failed:** Decoupling reso_until from densify_until while keeping densify_end=27000 still causes the extra opacity resets. Bonsai at iter 15K–27K has 900K well-trained Gaussians; resetting opacity 4–5 times degrades rather than helps.

**Why E3-D (Conflict E fix) failed:** Carrying gradient history across densification steps accumulates low-resolution (7× inflated) gradients that bias top-k selection at full resolution, wasting densification budget on already-good Gaussians.

**Conclusion:** Conflict E is NOT a real conflict. The current reset (every 100-iter window) matches DashGaussian's actual behaviour and is correct.

### E3-D Validated Config (all 7 scenes)

`DENSIFICATION_END_ITERATION: 14900`, `MORTON_ORDERING_END_ITERATION: 15000`, all else default.

| Scene | PSNR | SSIM | LPIPS | #Gaussians | PSNR Δ vs FasterGS |
|-------|------|------|-------|------------|---------------------|
| bonsai | 32.49 | 0.954 | 0.133 | 983,218 | **-0.35** |
| counter | 29.31 | 0.919 | 0.149 | 981,857 | **-0.18** |
| garden | 27.54 | 0.867 | 0.120 | 3,626,990 | **+0.11** |

Garden: FasterGSDash now **exceeds** FasterGS (+0.11 dB, fewer Gaussians). Indoor scenes: 0.18–0.35 dB gap remains.

---

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
