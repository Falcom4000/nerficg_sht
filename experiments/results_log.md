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

## Phase 4: Indoor Scene Hyperparameter Tuning (E4, bonsai)

### 背景：室内场景 gap 的本质原因

FFT 能量分析对室内场景（bonsai、counter）计算出 `max_reso_scale≈6–7`，对室外场景（bicycle、garden）计算出 `max_reso_scale≈4–5`。这一差异直接决定了训练早期的分辨率下采样倍数，进而影响 densify_rate 预算和模型形成质量。

`densify_rate = (max_n − init_n) / scale^(2 − iter/densify_until_iter)`

scale=7 时分母约为 49，scale=4 时约为 16，早期预算相差 3×。

### E4 实验结果（bonsai，FasterGS 基准 32.84）

| ID | 变量 | max_reso_scale（实际） | render_scale₀ | PSNR | #Gaussians |
|----|------|----------------------|--------------|------|------------|
| E3-A（基准）| 默认配置 | ~6.3 | 6 | 32.49 | 983K |
| E4-A | `densify_mode=free`（关闭 top-k 限速）| 6.3 | 6 | 32.37 | 1,078K |
| E4-B | `MAX_RESO_SCALE=4` | 4.0 | 3 | 32.61 | 991K |
| E4-C | `START_SIGNIFICANCE_FACTOR=2.0` | 3.0 | 3 | **32.67 ± 0.16**（4次均值）| 1,020K |
| E4-D | `MAX_RESO_SCALE=4` + `START_SIGNIFICANCE_FACTOR=2.0` | 3.0 | 3 | ≈E4-C（schedule 完全相同）| 1,029K |

E4-C 的 4 次原始数据：32.72 / 32.43 / 32.86 / 32.67（std dev ≈ 0.16 dB）。

### 关键发现

**1. top-k 预算限制不是室内 gap 的主因（E4-A）**
关闭 top-k（`densify_mode=free`）后 Gaussian 数量超过 FasterGS（1.08M vs 1.06M），但 PSNR 反而下降到 32.37。原因：低分辨率阶段（scale=6-7）梯度被放大 6-7×，无 top-k 保护时早期浪费 budget 在低质量 Gaussian 上。**top-k 是对的，scale 才是问题。**

**2. 限制 max_reso_scale 有效（E4-B vs E4-C）**
- `MAX_RESO_SCALE=4`（E4-B）：强制 FFT 结果 ≤ 4，bonsai 从 scale=6.3 降到 4.0，PSNR +0.12 dB
- `START_SIGNIFICANCE_FACTOR=2.0`（E4-C）：放宽能量阈值（25%→50%），bonsai 自动算出 max_reso_scale=3.0，PSNR 均值 +0.18 dB，效果更好

**3. 两个参数组合不产生额外收益（E4-D）**
当 `START_SIGNIFICANCE_FACTOR=2.0` 算出的 max_reso_scale=3.0 已经低于 `MAX_RESO_SCALE=4` 的 cap 时，cap 不再生效，E4-D 与 E4-C 产生完全相同的 schedule。对于室内场景，`START_SIGNIFICANCE_FACTOR=2.0` 已经足够激进，无需再叠加 cap。

**4. 方差问题**
bonsai 训练存在约 ±0.16 dB 的 run-to-run 方差（GPU atomic 操作非确定性），单次结果不可靠。结论基于 4 次重复均值。

### 室内/室外场景推荐超参

| 参数 | 室外场景（bicycle、garden）| 室内场景（bonsai、counter、kitchen、room） |
|------|--------------------------|------------------------------------------|
| `DENSIFICATION_END_ITERATION` | 14900 | 14900 |
| `MORTON_ORDERING_END_ITERATION` | 15000 | 15000 |
| `DENSIFY_MODE` | `freq` | `freq` |
| `RESOLUTION_MODE` | `freq` | `freq` |
| `START_SIGNIFICANCE_FACTOR` | 4.0（默认）| **2.0** |
| `MAX_RESO_SCALE` | 8（默认）| 8（让 sig_factor 自然限制）|
| `MAX_N_GAUSSIANS` | -1（momentum 自适应）| -1 |
| `INITIAL_MOMENTUM_FACTOR` | 5 | 5 |

**判断室内/室外的实用依据：** 训练开始时 log 打印 `max_scale=X`。X > 5 → 室内场景，考虑使用 `START_SIGNIFICANCE_FACTOR=2.0`。

MipNeRF360 各场景实测 max_reso_scale（单张图估算，仅供参考）：

| Scene | factor=4（默认）| factor=2 | render_scale₀（factor=2）| 推荐 factor |
|-------|----------------|----------|--------------------------|------------|
| bonsai | 8.0 | 3.2 | 3 | **2.0** |
| counter | ~8.0 | ~3.0 | ~3 | **2.0** |
| kitchen | 8.0 | 3.2 | 3 | **2.0** |
| room | 8.0 | 3.2 | 3 | **2.0** |
| stump | 6.3 | 2.9 | 2 | 默认（待测）|
| garden | 4.8 | 2.4 | 2 | 默认 |
| bicycle | ~4–5 | ~2–3 | 2–3 | 默认 |

stump 用 factor=2 会使 render_scale₀=2（几乎从全分辨率开始），速度收益大幅减少，未验证质量影响，暂保守处理。

### 调整后的最终质量（bonsai，4次均值 vs 单次）

| 方法 | PSNR | vs FasterGS |
|------|------|-------------|
| FasterGS | 32.84 | — |
| FasterGSDash E3-A（默认）| 32.49（单次）| -0.35 |
| FasterGSDash E4-C（sig_factor=2）| **32.67 ± 0.16**（4次均值）| **-0.17** |

室内场景 gap 从 -0.35 dB 缩小到 -0.17 dB，已接近 FasterGS 单次结果的方差范围内。

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
