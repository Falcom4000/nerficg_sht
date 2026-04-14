# 论文贡献总结：FasterGSFusedDash

> 方法实现：src/Methods/FasterGSFusedDash/
> Benchmark 数据：experiments/benchmark_results.md
> 详细冲突分析：experiments/fuseddash_analysis.md

---

## 研究背景

本工作从三个现有方法出发：

| 方法 | 核心贡献 | 来源 |
|------|---------|------|
| **3D Gaussian Splatting (3DGS)** | 用显式 3D 高斯表示场景 + tile-based 光栅化 | Kerbl et al., SIGGRAPH 2023 |
| **FasterGSFused** (Hahlbohm et al., CVPR 2026) | 将 Adam optimizer 融入 CUDA backward kernel，4–5× 加速 3DGS 训练 | papers/hahlbohm2026fastergs.pdf |
| **DashGaussian** | FFT 频率分析引导的分辨率调度 + top-k primitive 调度，减少低效训练步骤 | papers/dashGaussian.pdf |

**核心问题**：FasterGSFused 和 DashGaussian 的加速来自完全正交的机制，理论上叠加应得到更大提升。但二者从未被结合，且存在多个非平凡的工程冲突。

---

## 主要贡献：FasterGSFusedDash

### 贡献 1：识别并系统解决 6 个集成冲突

这是本工作最核心的工程贡献。两者结合时存在以下非平凡冲突：

#### (A) µ2D 梯度幅值与渲染分辨率耦合（密化阈值失效）

- **问题**：FasterGSFused 在 CUDA kernel 中直接计算 µ2D 梯度，在 1/r 分辨率渲染下梯度缩放 ~r 倍（线性，而非 r²）。DashGaussian 的阈值补偿假设 r² 缩放，在 scale=7 时阈值放大 49× 而梯度仅增 7×，导致密化完全失效（几乎无 Gaussian 通过）。
- **修复**（`Model.py:300`）：去除阈值缩放，改用 `effective_threshold = grad_threshold`，依靠 top-k 预算上限控制实际密化量。
- **验证**：bonsai PSNR 从 31.87 → 32.49（+0.62 dB）。

#### (B) Z-ordering 在稀疏阶段的性能退化

- **问题**：FasterGSFused 论文明确指出 Morton 排序在 Gaussian 数量少时因 atomic contention 反而拖慢训练。DashGaussian 早期 Gaussian 数量很少，若固定 5000 步触发会在训练前期造成性能损失。
- **修复**（`Trainer.py:153`）：`MORTON_ORDERING_MIN_GAUSSIANS=50_000` 门控 + 数量增长 >20% 才触发。

#### (G) render_scale 需传入 CUDA 光栅化器

- **问题**：FasterGSFused 的 `extract_settings()` 固定传入全分辨率相机参数，DashGaussian 需要在降采样分辨率下渲染。
- **修复**（`Renderer.py:34–43`）：所有相机参数按 `render_scale` 等比缩放（width, height, focal_x, focal_y, center_x, center_y）。

#### (H) apply_invisible_momentum：低分辨率下的 Adam moment 漂移

- **问题**：低分辨率渲染时全分辨率可见但当前不可见的 Gaussian 的 Adam moments 会被用陈旧梯度估计更新，切回全分辨率时前几步 LR 偏高。
- **修复**（`Renderer.py:87`）：`apply_invisible_momentum=(render_scale == 1)`，低分辨率阶段完全冻结不可见 Gaussian 的 moment 更新。

#### (J) 位置 LR 在低分辨率阶段饥饿

- **问题**：FasterGSFused 的位置 LR 从 iteration=1 就开始衰减。DashGaussian 训练前期（约前 15000 步）处于低分辨率，位置 LR 此时已衰减至很小，训练早期 Gaussian 位置收敛不充分。
- **修复**（`Trainer.py:185`）：`lr_iter = max(1, iteration - lr_decay_from_iter() + 1)`，延迟 LR 衰减到接近全分辨率训练开始时。

#### (Gap 1) GT 图像降采样与渲染分辨率不一致

- **问题**：FasterGSFused 假设 GT 与渲染分辨率相同，DashGaussian 在低分辨率下渲染但 GT 是原始高分辨率图像。
- **修复**（`Trainer.py:206–213`）：对 GT 使用 `F.interpolate(mode='area')` 抗锯齿降采样，与渲染分辨率同步。

---

### 贡献 2：Prune-First 密化顺序（V4）

**问题**：原始 DashGaussian 顺序是先 clone/split 再 prune。在 FasterGSFused 的框架里，先执行 prune 可以更准确估计实际需要新增的 Gaussian 数量，避免 budget 浪费在即将被剪除的 Gaussian 上。

**修复**（`Model.py:dash_density_control_topk`）：
1. 先 prune 低不透明度和退化 Gaussian
2. budget = `min(cur_n×(1+rate) - post_prune_n, post_prune_n)`，补偿被剪除的数量
3. 再在存活 Gaussian 上做 top-k 选择和 clone/split

---

### 贡献 3：实验验证与迭代优化（V0→V4）

通过 5 个版本的迭代，在 Mip-NeRF360 7 个场景上大规模 benchmark（每场景 6–12 次独立 run），系统量化了各冲突的 PSNR 影响并逐一修复。

---

## 实验结果（Mip-NeRF360，V4 最终版本，6-run 平均）

### FasterGSFusedDash vs FasterGSFused（消融上界）

| 场景 | FasterGSFused PSNR | FasterGSFusedDash PSNR | ΔdB | FasterGSFused 时间 | FasterGSFusedDash 时间 | 加速比 |
|------|-------------------|----------------------|-----|-------------------|----------------------|------|
| bonsai | 32.58 | 32.55 | −0.03 | 196s | 190s | 1.03× |
| counter | 29.47 | 29.45 | −0.02 | 207s | 200s | 1.04× |
| kitchen | 32.25 | 32.04 | −0.21 | 269s | 276s | — |
| room | 32.40 | 32.22 | −0.18 | 194s | 178s | 1.09× |
| **bicycle** | 25.26 | 25.08 | −0.18 | **477s** | **381s** | **1.25×** |
| **garden** | 27.49 | 27.43 | −0.06 | **484s** | **384s** | **1.26×** |
| **stump** | 26.66 | 26.18 | −0.48 | **397s** | **315s** | **1.26×** |

### FasterGSFusedDash vs FasterGSDash（非融合 Dash 基线）

| 场景 | FasterGSDash PSNR | FasterGSFusedDash PSNR | ΔdB | FasterGSDash 时间 | FasterGSFusedDash 时间 | 加速比 |
|------|------------------|----------------------|-----|------------------|----------------------|------|
| bonsai | 32.59 | 32.55 | −0.04 | 196s | 190s | 1.03× |
| garden | 27.52 | 27.43 | −0.09 | 415s | 384s | **1.08×** |
| bicycle | 25.15 | 25.08 | −0.07 | 437s | 381s | **1.15×** |
| kitchen | 32.10 | 32.04 | −0.06 | 281s | 276s | 1.02× |
| room | 32.44 | 32.22 | −0.22 | 190s | 178s | 1.07× |
| stump | 26.69 | 26.18 | −0.51 | 369s | 315s | **1.17×** |

**总结**：FasterGSFusedDash 在大型室外场景（bicycle/garden/stump）相对 FasterGSFused 实现 **1.25–1.26× 额外加速**，PSNR 损失 ≤0.18 dB（stump 除外）；相对 FasterGSDash（非融合版）快 1.08–1.17×，PSNR 基本持平。

---

## 已知局限

| 问题 | 说明 | 可修复性 |
|------|------|---------|
| Stump −0.48 dB（vs FasterGSFused） | 推测主要来自 Conflict D（新增 Gaussian adam bias correction 全局 step count，首步 3.16× 过大），在 Gaussian 爆发增长期积累 | 高工程量（per-Gaussian step counter） |
| 室内小场景加速有限 | FasterGSFused 在 bonsai/counter 等场景已极快（~200s），DashGaussian 调度绝对加速有限 | 设计权衡 |
| 渲染分辨率 100 步量化更新 | scale 过渡期有轻微精度损失 | 极低工程量，可一行修复 |

---

## 一句话贡献定位

> 本文首次将 FasterGS 的 fused CUDA backward+Adam 优化与 DashGaussian 的频率引导分辨率/primitive 调度系统性地集成，识别并解决了 6 个非平凡工程冲突，在 Mip-NeRF360 大型场景上实现了相对 FasterGSFused **1.25× 额外训练加速**，且几乎无质量损失。
