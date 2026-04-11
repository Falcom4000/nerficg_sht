# FasterGSFusedDash 改动记录与实验结果

Date: 2026-04-09

---

## 基线方法

| 方法 | 说明 |
|------|------|
| **FasterGSFused** | FasterGS 论文的 fused-Adam 变体，无 DashGaussian 调度 |
| **FasterGSDash** | FasterGS + DashGaussian 调度，使用 PyTorch Adam（非 fused） |
| **FasterGSFusedDash** | FasterGSFused + DashGaussian 调度（本项目目标：最快+最优） |

---

## 版本 V0：初始实现（改动前基线）

首次将 DashGaussian 的 FFT 分辨率调度 + top-k 密化预算融入 FasterGSFused 的 fused-Adam 管线。已处理的冲突：

| 冲突 | 解决方案 |
|------|---------|
| A — 梯度阈值在低分辨率失效 | `effective_threshold = grad_threshold`（无缩放） |
| B — Z-ordering 在少量 Gaussian 时拖速 | `MIN_GAUSSIANS=50000` 门控 |
| G — render_scale 未传入 CUDA | `width // scale`, `focal / scale` |
| H — invisible momentum 低分辨率漂移 | `apply_invisible_momentum=(render_scale == 1)` |
| J — LR 衰减从 iter=1 starving 低分辨率 | `lr_decay_from_iter()` 延迟 |
| Gap 1 — GT 降采样 | `F.interpolate(mode='area')` |
| Gap 2 — SH 低分辨率解锁 | `near_full_resolution()` 门控 |

### V0 结果（3-run avg）

| 场景 | FasterGSDash | FusedDash V0 | Gap | 加速比 |
|------|-------------|-------------|-----|--------|
| bonsai | 32.59 / 196s | 32.60 / 189s | +0.01 | 1.04× |
| garden | 27.52 / 415s | 27.31 / 382s | **-0.21** | 1.09× |
| bicycle | 25.15 / 437s | 25.11 / 383s | -0.04 | 1.14× |

---

## 版本 V1：Cold-primitive bias correction + 训练循环优化

Commit: `6972116`

### 改动内容

**1. Cold-primitive bias correction（CUDA）**

FasterGSFused 的 fused Adam 用全局 `adam_step_count`（= iteration）做 bias correction。新 Gaussian（moments=0）在 iter=20000 时首步更新幅度 = 3.16 × lr。而 PyTorch Adam（FasterGSDash 所用）为每个参数维护独立 step count，首步幅度 = 1 × lr。

修复：在 `preprocess_backward_cu` 中检测 cold primitive（`moments_means` 全零），自动切换到 t=1 的 bias correction。不修改 `adam_step_helper` 签名，只覆盖局部变量 `eff_bc1_rcp` / `eff_bc2_sqrt_rcp` / `eff_step_size_means`。

文件：`kernels_backward.cuh`（DASH 本地 CUDA 后端）

**2. 每步更新 render_scale**

原来 render_scale 仅在密化回调（每 100 iter）时更新，导致阶梯式分辨率变化。改为每个 training iteration 调用 `get_res_scale()`。

文件：`Trainer.py`

**3. Morton ordering 自适应触发**

原来固定每 5000 步触发。改为每 100 步检查，当 Gaussian 数量增长 ≥ 20% 时才触发，匹配 DashGaussian 的凹形增长曲线。

文件：`Trainer.py`

**4. Renderer 导入切换**

将 Renderer.py 的 CUDA 后端导入从 BASE FasterGSFused 切换到本地 DASH 副本，并在本地副本中恢复了 `apply_invisible_momentum` 条件控制。

文件：`Renderer.py` + 5 个 CUDA/Python 绑定文件

### V1 结果（3-run avg）

| 场景 | FasterGSDash | FusedDash V0 | FusedDash V1 | V0→V1 变化 | vs GSDash |
|------|-------------|-------------|-------------|-----------|-----------|
| bonsai | 32.59 / 196s | 32.60 / 189s | **32.70** / 189s | **+0.10** | +0.11 |
| garden | 27.52 / 415s | 27.31 / 382s | **27.45** / 383s | **+0.14** | -0.07 |
| bicycle | 25.15 / 437s | 25.11 / 383s | 25.10 / **376s** | ±0 / **-7s** | -0.05 |
| counter | — | — | 29.40 / 200s | — | — |
| kitchen | 32.10 / 281s | — | 31.84 / 274s | — | -0.26 |
| room | 32.44 / 190s | — | 32.24 / 176s | — | -0.20 |
| stump | 26.69 / 369s | — | 25.78 / 288s | — | **-0.91** |

**核心收益**：garden 差距从 -0.21 → -0.07 dB（+0.14），bonsai +0.10 dB，bicycle 速度 383→376s。

**暴露问题**：kitchen/room 有 -0.2~0.3 dB gap，stump -0.91 dB 严重异常（run 3 仅 24.98 dB，疑似训练不稳定）。

---

## 版本 V2：对齐 DashGaussian 原版 + 通用优化

Commit: `4621f17`

### 改动内容

**1. 先剪枝再密化 + 后剪枝 budget（对齐 DashGaussian）**

V0/V1 顺序：top-k → clone/split → prune（opacity + degenerate + oversized）
DashGaussian 原版顺序：prune（opacity + degenerate） → 计算 budget → top-k → clone/split → 移除 split 父节点

问题：低 opacity Gaussian 往往梯度大（优化器在"拼命救"），但注定被剪，浪费 top-k 名额。

同时将 budget 公式从 `N × rate` 改为 `min(target_N - post_prune_N, post_prune_N)`（剪枝越多 → 增长空间越大）。

文件：`Model.py` — `dash_density_control_topk()`

**2. Opacity reset 不清零 moments（对齐 DashGaussian）**

V0/V1：`reset_opacities()` 同时清零 `moments_opacities`。配合全局 `adam_step_count`，reset 后全部 Gaussian 的 opacity 首步更新 3.16× 放大。

DashGaussian 的 PyTorch Adam 保留 optimizer state，不清零。改为匹配此行为。

文件：`Model.py` — `reset_opacities()`

**3. SH unlock 去掉 resolution gate（对齐 DashGaussian）**

V0/V1 的 Gap 2 fix：`near_full_resolution(threshold=4.0)` 门控 SH 解锁。但 DashGaussian 原版每 1000 iter 无条件解锁，且经验验证有效。该 gate 可能过于保守，推迟了 view-dependent 颜色学习。

文件：`Trainer.py` — `increase_sh_degree()`

**4. 密化起始 600 → 500（对齐 DashGaussian）**

文件：`Trainer.py` — config `DENSIFICATION_START_ITERATION`

**5. Invisible Gaussian moment 衰减（CUDA 通用优化）**

V0/V1：`render_scale > 1` 时完全冻结 invisible Gaussian 的 moments。切回全分辨率时，陈旧 momentum（可能冻结了 15000+ 步）仍以原方向作用。

改为：每步衰减 moments（`moments *= (beta1, beta2)`）但不更新参数。`0.9^15000 ≈ 0`，陈旧 momentum 自然消失。新增 `decay_invisible_moments` CUDA kernel。

文件：`kernels_backward.cuh` + `backward.cu`

**6. GT 多分辨率预缓存**

消除每步 `F.interpolate` 开销（~0.5ms/iter × ~20k 低分辨率步 ≈ 10s）。在 `setup_gaussians` 时预计算所有 (view, scale) 组合的降采样 GT，训练时直接查缓存。

文件：`Trainer.py`

### V2 初步结果（跑崩，触发 V2.1 修复）

bonsai run 1: 32.50 / **301s** / 1,317k Gaussians（V1: 189s / 1,120k）— 训练时间 +60%，Gaussian 爆炸增长。

**根因**：改动 2（opacity moments 不清零）+ 改动 1（budget = target_N - post_prune_N）联合导致：
- Opacity reset 后 momentum 把 opacity 推回高值 → reset 形同虚设 → 剪枝剪不掉 Gaussian
- Budget 公式在剪枝少时仍给大预算 → Gaussian 数量爆炸

---

## 版本 V2.1：修复 V2 爆炸增长

Commit: `d861c1d`

### 改动内容

**回退改动 2**：恢复 `moments_opacities.zero_()`。Fused Adam 没有 PyTorch Adam 的 per-parameter state 替换机制，保留 moments 导致 reset 无效化。

**修正改动 1 的 budget 公式**：从 DashGaussian 的 `min(target_N - post_prune_N, post_prune_N)` 改为 `post_prune_N × rate`。保留 prune-first 顺序的优势（不浪费 top-k 名额），但 budget 基于实际存活数量，避免过度增长。

### V2.1 = V1 + 以下有效改动

| V2 改动 | V2.1 状态 |
|---------|----------|
| 1. 先剪枝再密化 | **保留**，budget 公式改为 `post_prune_n × rate` |
| 2. Opacity moments 不清零 | **回退**（恢复清零） |
| 3. SH unlock 去掉 gate | 保留 |
| 4. 密化起始 600→500 | 保留 |
| 5. Invisible moment decay (CUDA) | 保留 |
| 6. GT 多分辨率预缓存 | 保留 |

### V2.1 结果（6-run avg，ccc033d）

| 场景 | FasterGSFused | FasterGSDash | V2.1 | vs GSDash |
|------|-------------|-------------|------|-----------|
| bonsai | 32.73 / 195s | 32.59 | 32.07 / 187s | -0.52 |
| counter | 29.44 / 208s | — | 29.33 / 196s | — |
| kitchen | 32.28 / 270s | 32.10 | 31.93 / 273s | -0.17 |
| room | 32.40 / 194s | 32.44 | 32.07 / 169s | -0.37 |
| bicycle | 25.28 / 475s | 25.15 | 24.92 / 326s | -0.23 |
| garden | 27.50 / 484s | 27.52 | 27.35 / 353s | -0.17 |
| stump | 26.68 / 396s | 26.69 | 25.47 / 253s | -1.22 |

V2.1 全面差于 V1，所有方向均退步。outdoor 场景 Gaussian 数量骤降（bicycle 4073k→3163k），根因为 `post_prune_n × rate` budget 公式在 prune-first 下天然偏小。

**结论**：V2 全部改动废弃，回退到 V1 基线。

---

## 版本 V3：V1 + 移除 SH resolution gate

Commit: `2548fea`

### 改动内容

仅移除 `increase_sh_degree` 回调中的 `near_full_resolution()` 检查，匹配 DashGaussian 原版（每 1000 iter 无条件解锁）。

文件：`Trainer.py`

### V3 结果（3-run avg，2548fea）

| 场景 | FasterGSDash | V1 | V3 | V1→V3 | V3 vs GSDash |
|------|-------------|----|----|-------|--------------|
| bonsai | 32.59 | 32.70 | 32.57 | **-0.13** | -0.02 |
| counter | — | 29.40 | 29.48 | **+0.08** | — |
| kitchen | 32.10 | 31.84 | 31.80 | -0.05 | -0.30 |
| room | 32.44 | 32.24 | 32.30 | +0.06 | -0.14 |
| bicycle | 25.15 | 25.10 | 25.09 | -0.01 | -0.06 |
| garden | 27.52 | 27.45 | 27.47 | +0.02 | -0.05 |
| stump | 26.69 | 25.78 | 26.33 | **+0.55** | -0.36 |

**净收益：+0.53 dB 跨场景之和。** stump 从 -0.91 收窄到 -0.36 vs GSDash。
bonsai 小幅退步 (-0.13)，可接受（V3 vs GSDash 仍为 -0.02，基本持平）。

SH gate 移除对 outdoor 场景帮助大（view-dependent 效果更丰富，早期解锁有收益），
对 indoor 场景略有负面（低分辨率早期 SH 可能引入噪声）。

**V3 确立为新基线。**

---

## 版本 V4：prune-first + DashGaussian budget 公式

Commit: `02a1fef`

### 改动内容

V4 = V3 + prune-first 密化顺序 + DashGaussian 原始 budget 公式。

V2 尝试过相同方向但失败，根因是 opacity moments 未清零导致密化爆炸。
现在 opacity moments 已恢复清零，公式可以安全使用：

```python
n_budget = min(int(cur_n * (1 + densify_rate) - post_prune_n), post_prune_n)
```

此公式补偿了被剪掉的 Gaussian：若本轮剪了 5k，budget 相应增加 5k，
使总数量稳定跟踪调度器目标。prune-first 确保 top-k 名额不浪费在注定被剪的 Gaussian 上。

文件：`Model.py` — `dash_density_control_topk()`

### V4 结果（3-run avg，02a1fef）

| 场景 | GSDash | V1 | V3 | V4 | V3→V4 | V4 vs GSDash |
|------|--------|----|----|-----|-------|--------------|
| bonsai | 32.59 | 32.70 | 32.57 | **32.69** | +0.11 | **+0.10** |
| counter | — | 29.40 | 29.48 | 29.47 | -0.01 | — |
| kitchen | 32.10 | 31.84 | 31.80 | **32.12** | **+0.32** | **+0.02** |
| room | 32.44 | 32.24 | 32.30 | 32.32 | +0.02 | -0.12 |
| bicycle | 25.15 | 25.10 | 25.09 | 25.09 | +0.00 | -0.06 |
| garden | 27.52 | 27.45 | 27.47 | 27.38 | -0.10 | -0.14 |
| stump | 26.69 | 25.78 | 26.33 | 26.33 | +0.00 | -0.36 |

训练时间和 Gaussian 数量与 V3 几乎相同（无爆炸增长）。

**核心收益**：kitchen +0.28 vs V1（现与 GSDash 持平），bonsai 恢复 +0.10 vs GSDash。
**唯一回退**：garden -0.07 vs V1（prune-first 对 outdoor 场景轻微负面）。
**V4 确立为新基线。**

### V1→V4 累计改进总结

| 场景 | V1 | V4 | 改进 |
|------|----|----|------|
| bonsai | 32.70 | 32.69 | -0.01 |
| kitchen | 31.84 | 32.12 | **+0.28** |
| room | 32.24 | 32.32 | +0.08 |
| bicycle | 25.10 | 25.09 | -0.01 |
| garden | 27.45 | 27.38 | -0.07 |
| stump | 25.78 | 26.33 | **+0.55** |

---

## 已知冲突与设计决策汇总

| 冲突 | 状态 | 版本 | 备注 |
|------|------|------|------|
| A — 梯度阈值 | SOLVED | V0 | 不缩放，top-k 预算兜底 |
| B — Z-ordering 少量 Gaussian | SOLVED | V0→V1 改进 | MIN_GAUSSIANS + 自适应 20% 增长触发 |
| C — MCMC 不兼容 | N/A | V0 | USE_MCMC=False |
| D — Fused Adam bias correction | SOLVED | V1 | Cold-detection in CUDA |
| E — 梯度 reset 时机 | NOT A CONFLICT | V0 | 实测一致 |
| F — 梯度归一化 | NOT A CONFLICT | V0 | 数学等价 |
| G — render_scale 传入 CUDA | SOLVED | V0 | 缩放 width/height/focal/center |
| H — Invisible momentum 漂移 | SOLVED→IMPROVED | V0→V2.1 | V0: 冻结; V2.1: 衰减 |
| J — LR 衰减 starving | SOLVED | V0 | `lr_decay_from_iter()` |
| Gap 1 — GT 降采样 | SOLVED→IMPROVED | V0→V2.1 | V0: 每步 interpolate; V2.1: 预缓存 |
| Gap 2 — SH 低分辨率解锁 | REVERTED | V0→V2.1 | V0: gate; V2.1: 无 gate（匹配 DashGS） |
| Prune 顺序 | ALIGNED | V2.1 | 先剪枝再密化 |
| Budget 公式 | MODIFIED | V2→V2.1 | V2: target-post_prune（爆炸）; V2.1: post_prune×rate |
| Opacity moments reset | REVERTED | V2→V2.1 | V2: 不清零（爆炸）; V2.1: 恢复清零 |
| 密化起始 iter | ALIGNED | V2.1 | 600→500 |

---

## 版本 V6：invisible moment decay + 分辨率跳变 moment 重缩放

### 改动内容

**1. Invisible moment decay（CUDA，#13）**

`apply_invisible_momentum=False`（低分辨率阶段）的语义从"完全冻结"改为"只衰减 moments"。

原来（V0–V5）：低分辨率下 invisible Gaussian 的 moments 完全冻结，切回全分辨率时
陈旧 momentum 仍保留原方向和幅值，导致方向性偏差。

现在：每步对 invisible Gaussian 执行 `m1 *= beta1; m2 *= beta2`，无参数更新。
`0.9^N → 0` 保证陈旧 momentum 自然消散，切换分辨率时 new high-res gradients 可快速主导。

文件：`FasterGSFusedCudaBackend/rasterization/include/kernels_backward.cuh`（新增 `decay_moments_invisible` kernel）
      `FasterGSFusedCudaBackend/rasterization/src/backward.cu`（新增 else 分支）

注：Python API 无变化，`apply_invisible_momentum=(render_scale == 1)` 语义不变，
仅 False 分支行为从"do nothing"改为"decay only"。

**2. 分辨率跳变时 Adam moment 重缩放（Python，#4）**

当 `render_scale` 下降（分辨率提升）时，对全部参数组的 moments 做等比缩放：
- `m1 *= ratio`（ratio = old_scale / new_scale）
- `m2 *= ratio²`

理论依据：低分辨率梯度幅值约为高分辨率的 1/ratio。不做重缩放时，
m2 在分辨率跳变后短期内相对 m1 偏小（m2 是梯度²，恢复更慢），
导致 Adam effective step size 短暂偏高。重缩放使两者同步跟上新梯度量级。

注：这是对各参数类型梯度缩放关系的近似处理，实际效果以 benchmark 验证为准。

文件：`Trainer.py`（新增 `_rescale_moments_on_resolution_change` + 在 `training_iteration` 中调用）

### 为什么这两项一起做

- #13 解决"方向性"问题：冻结的 momentum 携带错误方向
- #4 解决"幅值性"问题：跳变后 m1/m2 比例失衡
- 两者正交，无相互依赖，同时引入不影响隔离分析
- 均为已知问题的最小改动，无副作用（#13 纯 CUDA 新增路径，#4 纯 Python 3 行）

### 预期结果

- garden/stump 等 Gaussian 快速增长的场景（低分辨率期长、invisible Gaussian 多）改善最明显
- 若 #13 单独贡献可观，#4 贡献有限（Adam 本身对幅值鲁棒），后续可隔离验证

