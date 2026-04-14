# FasterGSFusedDash 融合审查报告

Date: 2026-04-09

## 1. 已解决冲突（确认正确实现）

| 冲突 | 代码位置 | 解决方案 |
|------|---------|---------|
| A — 梯度阈值在低分辨率下失效 | `Model.py:287` | `effective_threshold = grad_threshold`（无缩放） |
| B — Z-ordering 在少量 Gaussian 时拖速 | `Trainer.py:154` | `MORTON_ORDERING_MIN_GAUSSIANS=50000` 门控 |
| G — render_scale 未传入 CUDA 渲染器 | `Renderer.py:34–43` | `width // render_scale`, `focal / render_scale` |
| H — invisible momentum 在低分辨率下漂移 | `backward.cu:129` | `if (apply_invisible_momentum)` 条件块 |
| J — LR 衰减从 iter=1 开始，starving 低分辨率阶段 | `Trainer.py:183` | `lr_decay_from_iter()` 延迟 |
| Gap 1 — GT 降采样与渲染分辨率不一致 | `Trainer.py:206–210` | `F.interpolate(mode='area')` 实时降采样 |
| Gap 2 — 高阶 SH 在低分辨率下被解锁 | `Trainer.py:123` | `near_full_resolution()` 门控解锁 |

---

## 2. 残留问题（已接受但有代价）

### 2.1 Conflict D：新增 Gaussian 的 Adam bias correction 失真

**位置**：`Trainer.py:182` `adam_step_count=optimization_step`

**机制**：FasterGSFused 使用全局 `optimization_step` 作为 Adam bias correction 的 step count。在 iter=20000 新增的 Gaussian（moments 初始化为 0，但 step=20001）：

```
bias_correction1_rcp  = 1/(1 - 0.9^20001)  ≈ 1.0   (本应补偿 cold-start)
bias_correction2_rcp  = 1/(1 - 0.999^20001) ≈ 1.0
```

实际首步更新幅度：

```
lr × (0.9 × 0 + 0.1 × g) / sqrt(0.999 × 0 + 0.001 × g²)
  = lr × 0.1g / 0.0316|g|
  = 3.16 × lr × sign(g)                  ← 正确值应为 1×
```

**影响**：约 3.16× 过大首步，100 步后自然收敛。DashGaussian 爆炸增长阶段（iter 15000–25000）持续有新 Gaussian 加入，累积效应是 garden/bicycle 场景与 FasterGSDash 产生 **~0.2 dB PSNR 差距**的最可能原因。

**可修复性**：需要在 CUDA 层为每个 Gaussian 维护独立的 step count，工程量较大。当前接受现状。

---

### 2.2 apply_invisible_momentum=False 的范围比预期更广

**位置**：`Renderer.py:87` `apply_invisible_momentum=(render_scale == 1)`

当 render_scale > 1 时，所有 `n_touched_tiles=0` 的 Gaussian 都被完全冻结，包含两类：

| 类型 | 应该冻结？ | 当前行为 |
|------|---------|---------|
| 全分辨率可见但低分辨率不可见的小 Gaussian | ✅ 是（避免漂移） | 正确冻结 |
| 真正在视锥外/相机背后的 Gaussian | ❌ 否（应该受 momentum 推动） | 也被冻结 |

**重要缓解因素**：FasterGSDash（PyTorch Adam 版本）对不可见 Gaussian 同样没有 momentum 补偿机制——PyTorch Adam 只在有梯度时更新，不执行 `adam_step_invisible` 等价操作。因此两者行为实质等价，不是相对 FasterGSDash 新引入的倒退。

---

### 2.3 渲染分辨率更新粒度（100 步量化）

**位置**：`Trainer.py:144` `current_render_scale = self.dash_scheduler.get_res_scale(iteration)`

DashGaussian 原始实现每步都调用 `get_res_scale()`，分辨率平滑过渡。当前实现仅在每次 densification（每 100 步）时更新，导致 scale 呈"阶梯形"变化。

在 scale 快速下降的过渡期（如 4 → 2），一个 100 步窗口内部分步使用了偏高的 scale，GT 采样不足。实际影响很小（100 步内 scale 变化幅度有限），但与原论文行为有偏差。

**可修复性**：低。将 `get_res_scale(iteration)` 移入 `training_iteration` 每步调用即可，纯 Python 改动，无需改 CUDA。

---

## 3. CUDA 层面优化机会

### 3.1 Tile buffer 动态重分配开销

**位置**：`rasterization_api.cu:51–54` `resize_function_wrapper`

每次 `forward()` 都通过 `resize_function_wrapper` 按需动态分配 tile/instance/bucket 四类 buffer。DashGaussian 的渲染分辨率单调递增（scale 从高到低），因此 buffer 大小在训练过程中单调增大——每次分辨率提升都触发真实的 CUDA 内存分配（PyTorch CUDACachingAllocator 可以复用旧内存，但不能保证）。

**改进方向**：在 `Trainer.__init__` 时以全分辨率尺寸预算最大 buffer，传入 CUDA 层一次性分配，后续所有 forward 复用。需修改 `rasterization_api.cu` 接口，工程量中等。

---

### 3.2 densification 中 12× torch.cat 的内存拷贝压力

**位置**：`Model.py:331–342`

每次密化需要 cat 6 个参数张量 + 6 个 moment 张量，每组都是 O(N) 内存拷贝：

```python
self._means.data = torch.cat([self._means, duplicated_means, split_means])   # ×12
```

在 N≈400 万时，单次密化涉及约 1 GB 数据搬运，产生 12 个中间 tensor，并触发 PyTorch 内存分配器压力。配合 `requires_empty_cache` 的 `torch.cuda.empty_cache()` 调用，还引入额外的 CPU-GPU 同步开销。

**改进方向（纯 Python）**：为参数和 moment 按 `max_n_gaussian` 预分配固定大小 buffer，用计数器追踪有效数量，densify 时只写入新位置（`copy_`）而不重新分配。CUDA 内核只处理前 N 个 primitive，支持此方案。

**改进方向（CUDA）**：将 mask 生成 + topk 选择 + split 计算 + buffer 写入融合成单个 CUDA kernel，减少多次顺序 kernel launch 和中间内存分配。

---

### 3.3 adam_step_invisible 的 6 个独立 kernel launch

**位置**：`backward.cu:137–215`

6 个参数组（means/scales/rotations/opacities/sh_0/sh_rest）分别 launch `adam_step_invisible`，每个都是轻量的 O(N) kernel。在 N 较小的训练早期，6× kernel launch latency 是主要开销而非计算本身。

**改进方向**：将 6 个 kernel 合并为单一 kernel，一次 launch 同时处理所有参数组（不同 stride 的 float2 数组需统一寻址）。在大 N 时改善有限，在小 N 早期有一定加速。

---

### 3.4 Morton ordering 触发策略与 Dash 增长曲线不匹配

**位置**：`Trainer.py:150–156`

当前策略：固定每 5000 步触发（有 `MIN_GAUSSIANS=50000` 门控）。

DashGaussian 的 Gaussian 增长曲线为"凹形"（concave-up）：
- 训练前期（~iter 0–15000）：增长缓慢，相邻两次 Z-ordering 之间 N 几乎不变，Z-ordering 效益低
- 训练后期（~iter 15000–25000）：爆发式增长，5000 步间隔可能导致 cache locality 快速退化而未及时触发 Z-ordering

**改进方向（纯 Python，无需改 CUDA）**：维护 `_last_morton_n`，当 `n_gaussians > _last_morton_n * 1.2` 时触发，使 Z-ordering 频率与实际增长速度挂钩。这是已验证的方向正确的改进。

---

## 4. 数值/实现细节问题

### 4.1 优先级排序导致 SH unlock 检查滞后

`increase_sh_degree`（priority=110）比 `densify`（priority=100）优先级更高（先执行）。因此在 iter 1000 时：

1. `increase_sh_degree` 先执行 → `near_full_resolution()` 读取的 `next_i` 来自 iter 900 的密化
2. `densify` 后执行 → 更新 `next_i` 到 iter 1000

SH unlock 检查比实际调度状态滞后约 100 步（偏保守），不是错误，但与预期语义有轻微偏差。

### 4.2 optimization_step 从 2 开始

`Trainer.py:182`：`optimization_step = iteration + 1`，iteration 从 1 开始，step 从 2 开始，跳过了 step=1 时最大的 bias correction 时刻（理论值 10×，实际只有 5.3×）。影响极小。

### 4.3 moments 张量维度的 CUDA 等价性

`moments_means` 形状为 `(N, 3, 2)`（Adam m1/m2 在最后维），CUDA 将其 `reinterpret_cast<float2*>` 当作 `N×3` 个 float2 处理，内存连续时等价。现有 prune/sort/cat 均在第 0 维操作且调用了 `.contiguous()`，没有问题。如果未来添加操作，需确保变换后仍内存连续。

---

## 5. 质量差距来源分析

当前 FasterGSFusedDash vs FasterGSDash 的质量对比（3 runs avg）：

| 场景 | FasterGSDash PSNR | FasterGSFusedDash PSNR | 差值 |
|------|------------------|----------------------|------|
| bonsai | 32.59 | 32.60 | +0.01 |
| garden | 27.52 | 27.31 | −0.21 |
| bicycle | 25.15 | 25.11 | −0.04 |

主要差距来源推测：

| 来源 | 影响量级 | 可修复性 |
|------|---------|---------|
| Conflict D：新 Gaussian 首步 3.16× 过大更新（garden 增长最爆炸） | 中等 | 高工程量（per-Gaussian step count） |
| 渲染分辨率 100 步量化（scale 变化期误差） | 小 | 低工程量（每步 query） |
| apply_invisible_momentum=False 冻结视锥外 Gaussian | 极小 | 不必要修 |

**最值得先尝试的单点改进**：将 `get_res_scale()` 移入 `training_iteration` 每步调用，消除 100 步量化误差，改动极小，风险极低。

---

## 6. 参考数据

### 训练时间对比（bonsai，iter=30000）

| 方法 | 时间 (s) | VRAM alloc (GiB) |
|------|---------|-----------------|
| FasterGSFused | ~196 | 1.14 |
| FasterGSDash | ~196 | 1.25 |
| FasterGSFusedDash | ~189 | 1.34 |

FasterGSFusedDash 相对 FasterGSDash 节省约 4% 训练时间（fused Adam 省去了 Python-side optimizer overhead）。相对 FasterGSFused 无速度损失，但 VRAM 略高（+0.20 GiB，来自额外的 moment 张量 + Dash 调度内存）。
