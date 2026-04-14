# FasterGSFused 相对 Vanilla 3DGS 的贡献总结

> 来源：Hahlbohm et al., "Faster-GS: Analyzing and Improving Gaussian Splatting Optimization", CVPR 2026
> 实现：src/Methods/FasterGSFused/

---

## 背景：3DGS 的训练瓶颈分析

原始 3DGS 在 RTX 4090 上训练 Mip-NeRF360 需要约 18–20 分钟，时间开销来自四个方面：

| 瓶颈 | 占比 | 原因 |
|------|------|------|
| Adam optimizer | ~40–60% | Python-side optimizer，每步 6 次独立 kernel，含 CPU-GPU 同步 |
| Backward pass | ~20–30% | 按 pixel 并行 → 大量 atomic 操作竞争 |
| Radix sort | ~10–15% | 64-bit key 全 bit 排序，复杂度高 |
| Cache miss | 分散在全程 | Gaussian 在内存中空间乱序，warp 访问不连续 |

FasterGSFused 用 6 项优化一次性解决上述所有瓶颈。

---

## 优化 1：Tight Opacity-aware Bounding Box（消除无效 tile 覆盖）

**3DGS 原版**：用保守正方形包围每个投影 Gaussian，基于协方差矩阵最大特征值估计半径，大量 tile 被纳入但 alpha 贡献接近零。

**FasterGSFused**：在 `kernels_forward.cuh` 的 `preprocess_cu` 中，精确计算椭圆在屏幕上的投影包围框（用 conic 矩阵的椭圆边界），以 `ushort4 screen_bounds` 存储，仅覆盖实际有贡献的 tile。

**效果**：直接减少 `n_instances`（tile × Gaussian 的实例总数），降低后续 sort 和 blend 的工作量。

---

## 优化 2：两阶段 Radix Sort（sort 复杂度降低 ~4×）

**3DGS 原版**：用 64-bit key（高 32 bit = tile index，低 32 bit = depth）做单次全位宽 radix sort，64-bit 需要 4× pass。

**FasterGSFused**（`forward.cu:102–152`）：采用 Splatshop 提出的两阶段分离排序：
- **第一阶段**：仅对深度做 32-bit key sort，确定 per-Gaussian 深度顺序
- **第二阶段**：展开 tile-instance 对后，仅用 `end_bit = ceil(log2(n_tiles))` 位做 tile index sort（场景 1000 个 tile 时仅需约 10 bit，而非 32 bit）

**效果**：sort 的实际 bit 宽度从 64 bit 降至约 32+10 = 42 bit，pass 数减少约 4×。

---

## 优化 3：Per-Gaussian Backward（消除 pixel-parallel atomic 冲突）

**3DGS 原版**：backward 按 pixel 并行 → 对同一 Gaussian 的梯度需要 `atomicAdd`，N 个像素同时更新同一 Gaussian 时产生严重 warp 发散和 L2 cache 争用。

**FasterGSFused**（`kernels_backward.cuh:preprocess_backward_cu`）：backward 完全改为按 Gaussian 并行。每个 thread 处理一个 Gaussian，从 forward pass 预存的 `grad_mean2d`, `grad_conic`, `grad_opacities`, `grad_colors` buffer 中直接读取汇总梯度，无需任何 atomic 操作。

**代价**：需要额外的中间 buffer（`grad_mean2d`, `grad_conic`, `grad_colors`，shape = `[N]`），但换来了完全无竞争的 per-Gaussian 并行。

---

## 优化 4：Backward + Adam Fused Kernel（核心贡献）

**3DGS 原版**：
```
backward() → 写梯度到 .grad 属性
optimizer.step() → 读 .grad + 读/写 momentum → 写参数更新
```
两次完整的 N × P 参数读写（N = Gaussian 数，P = 参数维度），Adam 的 momentum 读写是额外的显存带宽消耗。

**FasterGSFused**：在 `preprocess_backward_cu` 中，梯度计算和 Adam 参数更新**完全融合为单一 CUDA kernel**：

```cuda
// 在同一 kernel 内，对每个 Gaussian：
// 1. 计算梯度（SH backward + covariance backward + mean2d backward）
// 2. 立即读 moments_m1, moments_m2（float2）
// 3. 执行 Adam update（FMA 指令）
// 4. 写回更新后的参数 + 新 moments
```

参数以 `register_buffer` 存储（而非 `nn.Parameter`），Adam moments 以显式 `(N, P, 2)` float2 张量存储。虚拟 `autograd_dummy`（`loss += 0 * dummy`）维持计算图以触发 `loss.backward()`，但实际梯度流不经过 PyTorch autograd。

```python
# Trainer.py:150
loss = self.loss(image, rgb_gt) + 0.0 * autograd_dummy
loss.backward()  # 触发 CUDA kernel，Adam step 在 kernel 内完成
# 无 optimizer.step() 调用
```

**效果**：
- 消除 Python-side optimizer 的 6 次 kernel launch + CPU-GPU 同步
- 显存读写从 "先写梯度再读梯度" 变为 "一次读写完成梯度+update"
- Adam 用 FMA 指令（`fmaf`）代替分步浮点运算

同时引入 `adam_step_invisible` kernel，对当前帧不可见（`n_touched_tiles == 0`）的 Gaussian 推进 step count 以保持 bias correction 准确性。

---

## 优化 5：SH + 激活函数 + Backward 内核融合

**3DGS 原版**：SH 求值（forward）、激活函数（exp/sigmoid/normalize）、backward 分别在不同 kernel 或 Python 端执行，产生大量中间 buffer 和 kernel launch overhead。

**FasterGSFused**：在 `preprocess_cu`（forward）中将 SH 求值、covariance 计算、2D projection 完全融合；在 `preprocess_backward_cu` 中将 SH backward、covariance backward、Adam step 融合。激活函数（`expf`, `sigmoid`, `normalize`）内联在 kernel 内，无中间 tensor。

---

## 优化 6：Morton Z-ordering（提升 GPU cache 命中率）

**3DGS 原版**：Gaussian 在内存中的排列顺序取决于初始化顺序，训练过程中 clone/split 产生的新 Gaussian 追加在末尾，导致空间上相邻的 Gaussian 在内存中高度乱序，GPU warp 访问 cache miss 严重。

**FasterGSFused**（`Trainer.py:103–108`，`Model.py:apply_morton_ordering`）：每 5000 iter 对所有 Gaussian 按 Morton 编码重排：
```python
morton_encoding = morton_encode(self._means.data)  # Z-curve 3D→1D 映射
order = torch.argsort(morton_encoding)
self.sort(order)  # 同步更新所有参数 + moments
```
空间上相邻的 Gaussian 在内存中也相邻，warp 访问的 cache line 利用率大幅提升。同时对 Adam moments 张量也同步重排，保证 moments 与参数的对应关系。

---

## 量化结果（Hahlbohm et al. 2026，RTX 4090）

| 数据集 | 3DGS 时间 | FasterGSFused 时间 | 加速比 | PSNR 变化 |
|--------|-----------|-------------------|--------|----------|
| Mip-NeRF360 | 18m44s | **4m31s** | **4.1×** | ±0 dB |
| Deep Blending | 19m43s | **3m46s** | **5.2×** | ±0 dB |
| 推理帧率 | 290 fps | **903 fps** | **3.1×** | — |
| VRAM (Mip-NeRF360) | 8.8 GiB | **6.1 GiB** | −31% | — |

本实验环境（我们的 benchmark，V4 commit 71e1765，Mip-NeRF360）：

| 场景 | 训练时间 | VRAM alloc |
|------|---------|-----------|
| bonsai | ~196s (3.3 min) | 1.14 GiB |
| counter | ~207s (3.5 min) | 1.04 GiB |
| kitchen | ~269s (4.5 min) | 1.56 GiB |
| room | ~194s (3.2 min) | 1.22 GiB |
| bicycle | ~477s (8.0 min) | 4.87 GiB |
| garden | ~484s (8.1 min) | 4.14 GiB |
| stump | ~397s (6.6 min) | 4.20 GiB |

---

## 设计代价与局限

| 代价 | 说明 |
|------|------|
| 不支持 MCMC 密化 | MCMC 需要额外的 relocation optimizer 操作，与 fused kernel 不兼容 |
| 不支持 Mip-Splatting | 3D filter 需要修改 covariance 计算，需单独集成 |
| 不支持 Speedy-Splat | 依赖 PyTorch optimizer state 访问 |
| Adam step count 全局共享 | 新增 Gaussian 的 bias correction 在早期步骤偏差约 3.16×（100 步内自然收敛） |
| moments 张量额外显存 | 相比 3DGS 多约 2× 参数显存（但比 3DGS + 外部 Adam 的 state dict 少） |
