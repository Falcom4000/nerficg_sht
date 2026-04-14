# FasterGSFusedDash 后续优化路线图

Date: 2026-04-11

---

## 背景与现状

FasterGSFusedDash（当前 commit: 71e1765）已完成：
- 6 个 FasterGSFused × DashGaussian 集成冲突的修复（V0）
- Cold-primitive bias correction CUDA 内核（V1）
- Render_scale 每步更新（V1，已实现）
- 自适应 Morton 触发（V1）
- SH unlock 无 resolution gate（V3）
- Prune-first + DashGaussian budget 公式（V4）

当前质量差距（vs FasterGSFused，6-run avg，Mip-NeRF360）：
- 大型室外场景（bicycle/garden/stump）：-0.06 ~ -0.48 dB，换来 1.25× 加速
- 室内场景（bonsai/counter/kitchen/room）：-0.02 ~ -0.21 dB，速度相当

以下 12 项优化尚未实现，按推荐优先级排列。

---

## 优化路线图

| # | 优化项 | 类别 | 工程量 | 依赖 | 推荐度 |
|---|--------|------|-------|------|-------|
| 13 | Invisible moment decay（V2.1 写过，未单独验证） | CUDA 算法 | 极小 | DashGaussian | ★★★★ |
| 4 | Adam moment 重缩放（分辨率跳变时） | 训练算法 | 小 | DashGaussian | ★★★★ |
| 1 | 预分配 Gaussian buffer | 内存结构 | 中 | DashGaussian max_n 已知 | ★★★★★ |
| 2 | per-Gaussian Adam step count | CUDA 算法 | 中 | 通用 | ★★★ |
| 3 | tile buffer 一次性预分配 | 内存结构 | 中 | DashGaussian 单调分辨率 | ★★★ |
| 6 | backward 中间 buffer FP16 | CUDA 内核 | 中 | 通用 | ★★★ |
| 5 | 合并 6 个 invisible kernel 为 1 | CUDA 内核 | 小 | 通用 | ★★ |
| 7 | CUDA Graph 捕获训练步 | Python/CUDA | 大 | 通用 | ★★ |
| 11 | 稀疏 SH Adam 更新 | CUDA 算法 | 小 | 通用 | ★★ |
| 9 | preprocess_cu 向量化 load | CUDA 内核 | 中 | 通用 | ★★ |
| 8 | CUDA 端 top-k mask 生成 | CUDA 内核 | 大 | DashGaussian | ★ |
| 10 | SH 内存布局转置 | 内存结构 | 大 | 通用 | ★ |

---

## 各项详细说明

---

### #13 Invisible moment decay（★★★★，极小工程量）

**问题**

当 `render_scale > 1` 时，部分 Gaussian 在当前低分辨率下不可见（`n_touched_tiles == 0`）。
当前 V0/V1 实现（`apply_invisible_momentum=False`）完全冻结这些 Gaussian 的 Adam moments。
问题在于：低分辨率训练可能持续 10,000–15,000 步，冻结的 moment 在切回全分辨率后
仍保留原来的方向和幅值（`0.9^1 = 0.9`，未衰减），切换后前几百步出现方向性偏差。

**V2.1 尝试**

V2.1 引入了 `decay_invisible_moments` CUDA kernel：
```cuda
// 每步对 n_touched_tiles == 0 的 Gaussian 执行
moments[idx] *= make_float2(beta1, beta2);  // 0.9, 0.999
```
即不更新参数，但让 moments 自然衰减。`0.9^15000 ≈ 0`，陈旧 momentum 自然消失。

**为什么 V2.1 被回退**

V2.1 同时引入了 5 项改动（prune-first、opacity moments 不清零、budget 公式、invisible decay、GT 预缓存），
最终因为 opacity moments 不清零 + budget 公式联合导致 Gaussian 数量爆炸而整体回退。
**invisible decay 本身从未被单独验证。**

**当前状态**

71e1765 的 `Renderer.py:87` 仍是：
```python
apply_invisible_momentum=(render_scale == 1)  # V0 冻结行为
```

**建议**

单独将 decay 行为引入，跑 bonsai + garden 各 3 run 与 V4 baseline 对比。
CUDA 逻辑已知，工程量极小（~20 行 CUDA + Python 侧 flag 传参）。

---

### #4 Adam moment 重缩放（分辨率跳变时）（★★★★，小工程量）

**问题**

DashGaussian 的分辨率调度会产生若干次 `render_scale` 下降（如 8→4→2→1）。
每次下降时，所有参数的 Adam moments 中积累的是低分辨率下的梯度统计。
以 position 为例，低分辨率下 µ2D 梯度幅值约为高分辨率的 `1/render_scale`，
所以 m1（一阶矩）偏小约 `1/r`，m2（二阶矩）偏小约 `1/r²`。

切换到更高分辨率后，新梯度幅值突增，但 m2 的历史积累偏小，
导致 Adam 的有效步长在转换后短期内偏大（分母偏小）。

**已有缓解**

- Conflict J fix（`lr_decay_from_iter`）：延迟了 position LR 衰减，缓解了 position 的问题。
- 但其他参数（scales、rotations、opacities、SH）的 moment mismatch **未处理**。

**修复方案**

```python
# Trainer.py:densify 末尾，检测 render_scale 变化
old_scale = self.current_render_scale
new_scale = self.dash_scheduler.get_res_scale(iteration)
if new_scale < old_scale:  # 分辨率提升（scale 数值减小）
    ratio = old_scale / new_scale  # e.g., 4→2 时 ratio=2
    # m1 按梯度线性缩放：× ratio
    # m2 按梯度² 缩放：× ratio²
    for moments in [self.model.gaussians.moments_means,
                    self.model.gaussians.moments_scales,
                    self.model.gaussians.moments_rotations,
                    self.model.gaussians.moments_opacities,
                    self.model.gaussians.moments_sh_coefficients_0,
                    self.model.gaussians.moments_sh_coefficients_rest]:
        moments[..., 0] *= ratio      # m1
        moments[..., 1] *= ratio ** 2  # m2
```

**注意**

这是理论推导，实际效果需要实验验证。分辨率转换期只有约 100–500 步，
如果 Adam 自然适应速度够快，实际改善可能有限。建议先做 #13，再做 #4。

---

### #1 预分配 Gaussian buffer（★★★★★，中工程量）

**问题**

当前 `dash_density_control_topk`（`Model.py:266`）每次密化执行 12 次 `torch.cat`：
```python
self._means.data = torch.cat([self._means, duplicated_means, split_means])
# × 6 参数 + × 6 moments = 12 次
```
N=400 万时，单次密化涉及约 1 GB 数据搬运，触发 12 个中间 tensor 分配，
并在有 `requires_empty_cache=True` 时额外触发 `torch.cuda.empty_cache()`（CPU-GPU 同步）。

密化期（iter 600–27,000，共 264 次密化）累积开销显著。

**DashGaussian 为什么使这个优化可行**

标准 FasterGSFused 的 ADC 密化无上界，N 可以无限增长（最终达 4–6M），
预分配会严重高估（内存浪费）或低估（运行时溢出）。

DashGaussian 在 `setup_gaussians` 时就确定了 `max_n_gaussian`：
- `MAX_N_GAUSSIANS > 0`：用户硬上限，完全确定
- `MAX_N_GAUSSIANS = -1`：momentum-adaptive，由 `initial_momentum_factor × init_n` 给出上界

这个上界紧凑可靠，使预分配成为安全操作。

**修复方案（Python 侧，无需改 CUDA）**

```python
# setup_gaussians 时
max_n = self.dash_scheduler.max_n_gaussian
self._means      = torch.zeros(max_n, 3,    device='cuda')
self._scales     = torch.zeros(max_n, 3,    device='cuda')
self._rotations  = torch.zeros(max_n, 4,    device='cuda')
self._opacities  = torch.zeros(max_n, 1,    device='cuda')
self._sh_0       = torch.zeros(max_n, 1, 3, device='cuda')
self._sh_rest    = torch.zeros(max_n, 15, 3, device='cuda')
self.moments_*   = torch.zeros(max_n, ..., 2, device='cuda')
self._n_active   = init_n

# densify 时（O(n_new)，无分配）
end = self._n_active + n_new
self._means[self._n_active:end] = new_means
self._n_active = end

# 传给 CUDA 时切片
means = self._means[:self._n_active]
```

CUDA kernel 已经只处理前 `n_primitives` 个，无需修改 CUDA 代码。

**工程注意点**

- `prune` 操作需要改为在 active 范围内做 index 压缩（仍是 O(N) 但避免额外分配）
- `sort` 操作同理
- 需要在训练开始前确认 `max_n_gaussian` 不会被 under-estimate（建议加 10% margin）

---

### #2 per-Gaussian Adam step count（★★★，中工程量）

**问题**

FasterGSFused 用全局 `optimization_step`（= iteration + 1）作为所有 Gaussian 的 Adam step count
做 bias correction。对于在 iter=20,000 新增的 Gaussian（moments=0，t=20,001）：
```
bias_correction1_rcp = 1/(1 - 0.9^20001) ≈ 1.0   # 应补偿 cold-start，实际不补偿
bias_correction2_rcp = 1/(1 - 0.999^20001) ≈ 1.0
→ 首步更新幅度 = 3.16 × lr（正确值为 1.0 × lr）
```

**V1 的 Cold-primitive 检测（已实现）**

V1 在 `preprocess_backward_cu` 中检测 `moments_means` 全零（新 Gaussian 的标志），
自动切换到 t=1 的 bias correction，消除了首步 3.16× 过大更新的问题。

这覆盖了**绝大多数**场景：新 Gaussian 在加入后的前几步 moments 接近零，cold detection 生效。

**剩余 gap**

V1 cold detection 是二值的（全零 vs 非零），无法精确追踪每个 Gaussian 经历了多少步真实更新。
在 Gaussian 经历约 50–100 步更新后 moments 已非零，cold detection 失效，
但此时 t 实际可能只有 50，全局 step count 可能是 20,050，偏差仍存在（但幅度已降至 ~1.3× 而非 3.16×）。

**完整修复**

为每个 Gaussian 维护 `uint16` step count（`(N,)` tensor，仅 2 bytes/Gaussian）：
```cuda
const uint t = step_counts[primitive_idx];
const float bc1 = 1.0f / (1.0f - powf(beta1, (float)t));
const float bc2 = rsqrtf(1.0f - powf(beta2, (float)t));
step_counts[primitive_idx] = min(t + 1, 65535u);
```
densify 时新 Gaussian 的 step count 初始化为 0，prune/sort 时同步维护。

**优先级降级原因**

V1 已处理最坏情况（3.16× → ~1×），剩余偏差较小且会自然收敛。
相比 #13 和 #4，这项改动需要修改 CUDA 接口（新增 tensor 参数），
以及在 prune/sort/densify 的 Python 端同步维护，工程量相对较高，
而剩余收益估计小于 #13 和 #4。

---

### #3 tile buffer 一次性预分配（★★★，中工程量）

**问题**

当前 `rasterization_api.cu` 的 `forward_wrapper` 通过 `resize_function_wrapper` 按需动态分配 tile buffer：
```cpp
char* tile_buffers_blob = resize_tile_buffers(required<TileBuffers>(n_tiles));
```
`n_tiles` 取决于渲染分辨率。DashGaussian 的分辨率**单调递增**（render_scale 从 7 降到 1），
意味着 tile buffer 大小在训练过程中单调增大。

每次分辨率提升（约 4–6 次）都触发 CUDA 内存重分配。PyTorch CUDACachingAllocator
有内存复用机制，但不保证零开销，分辨率快速增长期可能产生额外碎片和分配延迟。

**修复方案**

`setup_gaussians` 时以全分辨率预分配并固定传入：
```python
full_w = dataset.train()[0].camera.width
full_h = dataset.train()[0].camera.height
n_tiles = ((full_w + 15)//16) * ((full_h + 15)//16)
self.pre_tile_buffer = torch.empty(TileBuffers_size(n_tiles), dtype=torch.uint8, device='cuda')
```
需要修改 C++ 接口，将 pre-allocated buffer 传入 `forward_wrapper` 而非内部分配。

**单独用 FasterGSFused 时此问题不存在**（分辨率固定），仅在 DashGaussian 动态分辨率下才有意义。

---

### #6 backward 中间 buffer FP16（★★★，中工程量）

**问题**

blend_backward kernel 将 per-Gaussian 汇总梯度写入中间 buffer：
```
grad_mean2d   (N, 2)  float32  → 8 bytes/Gaussian
grad_conic    (N, 3)  float32  → 12 bytes/Gaussian
grad_colors   (N, 3)  float32  → 12 bytes/Gaussian
grad_opacities (N, 1) float32  → 4 bytes/Gaussian
```
共 36 bytes/Gaussian。N=400 万时约 144 MB，在 blend_backward → preprocess_backward 之间完整读写一次。

**修复方案**

将中间 buffer 改为 float16（18 bytes/Gaussian），blend_backward 写 fp16，
preprocess_backward 读时转换回 fp32：
```cuda
__half2 grad_mean2d_fp16 = __float22half2_rn(grad_mean2d_val);
```
梯度本身是近似量，fp16 精度（半精度，~3 位小数）对 Adam update 已足够。

**收益场景**

N=400 万（bicycle/garden）：节省约 72 MB 带宽/iter × 30,000 iter = ~2 TB 总带宽节省。
N=100 万（bonsai/counter）：收益约 1/4，相对不显著。

---

### #5 合并 6 个 adam_step_invisible kernel（★★，小工程量）

**问题**

每步对不可见 Gaussian 执行 6 次独立 kernel launch（means/scales/rotations/opacities/sh_0/sh_rest）：
```cuda
// 6 次分别 launch，每次 O(N)
adam_step_invisible<3>(moments_means, ...)
adam_step_invisible<3>(moments_scales, ...)
...
```
每次 kernel launch 约 5–10 μs CPU overhead。6 次 = 30–60 μs/iter。

30,000 iter × 30–60 μs = 0.9–1.8 秒，占比约 0.5–1%。

**修复方案**

合并为单一 kernel，一个 thread 处理一个 Gaussian 的全部参数组：
```cuda
__global__ void adam_step_invisible_all(
    float2* moments_means,     // stride=3
    float2* moments_scales,    // stride=3
    float2* moments_rotations, // stride=4
    float2* moments_opacities, // stride=1
    float2* moments_sh_0,      // stride=3
    float2* moments_sh_rest,   // stride=45
    const uint* n_touched_tiles,
    const uint n_primitives)
{
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_primitives || n_touched_tiles[idx] != 0) return;
    // 按各自 stride 处理所有参数组
}
```

**优先级降级原因**

0.5–1% 的绝对收益较小。小 N 时（early training）有感，大 N 时 launch latency 不是瓶颈。
若同时实现 #1（预分配 buffer），早期小 N 阶段已被加速，此项边际收益进一步降低。

---

### #7 CUDA Graph 捕获训练步（★★，大工程量）

**问题**

每次 `training_iteration` 触发约 10+ 个 kernel launch，每个都有 ~5–10 μs 的 CPU→GPU 命令提交开销。
总 overhead 约 50–100 μs/iter × 30,000 iter = 1.5–3 秒，占比约 1–3%。

**修复方案**

用 `torch.cuda.CUDAGraph` 捕获"正常训练步"（非密化步），密化步 fallback 到常规执行：
```python
# 初始化（或 N 变化后重新捕获）
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    image, dummy = self.renderer.render_image_training(...)
    loss = ...
    loss.backward()

# 99% 的步骤直接 replay
if is_densification_step:
    # 正常执行 + 重新捕获
else:
    graph.replay()
```

**工程风险**

- N 每 100 步变化一次（密化步），需要在每次 N 变化后重新捕获（约 264 次）。
- CUDA Graph 要求静态内存地址，预分配 buffer（#1）是前置条件。
- GT 采样、sampler、wandb 等有副作用的操作需要独立处理。
- 调试困难，错误不直观。

**建议**：先完成 #1（预分配 buffer），再考虑 CUDA Graph。

---

### #11 稀疏 SH Adam 更新（★★，小工程量）

**问题**

训练前期（active_sh_degree < max_sh_degree）高阶 SH 系数的梯度为零，
但 fused Adam kernel 仍对这些系数执行 moment 更新（读写 moments_sh_rest 的全部 45 个 float2）。

以 degree=0（前 1000 步）为例：
- 实际有用的 SH 系数：degree-0，3 个 float
- 实际更新的 SH 系数：degree-0 至 degree-3，共 48 个 float
- 无效更新比例：(48-3)/48 = 94%

**修复方案**

在 CUDA kernel 中根据 `active_sh_bases` 跳过未激活的高阶 SH：
```cuda
const uint active = active_sh_bases;  // 传入 kernel
// 只更新前 active 个 basis 的 moments
for (uint b = 0; b < active; ++b) {
    adam_step_helper<3>(grad_colors[b], sh_rest, moments_sh_rest, ...);
}
```

**优先级**

收益窗口仅前 1000–3000 步（SH 解锁前），全程看贡献极小。
若 #1（预分配 buffer）已做，densification 性能已大幅改善，此项边际收益更低。

---

### #9 preprocess_cu 向量化 load（★★，中工程量）

**问题**

`preprocess_cu` 中 `means (float3)` 的 12-byte 访问不能利用 GPU 的 128-bit 向量化 load（`ld.global.v4.f32`），
实际利用率约 75%（12/16 bytes）。

**修复方案**

将 means 存储为 `float4`（增加 1 个 dummy float，padding 为 16 bytes）：
```cuda
const float4 mean4 = reinterpret_cast<const float4*>(means)[primitive_idx];
const float3 mean3d = {mean4.x, mean4.y, mean4.z};
```
Python 侧 `_means` 改为 `(N, 4)` shape，第 4 维填 0。

**工程影响**

需要修改所有读写 `_means` 的路径（prune/sort/densify/cat/ply export），影响面较广。
加上 #1（预分配 buffer）后改动量会更大，需要协调。

---

### #8 CUDA 端 top-k mask 生成（★，大工程量）

**问题（实际较小）**

`dash_density_control_topk`（`Model.py:305`）中用 Python 调用 `torch.topk`：
```python
topk_indices = scored.topk(n_budget).indices
```
`torch.topk` 是纯 GPU CUDA 操作（内部用 cub/thrust），并非真正的 Python bottleneck。
实际 overhead 仅为 Python dispatch 的一次 kernel launch 调度，约 5–10 μs，每 100 步触发一次。

**为什么推荐度低**

收益极边际：264 次密化 × 10 μs = 2.6 ms 总节省。
工程量却很大：需要在 `preprocess_backward_cu` 末尾写入 grad scores，
另外实现 CUDA topk kernel 或适配 cub `DeviceRadixSort`，
并修改 Python-CUDA 接口。

---

### #10 SH 内存布局转置（★，大工程量）

**问题**

SH 系数以 `(N, D, 3)` 存储（N=Gaussian 数，D=SH basis 数，3=RGB）。
backward 中对同一 Gaussian 的 D 个 basis 连续访问时，内存跨度 = 3 float，
不同 Gaussian 的同一 basis 在内存中间隔 D×3 float，warp 内访问不连续。

理论上 `(D, N, 3)` 布局更 coalesced：warp 内不同 Gaussian 的同一 basis 连续。

**为什么推荐度低**

改动影响极广：prune/sort/densify（`torch.cat`）/checkpoint 保存/ply export
全部需要适配新布局。V2 的教训表明多处同时改动容易产生难以定位的 bug。
且 SH backward 在总训练时间中占比相对较低，实际加速有限。

---

## 实施建议

### 最优先（风险低，收益可量化）

1. **#13 Invisible moment decay**：先单独实验，bonsai+garden 各 3 run 对比 V4。
   CUDA 逻辑已知，改动约 20 行。

2. **#4 Adam moment 重缩放**：单独实验，分辨率跳变时加 4 行 Python。
   若 #13 有正向收益，再叠加 #4 观察是否进一步改善。

### 中期（工程量中等，收益稳定）

3. **#1 预分配 Gaussian buffer**：需要重构 `Model.py` 的 prune/sort/densify。
   建议在 #13/#4 实验完成、质量稳定后再做，避免引入新变量。

### 低优先（边际收益或高风险）

4. **#2 per-Gaussian step count**：V1 cold-primitive 已处理主要问题，
   完整修复工程量中等，收益估计 <0.1 dB。

5. **#3 tile buffer 预分配**、**#6 FP16 buffer**：CUDA 改动，
   在前述优化稳定后作为性能精调。

6. **#5、#7、#9、#11**：边际收益，按需。

7. **#8、#10**：不建议，性价比过低。
