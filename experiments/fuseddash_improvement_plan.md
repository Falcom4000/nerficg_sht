# FasterGSFusedDash 改进计划

## Context

FasterGSFusedDash 将 DashGaussian 的分辨率调度融入 FasterGSFused 的 fused-Adam 管线。
当前 benchmark 显示 garden 场景有 -0.21 dB 质量差距（vs FasterGSDash），主因是 fused Adam 使用全局 step count 而 PyTorch Adam 有 per-parameter step count，导致新 Gaussian 首步更新 3.16× 过大。

只修改 DASH 目录的文件，不动 BASE FasterGSFused。

---

## 步骤总览

| 步骤 | 内容 | 文件 |
|------|------|------|
| 1 | DASH 本地 CUDA 后端恢复 `apply_invisible_momentum` | 5 个 CUDA/Python 文件 |
| 2 | 添加 cold-primitive bias correction | `kernels_backward.cuh` |
| 3 | Renderer 切换到本地 CUDA 后端 | `Renderer.py` |
| 4 | 每步更新 render_scale | `Trainer.py` |
| 5 | Morton ordering 自适应触发 | `Trainer.py` |
| 6 | 重编译 + 验证 | shell |

---

## 步骤 1：恢复 apply_invisible_momentum（对齐 BASE）

DASH 本地 CUDA 后端当前缺少 `apply_invisible_momentum` 参数。需在 5 个文件中补回，全部是小 diff。

BASE 路径前缀：`src/Methods/FasterGSFused/FasterGSFusedCudaBackend/FasterGSFusedCudaBackend/`
DASH 路径前缀：`src/Methods/FasterGSFusedDash/FasterGSFusedCudaBackend/FasterGSFusedCudaBackend/`

### 1a. `rasterization/include/backward.h`
行 48：`const int adam_step_count);` → `const int adam_step_count, const bool apply_invisible_momentum);`

### 1b. `rasterization/include/rasterization_api.h`
行 67：`const int instance_primitive_indices_selector);` → `const int instance_primitive_indices_selector, const bool apply_invisible_momentum = true);`

### 1c. `rasterization/src/rasterization_api.cu`
行 127：`const int instance_primitive_indices_selector)` → `const int instance_primitive_indices_selector, const bool apply_invisible_momentum)`
行 179：`adam_step_count` 之后添加 `, apply_invisible_momentum`

### 1d. `rasterization/src/backward.cu`
行 49：`const int adam_step_count)` → `const int adam_step_count, const bool apply_invisible_momentum)`
行 128 起：将所有 `adam_step_invisible` 调用包裹在 `if (apply_invisible_momentum) { ... }`

### 1e. `torch_bindings/rasterization.py`
对齐 BASE 版本：
- `_Rasterize.forward()` 添加 `apply_invisible_momentum: bool` 参数，保存到 ctx
- `_Rasterize.backward()` 传 `ctx.apply_invisible_momentum` 给 `_C.backward()`，return tuple 多一个 None
- `diff_rasterize()` 添加 `apply_invisible_momentum: bool = True` 参数并传入 `_Rasterize.apply()`

---

## 步骤 2：Cold-primitive bias correction（核心改动）

### 文件
`FasterGSFusedDash/.../rasterization/include/kernels_backward.cuh`

### 在 `preprocess_backward_cu` 中的 line 50 early-return 之后，插入：
```cuda
// Cold primitive detection: newly created Gaussians have all-zero moments.
// Use t=1 bias correction instead of the (much weaker) global-step correction
// to match PyTorch Adam's per-parameter step-count semantics.
const float2 _cold_check = moments_means[primitive_idx * 3];
float eff_bc1_rcp = bias_correction1_rcp;
float eff_bc2_sqrt_rcp = bias_correction2_sqrt_rcp;
float eff_step_size_means = step_size_means;
if (_cold_check.x == 0.0f && _cold_check.y == 0.0f) {
    eff_bc1_rcp = 1.0f / (1.0f - config::beta1);
    eff_bc2_sqrt_rcp = rsqrtf(1.0f - config::beta2);
    eff_step_size_means = (step_size_means / bias_correction1_rcp) * eff_bc1_rcp;
}
```

### 然后在函数其余部分做替换（仅在 preprocess_backward_cu 内部）：

| 原变量 | 替换为 | 涉及位置 |
|--------|--------|----------|
| `bias_correction1_rcp` | `eff_bc1_rcp` | step_size_opacities 计算 (L52)、step_size_scales 计算 (L225)、step_size_rotations 计算 (L243)、`convert_sh_to_color_backward` 调用 (L65) |
| `bias_correction2_sqrt_rcp` | `eff_bc2_sqrt_rcp` | 所有 `adam_step_helper` 调用的最后一个参数 (L53, L211-213, L226-228, L244-247)、`convert_sh_to_color_backward` 调用 (L66) |
| `step_size_means` | `eff_step_size_means` | means 的 adam_step_helper 调用 (L211-213) |

### 不需要修改的
- `adam_step_helper` 签名（kernel_utils.cuh）— 不变
- `sh_utils.cuh` — 通过参数传入已修正的值，自动正确
- `adam_step_invisible` kernel — 不需要 cold detection
- `Model.py:reset_opacities()` — 只清零 opacity moments，means moments 不受影响，不会误触发

---

## 步骤 3：切换 Renderer 导入

### 文件
`src/Methods/FasterGSFusedDash/Renderer.py`

行 14：
```python
from Methods.FasterGSFused.FasterGSFusedCudaBackend import diff_rasterize, RasterizerSettings
```
→
```python
from Methods.FasterGSFusedDash.FasterGSFusedCudaBackend import diff_rasterize, RasterizerSettings
```

---

## 步骤 4：每步更新 render_scale

### 文件
`src/Methods/FasterGSFusedDash/Trainer.py`

`training_iteration` 方法中，line 189：
```python
render_scale = self.current_render_scale
```
→
```python
render_scale = self.dash_scheduler.get_res_scale(iteration)
self.current_render_scale = render_scale
```

---

## 步骤 5：Morton ordering 自适应触发

### 文件
`src/Methods/FasterGSFusedDash/Trainer.py`

**5a.** `setup_gaussians` 末尾（line 117 之后）添加：
```python
self._last_morton_n = 0
```

**5b.** `morton_ordering` callback 改为：
```python
@training_callback(priority=99, end_iteration='MORTON_ORDERING_END_ITERATION',
                   iteration_stride='DENSIFICATION_INTERVAL')
@torch.no_grad()
def morton_ordering(self, *_) -> None:
    n = self.model.gaussians.means.shape[0]
    if n < self.MORTON_ORDERING_MIN_GAUSSIANS:
        return
    if n < self._last_morton_n * 1.2:
        return
    self.model.gaussians.apply_morton_ordering()
    self._last_morton_n = n
```

**5c.** 删除配置中的 `MORTON_ORDERING_INTERVAL=5000`（不再使用，`iteration_stride` 改为引用 `DENSIFICATION_INTERVAL`）。

---

## 步骤 6：编译 & 验证

```bash
cd src/Methods/FasterGSFusedDash/FasterGSFusedCudaBackend
pip install . --no-build-isolation
```

验证：
1. 跑 bonsai 确认不 crash、PSNR 不退步
2. 跑 garden 对比 PSNR 是否缩小与 FasterGSDash 的差距
