# Faster-GS 相比 3DGS 的具体改进是怎么实现的

## 1. 先说明范围

如果严格按目录名看，这个仓库里没有单独的 `src/Methods/FasterGS`，而是提供了：

- `src/Methods/GaussianSplatting`
- `src/Methods/FasterGSFused`
- `src/Methods/FasterGSFusedDash`

因此本文讨论“Faster-GS 相比 3DGS 的具体改进”时，主要以 `src/Methods/FasterGSFused` 作为当前仓库里最能体现 Faster-GS 思路的实现载体来分析。这样做是合理的，因为：

1. `GaussianSplatting` 保留了原始 3DGS 风格的 PyTorch 训练范式。
2. `FasterGSFused` 明确来自 Faster-GS 系列分支，并在 README 中说明其核心目标是把训练做得更快、更省显存。
3. `FasterGSFusedDash` 则是在 `FasterGSFused` 之上继续叠加 DashGaussian 的调度，不适合作为“Faster-GS 相比 3DGS”的第一层对照对象。

对应论文：

- 3DGS：`papers/3dgs.pdf`
- Faster-GS：`papers/hahlbohm2026fastergs.pdf`

---

## 2. 3DGS 在这个仓库里是怎么实现的

先看基线 `src/Methods/GaussianSplatting`。

它的训练结构非常接近大家熟悉的原始 3DGS：

1. 用 `nn.Parameter` 存储高斯参数  
   位置在 `src/Methods/GaussianSplatting/Model.py` 的 `Gaussians` 类中，包括：
   - `_positions`
   - `_features_dc`
   - `_features_rest`
   - `_scales`
   - `_rotations`
   - `_opacities`

2. 用 PyTorch optimizer 更新参数  
   在 `training_setup()` 里创建 optimizer，在 `Trainer.py` 里单独 `optimizer.step()`。

3. 训练时先渲染，再 backward，再统计 densification 信息  
   `src/Methods/GaussianSplatting/Trainer.py` 里的 `training_iteration()` 会：
   - 调 `renderer.render_image_training()`
   - 得到 `outputs['rgb']`
   - `loss.backward()`
   - 再用 `viewspace_points.grad` 统计 densification 所需的梯度信息

4. densification 由 Python 侧驱动  
   `src/Methods/GaussianSplatting/Model.py` 里的 `densify_and_prune()` 负责：
   - duplicate 小高斯
   - split 大高斯
   - prune 低 opacity 或过大的高斯

这个版本的重点是“方法表达清楚”，不是“训练路径压到极致”。

---

## 3. Faster-GS 的核心目标，在代码里体现为什么

Faster-GS 论文的核心，不是重新发明 3DGS 的损失函数，也不是彻底换掉表示，而是：

- 保留 3DGS 的基本优化目标和质量
- 把训练过程中的冗余访存、分散 kernel、Python 侧 optimizer 开销压下去
- 让训练在相同或接近相同质量下显著加速

这一点在 `FasterGSFused` 代码里体现得非常直接：  
它最大的变化不是 loss，而是“训练执行路径”。

---

## 4. 改进一：把参数更新从 PyTorch optimizer 搬进 CUDA 路径

这是最核心的一步。

### 4.1 3DGS 的做法

在 `GaussianSplatting` 里：

- 高斯参数是 `nn.Parameter`
- optimizer 是 `torch.optim.Adam` 或 Apex `FusedAdam`
- backward 只负责产生梯度
- 参数更新在 Python 侧由 optimizer 完成

这意味着训练每一步都要经过：

- autograd 生成梯度
- optimizer 读取参数和状态
- Python/PyTorch 侧再做一步更新

### 4.2 FasterGSFused 的做法

在 `src/Methods/FasterGSFused/Model.py` 中，高斯参数不再是 `nn.Parameter`，而是改成了 buffer：

- `_means`
- `_sh_coefficients_0`
- `_sh_coefficients_rest`
- `_scales`
- `_rotations`
- `_opacities`

同时显式维护 Adam moments：

- `moments_means`
- `moments_sh_coefficients_0`
- `moments_sh_coefficients_rest`
- `moments_scales`
- `moments_rotations`
- `moments_opacities`

真正的关键在于：  
这些 moments 不是给 PyTorch optimizer 用的，而是直接交给自定义 CUDA backward 内核使用。

入口在：

- `src/Methods/FasterGSFused/Renderer.py`
- `src/Methods/FasterGSFused/FasterGSFusedCudaBackend/.../torch_bindings/rasterization.py`
- `src/Methods/FasterGSFused/FasterGSFusedCudaBackend/.../rasterization/src/backward.cu`

`rasterization.py` 的自定义 autograd `backward()` 最终调用 `_C.backward(...)`。  
而 `backward_wrapper()` 和 `backward()` 里拿到的不只是梯度图像 `grad_image`，还直接拿到了：

- 参数张量
- 各类 moments
- densification_info
- 相机参数

这说明 FasterGSFused 不是“算完梯度再交给外部 Adam”，而是“反传阶段顺带把 Adam 更新做掉”。

### 4.3 为什么这会更快

因为它减少了三类开销：

1. 不再需要为每个参数维护标准 PyTorch optimizer 调度路径
2. 参数、梯度、optimizer 状态都能沿着更连续的 CUDA 访存路径走
3. 一些中间张量不必再在 Python 侧显式暴露和搬运

这正对应 Faster-GS 论文里“融合梯度计算和参数更新、降低内存访问成本”的主线。

---

## 5. 改进二：把 densification 统计从 Python 侧搬进 fused backward

原始 3DGS 里，densification 依赖 `viewspace_points.grad`。

这在 `src/Methods/GaussianSplatting/Renderer.py` 很明显：

- 训练时先创建 `viewspace_points`
- `retain_grad()`
- backward 后再在 Python 里读它的梯度

而在 `FasterGSFused` 里，这条路径被替换掉了。

在 `src/Methods/FasterGSFused/FasterGSFusedCudaBackend/.../rasterization/src/backward.cu` 中，`preprocess_backward_cu` 会在 fused backward 里直接更新 `densification_info`。

换句话说：

- 3DGS：先为 densify 额外保留一个可求导的 2D 投影张量，再从 Python 读梯度
- FasterGSFused：直接在 CUDA 后向里顺手统计 densify 所需信息

这样做的好处是：

1. 少一个专门暴露给 Python 的中间梯度接口
2. 少一次额外的梯度读写和同步
3. densification 信息和参数更新一起留在 fused 路径里

这也是 Faster-GS 在“减少训练时辅助张量开销”上的一个重要实现点。

---

## 6. 改进三：引入 Morton ordering，改善内存局部性

论文里一个很重要的方向是：  
很多训练加速并不来自数学公式变化，而来自 GPU 上的数据访问是否规整。

在 `FasterGSFused` 里，这个思想的代表实现就是 Morton ordering。

对应代码：

- `src/Methods/FasterGSFused/Model.py`：`apply_morton_ordering()`
- `src/Methods/FasterGSFused/Trainer.py`：训练过程中周期性调用 `morton_ordering()`

它做的事情并不复杂：

1. 用 Morton code 对高斯空间位置编码
2. 按编码顺序重排高斯
3. 同时重排所有相关张量
   - 参数
   - SH
   - scales / rotations / opacities
   - moments
   - densification_info

为什么这重要：

- 空间上相近的高斯更可能在渲染和排序时一起被访问
- 连续重排后，cache locality 更好
- 后续 rasterization / backward / densification 的访存模式更规整

它不直接改变质量，但会直接影响吞吐和显存利用率。

---

## 7. 改进四：围绕 3DGS 训练流程加入更完整的工程化增强

除了 fused backward，FasterGSFused 还把一些 3DGS follow-up work 中常见、但在原始 3DGS 里较弱或较散的训练增强吸收进来了。

### 7.1 可选 carving 的随机初始化

在 `src/Methods/FasterGSFused/Trainer.py` 和 `src/Methods/FasterGSFused/utils.py` 中，如果没有点云初始化，代码会：

1. 在 bounding box 里随机采样点
2. 可选地用 `carve()` 去掉明显不会被任何训练视角看到的点
3. 可选地结合 alpha mask 进一步去掉无效区域

这比原始 3DGS 里“没有点云就简单随机撒点”更稳。

### 7.2 随机背景色训练

`Trainer.py` 中的 `USE_RANDOM_BACKGROUND_COLOR` 会在训练时随机背景色。  
这对应 Faster-GS README 中提到的“减少 false transparency”。

实现上很直接：

- 渲染输入背景色随机化
- GT 也用同样背景色合成

它不是 fused 的核心，但确实是 Faster-GS 相比原始 3DGS 在训练鲁棒性上的增强。

### 7.3 非黑背景下的额外 opacity reset

在 `reset_opacities_extra()` 中，如果背景不是黑色，会额外 reset 一次 opacity。  
这是很典型的“基于 3DGS 经验问题加入的工程修复”：

- 原始 3DGS 的假设更接近黑/白背景
- 真正落地时，背景颜色更灵活
- 所以这里加了一层额外保护

---

## 8. 改进五：训练清理和状态管理更偏向高性能实现

原始 3DGS 的 `GaussianSplatting` 训练完会 bake activation，并保留较标准的参数表示。

`FasterGSFused` 的训练收尾则更明显地服务于“高效训练实现”：

- 清掉 densification_info
- prune 低 opacity / degenerate 高斯
- 再做一次 Morton ordering
- 释放 moments
- 输出 `n_gaussians.txt`

对应代码在 `src/Methods/FasterGSFused/Model.py` 的 `training_cleanup()` 和 `src/Methods/FasterGSFused/Trainer.py` 的 `finalize()`。

这说明它从一开始就不是把自己当作“一个最容易讲清楚原理的教学实现”，而是把自己定位成“一个可以长期跑实验的高性能训练基线”。

---

## 9. 什么没有变

这点也很重要。

虽然 FasterGSFused 改了大量实现路径，但有些东西它刻意没大改：

1. 基本表示仍然是 3D Gaussian primitive
2. 基本训练目标仍然是 `L1 + DSSIM`
3. 基本 densification 思想仍然是 duplicate / split / prune
4. SH 表示、opacity / scale / rotation 的参数化方式没有被彻底推翻

也就是说，Faster-GS 在这个仓库里的主线不是“换方法”，而是“把 3DGS 这条方法链条重新梳理成更快的实现”。

---

## 10. 一句话总结每项改进是怎么落到代码里的

如果压缩成最短版本，可以这么理解：

- **原始 3DGS**：参数是 `nn.Parameter`，optimizer 在 Python 侧，densification 统计依赖 `viewspace_points.grad`。
- **FasterGSFused**：参数和 Adam moments 改成 fused buffer，自定义 CUDA backward 里直接完成梯度传播、densification 统计和参数更新。
- **额外提速手段**：训练过程中加入 Morton ordering、carving、随机背景色、训练 cleanup 等，使整条训练路径更适合高吞吐 GPU 执行。

所以，Faster-GS 相比 3DGS 的“具体改进怎么实现”，最本质的答案是：

> 它没有大幅改写 3DGS 的建模目标，而是把 3DGS 原本分散在 PyTorch autograd、Python optimizer、辅助统计逻辑中的训练流程，重新压缩、融合、重排成一条更连续的 CUDA 执行路径。

这也是为什么它能在保持方法家族连续性的同时，把训练时间明显压下来。
