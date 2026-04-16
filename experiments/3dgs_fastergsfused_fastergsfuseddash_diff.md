# 从 3DGS 到 FasterGSFused 到 FasterGSFusedDash 的增量修改文档

## 1. 范围与对应关系

本文只分析本仓库中的三套实现，以及它们相对前一阶段“新增了什么、修改了什么”。

标注规则：

- `[论文]`：该点能直接在对应论文的摘要、方法或动机中找到明确表述
- `[论文落地]`：该点是论文思想在本仓库中的具体实现形态
- `[实现补丁]`：该点主要是仓库实现层为了稳定性、兼容性或工程效率额外加入，不一定是论文正文明确主张

- `src/Methods/GaussianSplatting`：原始 3DGS 风格实现
- `src/Methods/FasterGSFused`：Faster-GS 的 fused 训练实现
- `src/Methods/FasterGSFusedDash`：在 `FasterGSFused` 上叠加 DashGaussian 调度与稳定性修正

对应论文：

- `papers/3dgs.pdf`
- `papers/hahlbohm2026fastergs.pdf`
- `papers/dashGaussin.pdf`

---

## 2. 总体演进脉络

可以把三者理解为三层递进：

1. `GaussianSplatting` 解决“3DGS 如何作为显式表示和可微渲染框架工作”。 `[论文]`
2. `FasterGSFused` 解决“在不明显改变目标函数和质量的前提下，如何把训练过程做得更快、更省显存”。 `[论文]`
3. `FasterGSFusedDash` 解决“在 fused 高效实现基础上，如何进一步通过训练复杂度调度减少前期冗余计算”。 `[论文]`

简化地说：

- `3DGS -> FasterGSFused`：核心是“实现优化”
- `FasterGSFused -> FasterGSFusedDash`：核心是“训练调度优化”

---

## 3. 从 GaussianSplatting 到 FasterGSFused

### 3.1 参数表示方式改变

`GaussianSplatting` 使用 `nn.Parameter + Adam/FusedAdam` 的标准 PyTorch 训练范式。

- 位置：`src/Methods/GaussianSplatting/Model.py`
- 高斯参数：`_positions / _features_dc / _features_rest / _scales / _rotations / _opacities`
- 优化器：在 `training_setup()` 中创建，训练时单独 `optimizer.step()`

这部分对应原始 3DGS 的常规优化实现。 `[论文落地]`

`FasterGSFused` 把这套设计改成了：

- 高斯属性改为 `register_buffer`
- 显式维护 Adam 一阶/二阶矩
- 不再依赖 PyTorch optimizer 对每个参数逐项更新

对应位置：

- `src/Methods/FasterGSFused/Model.py`
- moments：`moments_means / moments_sh_coefficients_0 / moments_sh_coefficients_rest / moments_scales / moments_rotations / moments_opacities`

这一步的意义是：把“参数更新”从 Python/PyTorch optimizer 搬进 CUDA 路径，减少中间梯度张量和 optimizer 状态搬运。 `[论文]`

其中“buffer 化 + 显式 moment 张量”的具体组织方式是仓库的实现形态。 `[论文落地]`

### 3.2 训练主循环改变

`GaussianSplatting` 的训练路径是：

1. Python 调用 renderer
2. 得到 `viewspace_points`
3. `loss.backward()`
4. 在 Python 侧累积 densification 统计
5. `optimizer.step()`

对应位置：

- `src/Methods/GaussianSplatting/Trainer.py`
- `src/Methods/GaussianSplatting/Renderer.py`

`FasterGSFused` 改成：

1. Python 只负责调度
2. 渲染、反传、参数更新尽量走 fused rasterizer
3. 通过 `autograd_dummy` 触发自定义 autograd
4. densification 信息也直接由 CUDA backward 累积

对应位置：

- `src/Methods/FasterGSFused/Trainer.py`
- `src/Methods/FasterGSFused/Renderer.py`
- `src/Methods/FasterGSFused/FasterGSFusedCudaBackend/.../torch_bindings/rasterization.py`

这一步是 `FasterGSFused` 最核心的改造。 `[论文]`

其中 `autograd_dummy` 这种为了挂接自定义 autograd 的技巧是典型实现层手段。 `[实现补丁]`

### 3.3 密化与清理逻辑重构

`GaussianSplatting` 使用经典 `densify_and_prune()`：

- 基于 `viewspace_points.grad` 统计梯度
- duplicate 小高斯
- split 大高斯
- 再按 opacity 和 size prune

`FasterGSFused` 仍保留“duplicate/split/prune”思想，但实现改成面向 fused buffer 的版本：

- `adaptive_density_control()`
- `reset_densification_info()`
- `prune()` 直接同步裁剪参数和 moments

对应位置：

- `src/Methods/FasterGSFused/Model.py`

保留 3DGS 的 ADC 思路而重写执行路径，属于“保持原方法行为、优化实现”的典型 Faster-GS 方向。 `[论文]`

此外还加入了训练中的 Morton ordering：

- `apply_morton_ordering()`
- 训练中周期性重排高斯和 optimizer moments

对应位置：

- `src/Methods/FasterGSFused/Trainer.py`
- `src/Methods/FasterGSFused/Model.py`

它的目标不是改质量，而是改善内存局部性、排序和访问效率。 `[论文]`

但“在这些 Python/CUDA 边界上怎么接入 Morton ordering”属于仓库落地方式。 `[论文落地]`

### 3.4 初始化与训练细节扩展

相对 `GaussianSplatting`，`FasterGSFused` 还增加了几项工程化训练能力：

- 随机初始化可选 carving
- 非黑背景下的额外 opacity reset
- 可选随机背景色训练
- 训练结束后更明确的 cleanup 和 `n_gaussians.txt` 输出

对应位置：

- `src/Methods/FasterGSFused/Trainer.py`
- `src/Methods/FasterGSFused/utils.py`

其中：

- carving / 更灵活初始化：更接近 Faster-GS 所整合的训练增强方向。 `[论文]`
- `n_gaussians.txt` 输出、训练收尾结构化清理等：主要是仓库工程配套。 `[实现补丁]`

### 3.5 没有明显变化的部分

损失函数基本没变，仍然是：

- `L1`
- `DSSIM`
- `PSNR` 作为质量指标

对应位置：

- `src/Methods/GaussianSplatting/Loss.py`
- `src/Methods/FasterGSFused/Loss.py`

换言之，`FasterGSFused` 的增量主要不是“优化目标变化”，而是“训练执行方式变化”。

这和 Faster-GS 论文的核心叙述是一致的。 `[论文]`

---

## 4. 从 FasterGSFused 到 FasterGSFusedDash

### 4.1 新增 DashGaussian 调度器

这是这一阶段最明显的新增。

`FasterGSFusedDash` 在 trainer 中引入：

- `TrainingScheduler`
- `DASH` 配置块
- `render_scale`
- `densify_rate`

对应位置：

- `src/Methods/FasterGSFusedDash/Trainer.py`
- `src/Methods/FasterGSDash/schedule_utils.py`

调度器根据训练图像频域信息初始化，控制：

- 当前训练分辨率
- 当前 densification 允许的高斯增长速率

这两点都是 DashGaussian 论文的主张。 `[论文]`

而把调度器单独放在 `src/Methods/FasterGSDash/schedule_utils.py`，再由 `FasterGSFusedDash` 复用，是本仓库的模块化组织。 `[论文落地]`

### 4.2 训练由“固定全分辨率”改为“渐进分辨率”

`FasterGSFused` 训练时始终全分辨率渲染。

`FasterGSFusedDash` 则在训练中引入 `render_scale`：

- 早期低分辨率训练
- 中后期逐步升到全分辨率
- GT 也同步下采样监督

对应位置：

- `src/Methods/FasterGSFusedDash/Trainer.py`
- `src/Methods/FasterGSFusedDash/Renderer.py`

这里不是只把 GT 缩小，而是把相机内参也按比例缩放后传入 CUDA：

- `width / height`
- `focal_x / focal_y`
- `center_x / center_y`

所以它是“真正按低分辨率渲染”，不是简单的图像后处理。 `[论文落地]`

“训练早期低分辨率、后期逐步升到全分辨率”的思想本身来自 DashGaussian。 `[论文]`

### 4.3 密化由“过阈值即 densify”改为“带预算的 top-k densify”

`FasterGSFused` 使用 `adaptive_density_control()`，本质仍是标准 ADC。

`FasterGSFusedDash` 新增 `dash_density_control_topk()`，改动点包括：

- densify 前先 prune 低 opacity / degenerate 高斯
- 根据 `densify_rate` 计算本轮 budget
- 如果过阈值的高斯太多，只取 top-k
- duplicate/split 后只做必要的后处理 prune

对应位置：

- `src/Methods/FasterGSFusedDash/Model.py`

这一步的本质是让 primitive 数量增长与当前训练分辨率同步，避免在低分辨率阶段过早膨胀。 `[论文]`

而“prune-first、budget 公式、top-k 选择、再 duplicate/split”这些具体步骤，是仓库对 DashGaussian 思路的明确代码化。 `[论文落地]`

### 4.4 训练时序被重新设计

`FasterGSFusedDash` 相对 `FasterGSFused` 修改了多项默认时序：

- `DENSIFICATION_END_ITERATION`：`14900 -> 27000`
- Morton ordering 持续更久
- Morton ordering 变成“自适应触发”，只有数量足够大且增长超过阈值才做
- position learning rate decay 延后到接近全分辨率阶段

对应位置：

- `src/Methods/FasterGSFusedDash/Trainer.py`

这部分更像“训练 schedule 层”的算法修改，而不只是工程重写。

其中：

- 分辨率与 primitive 生命周期拉长、LR decay 延后，符合 DashGaussian 的调度动机。 `[论文]`
- Morton ordering 触发阈值、`27000` 这类具体时间点，更像本仓库为当前 fused 实现做的经验化调参。 `[实现补丁]`

### 4.5 为低分辨率训练新增稳定性修正

这部分是本仓库 `FasterGSFusedDash` 很重要、但论文名义上不一定完全展开的实现增量。

#### (1) 低分辨率阶段关闭 invisible momentum update

原因：

- 某些高斯在低分辨率下可能“看不见”
- 但在全分辨率下其实是应该被更新的
- 若继续沿用 invisible momentum，可能被旧动量错误推动

实现：

- `apply_invisible_momentum=(render_scale == 1)`

对应位置：

- `src/Methods/FasterGSFusedDash/Renderer.py`
- `src/Methods/FasterGSFusedDash/.../rasterization/src/backward.cu`

这一点更像仓库为“低分辨率 + fused Adam”组合带来的副作用做的工程修补。 `[实现补丁]`

#### (2) 低分辨率阶段对 invisible Gaussians 只衰减 moments，不更新参数

实现新增了 `decay_moments_invisible` kernel。

对应位置：

- `src/Methods/FasterGSFusedDash/.../rasterization/include/kernels_backward.cuh`
- `src/Methods/FasterGSFusedDash/.../rasterization/src/backward.cu`

这同样主要是实现层稳定性修正。 `[实现补丁]`

#### (3) 分辨率提升时，对 Adam moments 做重缩放

当 `render_scale` 从大变小时，梯度尺度会突变。为避免 Adam step 突然变大，代码对：

- 一阶矩按比例缩放
- 二阶矩按比例平方缩放

对应位置：

- `src/Methods/FasterGSFusedDash/Trainer.py`

“低分辨率到高分辨率切换时需要稳定优化器统计量”符合 DashGaussian 的工程需求，但这种 moments rescale 的具体做法更像仓库作者的实现性修正。 `[实现补丁]`

#### (4) 新生成 primitive 的 bias correction 修正

`FasterGSFusedDash` 在 fused backward 中增加了 cold primitive 检测：

- 若 moments 仍为零，说明是新高斯
- 不直接使用全局 step 的 bias correction
- 而是使用接近“该参数第一次更新”的修正

对应位置：

- `src/Methods/FasterGSFusedDash/.../rasterization/include/kernels_backward.cuh`

这个修改解决的是“新高斯沿用全局 step 导致更新过强/过弱”的问题。 `[实现补丁]`

### 4.6 没有明显变化的部分

`FasterGSFusedDash` 的损失函数与 `FasterGSFused` 基本一致：

- 仍然是 `L1 + DSSIM`
- 仍然记录 `PSNR`

它的主要增量不是 loss，而是 schedule 和稳定性控制。

这与 DashGaussian 的论文关注点一致。 `[论文]`

---

## 5. 结论：三个阶段分别引入了什么

### 5.1 3DGS -> FasterGSFused

核心新增/修改点：

- 参数从 `nn.Parameter` 变成 fused buffer
- Adam 更新从 Python optimizer 搬到 CUDA backward
- densification 统计从 Python 侧搬到 CUDA 侧
- 引入 Morton ordering 和更激进的工程优化
- 增加 carving、训练 cleanup、背景相关细节处理

来源判断：

- fused backward / fused update / 内存访问优化：`[论文]`
- buffer 组织、autograd 挂接、日志文件输出：`[论文落地]` 或 `[实现补丁]`
- cleanup、背景细节和部分训练配套：偏 `[实现补丁]`

一句话总结：

`FasterGSFused` 主要是在不明显改变训练目标的前提下，把 3DGS 训练过程“内核化、融合化、缓冲区化”。`

### 5.2 FasterGSFused -> FasterGSFusedDash

核心新增/修改点：

- 引入 DashGaussian 的 FFT 分辨率调度器
- 从固定全分辨率改成渐进分辨率训练
- 从普通 ADC 改成 budgeted top-k densification
- 延长 densification 生命周期并重设计 LR / Morton 时序
- 为低分辨率训练新增多项 Adam/momentum 稳定性修正

来源判断：

- 分辨率调度 + primitive 调度：`[论文]`
- scheduler 在本仓库中的接口和调用方式：`[论文落地]`
- invisible momentum 抑制、moments decay/rescale、cold primitive bias correction：偏 `[实现补丁]`

一句话总结：

`FasterGSFusedDash` 不是简单给 FasterGSFused 加一个 scheduler，而是把“训练复杂度调度”真正嵌入到了渲染、密化和 Adam 更新语义里。`

---

## 6. 最短结论

如果只看本仓库实现的“增量本质”，可以压缩成三句话：

- `GaussianSplatting`：标准 3DGS，重点是方法本身成立。
- `FasterGSFused`：不大改方法目标，重点是 fused 训练实现更快更省。
- `FasterGSFusedDash`：在 fused 实现上继续做训练复杂度调度，重点是前期低分辨率、受控 densify 和稳定性修补。
