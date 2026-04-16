# FasterGSFusedDash 相比 FasterGSFused 的具体改进

## 1. 范围说明

本文只分析本仓库中的两个实现：

- `src/Methods/FasterGSFused`
- `src/Methods/FasterGSFusedDash`

目标不是泛泛比较，而是回答一个更具体的问题：

> 在已经有 fused backward + fused Adam 的 `FasterGSFused` 基础上，`FasterGSFusedDash` 又额外改了什么？

一句话先概括：

- `FasterGSFused` 的核心是“把 3DGS 训练执行路径融合起来”
- `FasterGSFusedDash` 的核心是“在 fused 执行路径上，再把训练复杂度做成可调度对象”

也就是说，`FasterGSFusedDash` 的新增重点不是再做一次 kernel fusion，而是：

1. 在训练早期降低渲染分辨率
2. 控制 primitive 数量增长速度
3. 为低分辨率训练引入一组新的稳定性修正

---

## 2. 最核心的变化：从固定复杂度训练，改成可调度复杂度训练

`FasterGSFused` 训练时每一步都按固定分辨率渲染，也按标准 Faster-GS 节奏 densify。

而 `FasterGSFusedDash` 引入了一个新的调度器：

- 文件：`src/Methods/FasterGSDash/schedule_utils.py`
- 类：`TrainingScheduler`

它显式控制两个量：

1. `render_scale`
   - 当前训练步使用的整数下采样倍率
   - `1` 表示全分辨率
2. `densify_rate`
   - 当前 densification 步允许的 primitive 增长比例

这两个量在 `src/Methods/FasterGSFusedDash/Trainer.py` 中被接入训练循环。

### 2.1 训练循环怎么变了

`FasterGSFused` 的训练循环在 `src/Methods/FasterGSFused/Trainer.py` 中，逻辑是：

1. 固定全分辨率渲染
2. 计算 loss
3. 反向传播时由 fused CUDA 路径做参数更新

`FasterGSFusedDash` 在 `src/Methods/FasterGSFusedDash/Trainer.py` 中新增了：

- `self.dash_scheduler`
- `self.current_render_scale`
- `DASH` 配置块

训练中会先根据 scheduler 取当前 `render_scale`，再：

1. 用低分辨率相机参数去渲染
2. 把 GT 同步下采样到相同分辨率
3. 再计算 loss

这意味着它削减的是“每一步训练本身的计算量”，而不只是减少步数。

---

## 3. 改进一：引入渐进分辨率训练

这是 `FasterGSFusedDash` 相比 `FasterGSFused` 最直观、也是最重要的新增。

### 3.1 FasterGSFused 的做法

在 `src/Methods/FasterGSFused/Renderer.py` 中，训练渲染接口 `render_image_training()` 使用的是原始相机参数：

- `width`
- `height`
- `focal_x`
- `focal_y`
- `center_x`
- `center_y`

换句话说，训练阶段总是按完整分辨率渲染。

### 3.2 FasterGSFusedDash 的做法

在 `src/Methods/FasterGSFusedDash/Renderer.py` 中，`extract_settings()` 多了一个参数：

- `render_scale`

并且会把相机参数按比例缩放：

- `width // render_scale`
- `height // render_scale`
- `focal_x / render_scale`
- `focal_y / render_scale`
- `center_x / render_scale`
- `center_y / render_scale`

这说明它不是“先全分辨率渲染，再在图像上缩小”，而是从投影几何开始就真正按低分辨率执行。

### 3.3 GT 监督也同步降分辨率

在 `src/Methods/FasterGSFusedDash/Trainer.py` 中，GT 不再直接拿来算 loss，而是在 `render_scale > 1` 时用：

- `torch.nn.functional.interpolate(..., mode='area')`

做抗混叠下采样。

这一步很关键，因为如果只把渲染结果缩小，而 GT 仍保持原分辨率，loss 的定义就不一致了。

### 3.4 这带来的收益

它减少的不是少量 bookkeeping，而是训练早期最核心的像素级开销：

- forward rasterization
- backward blending
- 像素相关中间缓存

因此在 primitive 还不多、场景结构还没形成的早期，用低分辨率训练通常是划算的。

---

## 4. 改进二：densification 从“自由增长”改成“带预算增长”

`FasterGSFused` 和 `FasterGSFusedDash` 都会 densify，但增长控制逻辑不一样。

### 4.1 FasterGSFused 的 densify

在 `src/Methods/FasterGSFused/Model.py` 中，核心接口是：

- `adaptive_density_control()`

逻辑仍然接近 Faster-GS 标准风格：

1. 根据 densification_info 判断哪些高斯过阈值
2. 小高斯 duplicate，大高斯 split
3. 最后再 prune

本质上，它更像“谁符合条件就 densify”，增长相对自由。

### 4.2 FasterGSFusedDash 的 densify

在 `src/Methods/FasterGSFusedDash/Model.py` 中，新增：

- `dash_density_control_topk()`

它相比 `adaptive_density_control()` 多了三层控制：

1. 先按 scheduler 给出的 `densify_rate` 算出本轮 budget
2. 即使很多高斯都超过阈值，也只保留 top-k
3. `momentum_add` 会返回给 scheduler，用来更新未来的 primitive 预算上限

这就把 primitive 增长从“局部阈值驱动”改成了“全局预算驱动 + 局部阈值筛选”。

### 4.3 训练器如何接入 budget

在 `src/Methods/FasterGSFusedDash/Trainer.py` 中：

1. 先取当前高斯数量 `n_gaussians`
2. 调 `self.dash_scheduler.get_densify_rate(...)`
3. 把结果传给 `dash_density_control_topk(...)`
4. 再把 `momentum_add` 回写给 `self.dash_scheduler.update_momentum(...)`

这说明 `FasterGSFusedDash` 中的 densification 不再只是 model 内部局部决策，而是变成“trainer 与 scheduler 联合驱动”的过程。

### 4.4 为什么这很重要

低分辨率训练的一个典型风险是：

- 一开始分辨率低
- 但 densification 仍然疯狂增长
- 结果 primitive 数量很早就膨胀
- 前期省下来的像素开销又被 primitive 开销吃回去了

`FasterGSFusedDash` 就是在解决这个问题：

> 分辨率降低时，primitive 增长也要同时受控。

---

## 5. 改进三：训练时序整体后移并重新组织

`FasterGSFusedDash` 不是只加了一个 scheduler，它还系统性改了训练时间表。

### 5.1 densification 生命周期更长

`FasterGSFused` 默认：

- `DENSIFICATION_END_ITERATION = 14900`

`FasterGSFusedDash` 改成：

- `DENSIFICATION_END_ITERATION = 27000`

原因很直接：

- 既然前期采用低分辨率训练
- 那么 primitive 成长也需要覆盖更长的训练阶段
- 否则到分辨率升上来之后，densification 已经结束，会错过高频细节形成阶段

### 5.2 Morton ordering 生命周期也更长

`FasterGSFusedDash` 还把：

- `MORTON_ORDERING_END_ITERATION`

同步延长到 `27000`。

这说明作者认为：

- primitive 数量的主要变化期被延长了
- 因此数据重排带来的访存收益也需要跟着延长

### 5.3 学习率衰减延后

在 `src/Methods/FasterGSFused/Trainer.py` 中，mean learning rate 直接按训练步数衰减。

而在 `src/Methods/FasterGSFusedDash/Trainer.py` 中，改成：

- 通过 `self.dash_scheduler.lr_decay_from_iter()` 计算一个更晚的衰减起点

也就是说：

- 低分辨率阶段不急着让 position LR 衰减
- 要先让场景结构在低成本阶段形成起来
- 等接近全分辨率后再进入更精细的衰减段

这是一个很典型的“调度感知型学习率策略”。

---

## 6. 改进四：为低分辨率训练新增稳定性修正

这部分是 `FasterGSFusedDash` 最容易被忽略、但实际非常重要的改进。

原因在于：

- `FasterGSFused` 的 fused Adam 语义默认是围绕全分辨率训练设计的
- 一旦引入低分辨率阶段，一些原本合理的 update 规则会开始失真

所以 `FasterGSFusedDash` 新增了一组稳定性修正。

### 6.1 关闭低分辨率阶段的 invisible momentum update

在 `src/Methods/FasterGSFused/Renderer.py` 中，`diff_rasterize()` 默认允许对“当前 touched tiles 为 0 的高斯”继续应用 invisible momentum。

在全分辨率下，这通常没问题，因为：

- 真正不可见的高斯可以沿动量继续小幅更新

但在低分辨率下会出问题：

- 有些高斯只是因为分辨率低，所以当前没碰到 tile
- 它们在全分辨率下其实是可见的
- 如果继续按 invisible primitive 的逻辑更新，会产生漂移

因此在 `src/Methods/FasterGSFusedDash/Renderer.py` 中：

- `apply_invisible_momentum=(render_scale == 1)`

只有在全分辨率时才保留这一逻辑。

### 6.2 低分辨率阶段对 invisible primitives 只衰减 moments，不推进参数

对应实现：

- `src/Methods/FasterGSFusedDash/.../rasterization/include/kernels_backward.cuh`
- `src/Methods/FasterGSFusedDash/.../rasterization/src/backward.cu`

新增了：

- `decay_moments_invisible`

逻辑是：

- 对当前没触碰 tile 的 primitive
- 一阶矩和二阶矩继续衰减
- 但参数本身不跟着更新

这能避免“旧动量在低分辨率阶段把本不该动的高斯推偏”。

### 6.3 分辨率变化时，重缩放 Adam moments

在 `src/Methods/FasterGSFusedDash/Trainer.py` 中新增：

- `_rescale_moments_on_resolution_change()`

当 `render_scale` 从大变小，也就是训练分辨率提升时：

- 一阶矩按比例缩放
- 二阶矩按比例平方缩放

原因是：

- 分辨率变高后，梯度统计量尺度也会变化
- 如果继续沿用旧 moments，Adam 的有效 step size 会突然不平滑

这个修正的目的，就是让分辨率切换时优化器行为更连续。

### 6.4 对新生成 primitive 的 bias correction 做特殊处理

在 `src/Methods/FasterGSFusedDash/.../rasterization/include/kernels_backward.cuh` 中，新增了 cold primitive 检测：

- 如果某个 primitive 的 moments 还是零
- 说明它是刚 densify 出来的新高斯
- 这时不直接使用全局 `adam_step_count` 的 bias correction
- 而是用更接近“该参数第一次更新”的修正

否则会出现一个问题：

- 新高斯虽然刚出现
- 但却继承了一个已经很大的全局步数
- 这会让第一步更新幅度不符合 Adam 的正常语义

这个修正对带动态 densification 的 fused 训练尤其重要。

---

## 7. 改进五：SH 解锁和 Morton ordering 也被调度感知化

这部分不如前面显眼，但也反映了 `FasterGSFusedDash` 的整体思路。

### 7.1 SH degree 不再一味按固定步数放开

在 `FasterGSFused` 中，SH degree 每 1000 iter 提升一次。

而在 `FasterGSFusedDash` 中，虽然仍有 1000 iter 的节奏，但实际会先检查：

- `self.dash_scheduler.near_full_resolution()`

只有接近全分辨率，才真正放开更高阶 SH。

原因是：

- 低分辨率阶段本来就缺少高频监督
- 这时太早引入高阶 view-dependent 表达，容易带来噪声

### 7.2 Morton ordering 也被限制在“值得做”的场景

`FasterGSFusedDash` 增加了：

- `MORTON_ORDERING_MIN_GAUSSIANS`

在高斯数量太少时直接跳过 Morton ordering。  
这体现出一个很实际的工程判断：

- 重排本身也有成本
- 在小规模 primitive 阶段，不一定值回票价

---

## 8. 如果把改进压缩成最关键的三条

如果只保留最本质的差异，可以把 `FasterGSFusedDash` 相比 `FasterGSFused` 的改进压缩成三条：

### 8.1 训练复杂度不再固定

`FasterGSFused`：每一步固定按全分辨率和常规 densification 训练  
`FasterGSFusedDash`：每一步的像素复杂度和 primitive 增长速度都由 scheduler 控制

### 8.2 densification 从“局部触发”变成“全局预算 + 局部筛选”

`FasterGSFused`：谁过阈值谁 densify  
`FasterGSFusedDash`：先由 scheduler 给总预算，再在候选中做 top-k densify

### 8.3 为低分辨率阶段补齐优化器语义

`FasterGSFused`：优化器行为默认围绕全分辨率训练  
`FasterGSFusedDash`：增加 invisible momentum 抑制、moment decay、moment rescale、cold primitive bias correction

---

## 9. 结论

`FasterGSFusedDash` 不是简单的：

> “在 FasterGSFused 上面套一个分辨率 scheduler”

更准确地说，它做了三层递进：

1. 用 scheduler 改写训练早中晚期的计算量分布
2. 用 budgeted densification 让 primitive 增长和分辨率同步
3. 用一组新的 optimizer/stability 修正，让低分辨率训练不会破坏 fused Adam 的语义

所以从代码角度看，`FasterGSFusedDash` 相比 `FasterGSFused` 的真正新增不是“多了一个模块”，而是：

> 把训练复杂度调度正式嵌入到了 renderer、trainer、model 以及 CUDA backward 的更新规则里。

