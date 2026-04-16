# FasterGSFusedDash 改进路线图

## 1. 目标

本文档总结 `FasterGSFusedDash` 在后续迭代中最值得尝试的改进方向，目标聚焦两件事：

1. 提升训练速度
2. 在不牺牲甚至提升最终质量的前提下，让训练过程更稳定

对应当前实现：

- `src/Methods/FasterGSFusedDash`
- `src/Methods/FasterGSDash/schedule_utils.py`

---

## 2. 当前版本的强项与瓶颈

当前 `FasterGSFusedDash` 已经做到了三件关键事情：

1. 在 `FasterGSFused` 的 fused backward + fused Adam 基础上继续压缩训练时间
2. 通过 `render_scale` 把训练早期的像素复杂度降下来
3. 通过 `dash_density_control_topk()` 让 primitive 增长不再无约束膨胀

但它仍然有几个明显瓶颈：

- 分辨率调度依然偏静态，主要依赖 FFT 初始化结果
- densify 打分仍然过于依赖单一梯度指标
- 低分辨率阶段虽然已经做了不少稳定性修正，但策略仍偏保守和全局化
- CUDA 路径里仍存在一部分可继续压缩的中间缓冲开销

---

## 3. 优先级一：低风险高收益

这部分最适合优先做，因为：

- 对当前代码侵入相对有限
- 失败成本低
- 成功概率高

### 3.1 在线自适应升分辨率

### 动机

当前 `render_scale` 的主逻辑来自：

- `src/Methods/FasterGSDash/schedule_utils.py`

它的初始化主要基于训练图像频域分析，这很好，但仍然是“训练前一次性决定大框架”。  
问题在于：

- 有的场景比 FFT 预估更容易
- 有的场景细节形成更晚
- 固定日程不一定总是最优

### 建议改法

在当前 scheduler 基础上加入在线触发条件，综合以下信号：

- 最近 `K` 步 PSNR 提升幅度
- 最近 `K` 步 loss 下降幅度
- densify 命中率
- 高斯数量增长速率

触发规则示例：

- 如果低分辨率阶段收益开始明显减弱，则提前进入更高分辨率
- 如果当前阶段仍在快速收敛，则延后升分辨率

### 改动位置

- `src/Methods/FasterGSDash/schedule_utils.py`
- `src/Methods/FasterGSFusedDash/Trainer.py`

### 预期收益

- 速度：中等提升
- 质量：通常不降，某些场景可能提升
- 风险：低

---

### 3.2 densify 打分从单一 mean gradient 变成混合分数

### 动机

当前 `dash_density_control_topk()` 主要依赖：

- `densification_info[1] / densification_info[0]`

这本质上还是一个“平均梯度驱动”的策略。  
问题在于：

- 薄结构和小物体不一定总能在低分辨率阶段得到足够高的梯度
- 某些区域虽然梯度高，但未必值得优先 densify

### 建议改法

把当前打分改成混合分数，例如：

- mean gradient
- 最近可见次数
- opacity 风险
- 局部残差
- tile 覆盖变化率

可先做简单线性组合，不必一开始就引入复杂学习策略。

### 改动位置

- `src/Methods/FasterGSFusedDash/Model.py`

### 预期收益

- 质量：高概率提升
- 速度：变化不大，可能略升
- 风险：低到中

---

### 3.3 复用 backward 临时缓冲区

### 动机

当前 CUDA 路径里仍有每步重新分配的中间张量，例如：

- `grad_colors`
- `grad_opacities`
- `grad_mean2d_helper`
- `grad_conic_helper`

这类对象在每步训练里重复创建，容易带来：

- 显存碎片
- 分配/释放开销
- allocator 噪音

### 建议改法

为这些张量引入 persistent workspace：

- 张量大小不变时直接复用
- primitive 数量变化时按需扩容

### 改动位置

- `src/Methods/FasterGSFusedDash/FasterGSFusedCudaBackend/.../rasterization/src/rasterization_api.cu`

### 预期收益

- 速度：小到中等提升
- 质量：无直接影响
- 风险：低

---

## 4. 优先级二：中风险中收益

### 4.1 低分辨率全图 + 少量高分辨率 patch 混合监督

### 动机

当前训练是整张图统一使用 `render_scale`。  
这对大结构很高效，但容易伤到：

- 细边界
- 小物体
- 高频纹理

### 建议改法

训练时保留当前全图低分辨率路径，同时额外采样少量高残差 patch：

1. 全图仍按低分辨率训练，保证速度
2. 每步增加少量高分辨率 patch 监督，补高频细节

### 改动位置

- `src/Methods/FasterGSFusedDash/Trainer.py`
- 可能还需要扩展 renderer 支持局部 patch 渲染

### 预期收益

- 质量：明显有潜力提升
- 速度：会损失一些，但通常可控
- 风险：中

---

### 4.2 invisible momentum 从二值开关变成连续控制

### 动机

当前逻辑是：

- `render_scale == 1` 时允许 invisible momentum
- 否则关闭

这个规则稳定，但偏保守。  
它的潜在问题是：

- 某些高斯虽然低分辨率阶段未触碰 tile，但完全关闭动量可能让收敛变慢

### 建议改法

把 invisible momentum 改成连续权重控制，例如按以下信号调节：

- 当前 `render_scale`
- 距离上次可见的步数
- 历史可见频率

比如：

- 分辨率越低，invisible momentum 权重越小
- 长时间不可见的 primitive 动量衰减更快

### 改动位置

- `src/Methods/FasterGSFusedDash/Renderer.py`
- `src/Methods/FasterGSFusedDash/FasterGSFusedCudaBackend/.../rasterization/src/backward.cu`
- `src/Methods/FasterGSFusedDash/FasterGSFusedCudaBackend/.../rasterization/include/kernels_backward.cuh`

### 预期收益

- 速度：可能小幅提升
- 质量：有机会提升收敛稳定性
- 风险：中

---

### 4.3 SH 解锁更细粒度

### 动机

当前 SH 解锁已经比 `FasterGSFused` 更谨慎，但仍然是相对全局的阶段控制。  
可以进一步利用：

- 残差
- 分辨率状态
- 视角覆盖程度

来决定何时解锁更高阶 SH。

### 建议改法

把当前“是否 near full resolution”扩展成更丰富的条件，例如：

- render_scale 足够小
- 最近图像残差下降停滞
- 可见区域统计达到阈值

### 改动位置

- `src/Methods/FasterGSFusedDash/Trainer.py`
- `src/Methods/FasterGSDash/schedule_utils.py`

### 预期收益

- 质量：中等提升潜力
- 速度：小影响
- 风险：中

---

## 5. 优先级三：高风险高潜力

### 5.1 学习型调度器

### 动机

当前所有调度规则本质上仍是手工设计的启发式。  
如果想进一步突破，最自然的方向是让调度器直接根据训练过程学会：

- 什么时候升分辨率
- 什么时候加快或放缓 densification

### 建议改法

输入在线统计量：

- PSNR 变化
- loss 变化
- primitive 增长速率
- 可见高斯比例
- densify 命中数

输出：

- 下一阶段 `render_scale`
- 下一阶段 `densify_rate`

一开始可以用简单规则拟合，再考虑学习型方法。

### 风险

- 实现复杂
- 实验成本高
- 容易过拟合特定数据集

---

### 5.2 区域级 primitive budget

### 动机

当前 budget 是全局一个数。  
但场景不同区域复杂度差异很大：

- 平坦背景不需要过早 densify
- 高频结构区则需要更快增长 primitive

### 建议改法

将 densify budget 从全局扩展到区域级或 tile 级：

- 每个区域单独维护复杂度预算
- 高频区域优先 densify
- 简单区域严格受控

### 风险

- 需要额外的数据结构
- 可能引入较大工程复杂度

---

### 5.3 更深的 CUDA fusion

### 动机

尽管当前已经 fused 得很多，但仍存在一些明显可继续融合的点，例如：

- 某些 helper gradient 张量还能合并
- 某些 kernel 之间的中间结果还可以直接复用

### 建议改法

沿着现有 TODO 和中间缓冲路径继续压缩 backward：

- 减少 helper tensor
- 减少 kernel 边界
- 更强地复用 shared / temporary buffers

### 风险

- 开发成本高
- 调试复杂
- 对质量无直接帮助，主要收益是速度

---

## 6. 我建议的执行顺序

如果按“投入产出比”排序，我建议这样做：

1. 在线自适应升分辨率
2. 混合 densify 打分
3. backward 临时缓冲区复用
4. patch 混合监督
5. invisible momentum 连续化
6. SH 解锁更细粒度
7. 更深层 CUDA fusion
8. 区域级 budget
9. 学习型调度器

---

## 7. 评估方法

后续做任何改动，都建议统一记录下面这些指标。

### 7.1 速度指标

- 总训练时长
- 每 1000 iter 平均 step time
- 峰值 VRAM

### 7.2 质量指标

- PSNR
- SSIM
- LPIPS
- 最终高斯数量

### 7.3 过程指标

- `render_scale` 变化节点
- 每次 densify 命中数
- 每次 prune 数量
- 可见高斯比例
- scheduler 输出的 densify_rate

这些过程指标很重要，因为很多改动的收益不一定立即体现在最终 PSNR 上，但会明显改变训练轨迹。

---

## 8. 最终判断

我的总体判断是：

- `FasterGSFusedDash` 已经是一个很强的“第一版工程解”
- 它最值得继续挖的不是再做一个小 kernel trick
- 而是把当前这些全局、静态、手工启发式规则，逐步改成更自适应、更细粒度的调度策略

最值得优先尝试的两件事是：

1. 在线自适应分辨率调度
2. 更好的 densify 打分函数

如果只允许选一个起点，我会先做：

> 在线自适应升分辨率

因为它对现有代码侵入最小、最容易验证，而且同时对速度和质量都有机会带来正收益。

