# Findings

## 2026-04-16

- `src/Methods/FasterGSFusedfast` is currently identical to `src/Methods/FasterGSFused` at the file level.
- The copied method still contains stale imports pointing to `Methods.FasterGSFused` instead of `Methods.FasterGSFusedfast`.
- The current training path already fuses backward and optimizer updates in CUDA:
  - Python entry: `src/Methods/FasterGSFusedfast/Renderer.py`
  - Autograd wrapper: `.../torch_bindings/rasterization.py`
  - CUDA backward update path: `.../rasterization/src/backward.cu`
- The current densification logic is vanilla/gradient-based and uses a `(2, N)` `densification_info` buffer updated in CUDA backward.
- FastGS integration should target:
  - A new score-only rasterization path in the backend
  - Python multi-view camera sampling and score aggregation
  - Trainer/model densification and pruning scheduling
- Main merge contradictions have been identified and turned into explicit decision points:
  - training render vs score-only render separation
  - replace-vs-hybrid ADC design
  - rasterizer ownership
  - CUDA-vs-Python score computation
  - score-buffer reuse strategy
  - pruning score semantics
  - scoring/pruning schedule
  - late-stage VCP callback
  - invisible momentum disabled for scoring
- The reviewed direction is now fixed:
  - `FasterGSFusedfast` remains the sole execution backend.
  - FastGS controls Gaussian count through VCD/VCP.
  - Densify/prune behavior should track FastGS as closely as practical.
  - Score rendering is a separate no-update path.
- FastGS's ADC is not just "one more threshold":
  - clone candidates use averaged signed screen-space mean gradients
  - split candidates use averaged absolute screen-space mean gradients
  - therefore the fused backend buffer had to grow beyond the original `(2, N)` layout
- The merged implementation now stores FastGS-style densification stats in fused backward as:
  - update count
  - signed x/y accumulators
  - absolute x/y accumulators
- The merged score path is now implemented in the fused backend as a dedicated score-only forward:
  - it reuses fused preprocess + depth/tile sorting
  - it runs a separate `blend_score` kernel that atomically accumulates FastGS metric counts per Gaussian
  - it does not update params, moments, densification stats, or invisible momentum
- The model-side merge now follows FastGS more closely:
  - FastGS-style clone/split separation is implemented
  - interval-triggered VCD/VCP scoring drives densification/pruning
  - late-stage FastGS prune callback is implemented
  - opacity capping after densify/prune follows FastGS
- One deliberate compromise remains:
  - original FastGS uses `max_radii2D` for large-screen pruning
  - the current fused merge keeps the existing world-space large-Gaussian pruning guard because radii export is still absent from the training path

## Implementation Risks

### R1. ADC Threshold Scale May Drift From Original FastGS

- Current status:
  - The merged backend now records FastGS-style signed/absolute screen-space gradient accumulators.
  - The trainer also reuses FastGS-like `grad_threshold` / `grad_abs_threshold` values.
- Risk:
  - The fused backend's mean2D gradient scale may still differ from original FastGS rasterizer numerics.
  - If the scale is off, clone/split frequency will drift even though the logic shape matches FastGS.
- Likely consequence:
  - Over-densification, under-densification, or wrong clone-vs-split balance.
- Validation needed:
  - Compare densify event counts and Gaussian-count curves against original FastGS on a short run.

### R2. Score-Only Metric Counts Are Semantically Close, Not Yet Runtime-Verified Equivalent

- Current status:
  - The score path reuses fused preprocess + sorting and a dedicated `blend_score` kernel.
- Risk:
  - Metric-count accumulation may not match original FastGS exactly on thin structures, deep occlusion, or alpha-threshold edge cases.
- Likely consequence:
  - VCD/VCP ranking can shift, which changes which Gaussians get cloned, split, or pruned.
- Validation needed:
  - Compare `importance_score` / `pruning_score` distributions on identical sampled views.

### R3. Large-Gaussian Pruning Still Uses World-Space Guard, Not FastGS `max_radii2D`

- Current status:
  - The merged implementation intentionally kept the fused world-space large-Gaussian safeguard.
- Risk:
  - Screen-space oversized splats may survive longer than they would in original FastGS.
- Likely consequence:
  - Near-camera blur blobs or view-dependent oversized splats may prune later than desired.
- Validation needed:
  - Visual check on scenes with close-up cameras or large foreground splats.

### R4. Randomized Scoring/Pruning Adds Variance During Tuning

- Current status:
  - View sampling for score aggregation is random.
  - Prune selection still uses stochastic budgeted sampling.
- Risk:
  - Repeated runs can diverge enough to obscure whether a regression comes from logic or randomness.
- Likely consequence:
  - Harder ablation and threshold tuning.
- Validation needed:
  - Fix random seeds when first comparing against FastGS baseline.

### R5. Only Python Static Validation Is Done So Far

- Current status:
  - `python3 -m compileall` passed.
- Risk:
  - CUDA extension symbols, kernel launches, tensor dtypes, and runtime behavior are not yet proven end to end.
- Likely consequence:
  - First real import/build/run may still expose compile or runtime issues.
- Validation needed:
  - Force extension rebuild/import, then run a short training smoke test.
