# Task Plan

## Goal

Merge FastGS-style multi-view consistent densification/pruning into `src/Methods/FasterGSFusedfast` while preserving the FasterGSFused fused training path and CUDA rasterizer optimizations.

## Scope

- Work only in `src/Methods/FasterGSFusedfast`
- Keep FasterGSFused as the execution baseline
- Add FastGS-inspired VCD/VCP scoring and scheduling
- Avoid replacing the existing fused rasterizer main path unless needed for score-only rendering

## Phases

| Phase | Status | Description |
|------|--------|-------------|
| 1 | complete | Create local plan files and inspect integration points |
| 2 | complete | Fix copied package imports and module references in `FasterGSFusedfast` |
| 3 | complete | Add score-only rasterization API path in the CUDA backend |
| 4 | complete | Add Python multi-view scoring utilities and renderer hooks |
| 5 | complete | Replace/extend current ADC scheduling with FastGS-style VCD/VCP in trainer/model |
| 6 | complete | Sanity-check structure, imports, and summarize resulting merged design |

## Intended Design

1. Use `FasterGSFusedfast` as the runtime/training base.
2. Keep fused backward and fused Adam updates for normal training renders.
3. Add a no-update score-forward path that accepts a metric map and returns per-Gaussian counts.
4. Compute FastGS-style multi-view importance/pruning scores in Python by sampling training views.
5. Use existing gradient-based densification info only as candidate filtering, and use the new multi-view scores for actual FastGS-style densify/prune decisions.

## Risks

- Score renders must never trigger parameter updates.
- Backend API changes must stay consistent across Python bindings, C++ wrappers, and CUDA code.
- The copied method currently still imports from `Methods.FasterGSFused`; those references must be corrected before integration.
- This round only completed Python static validation; the CUDA extension still needs a real import/build/runtime pass to validate the new score kernel end to end.
- Even after the logic merge, FastGS threshold transfer may still drift if fused screen-space gradient magnitudes are numerically different from original FastGS.
- The new score-only kernel is logically aligned with FastGS, but metric-count equivalence has not yet been confirmed on real scenes.
- Large-Gaussian pruning still uses a world-space safeguard instead of original FastGS `max_radii2D`, so pruning behavior can differ on near-camera splats.
- Random view sampling and stochastic prune-budget sampling will make behavior comparisons noisy unless runs are seeded.

## Open Questions

- Whether the score-only path should reuse the existing forward buffers directly or allocate a slimmed-down score variant.
- Whether to keep original densification info accumulation in parallel with FastGS scoring for candidate selection.

## Decision Points To Review

### D1. Training Render vs Score Render

- Conflict:
  - FasterGSFused training render updates parameters during backward.
  - FastGS multi-view scoring must never update parameters.
- Proposed decision:
  - Add a strict no-update score-only render path.
  - Keep the fused training render path unchanged for normal optimization.
- Confirmed decision:
  - Use an independent score-only render API.
  - Training render and score render are separate execution paths.

### D2. Replace ADC vs Hybrid ADC

- Conflict:
  - FasterGSFused currently uses gradient-driven ADC via `densification_info`.
  - FastGS uses multi-view consistent VCD/VCP for densify/prune decisions.
- Proposed decision:
  - Keep current gradient statistics only for candidate filtering.
  - Use FastGS-style multi-view scores as the final densify/prune criterion.
- Confirmed decision:
  - Follow FastGS original densify/prune logic as closely as practical.
  - Existing fused gradient statistics remain only where needed to support FastGS-style candidate selection and clone/split handling.
  - `FasterGSFusedfast`'s current ADC must not remain the dominant policy.

### D3. Rasterizer Ownership

- Conflict:
  - FastGS has its own rasterizer-side optimization ideas, including compact-box style logic.
  - FasterGSFused already has exact tile overlap, separate sorting, and fused backward.
- Proposed decision:
  - Keep the FasterGSFused rasterizer as the sole rendering backend.
  - Do not merge FastGS compact-box code unless a concrete gap is found.
- Confirmed decision:
  - `FasterGSFusedfast` is the only rendering/training backend.
  - FastGS contributes policy, not a second rasterizer stack.

### D4. Where To Compute FastGS Scores

- Conflict:
  - Python-side reconstruction of Gaussian footprint statistics would be simpler conceptually.
  - CUDA-side counting is more faithful and much faster.
- Proposed decision:
  - Compute per-Gaussian high-error pixel counts in CUDA forward.
  - Expose them through a dedicated score-only backend API.
- Confirmed decision:
  - Python builds per-view `metric_map`.
  - CUDA score-forward computes per-Gaussian footprint counts.

### D5. Scoring Buffer Strategy

- Conflict:
  - Reusing current forward buffers reduces code churn.
  - A slim score-only path could reduce overhead but needs more backend changes.
- Proposed decision:
  - First implementation should reuse the current forward pipeline where possible.
  - Only optimize into a slimmer score path if overhead becomes problematic.
- Confirmed decision:
  - Implement a dedicated score-only API.
  - Internally reuse the existing forward pipeline as much as possible instead of building a second full rasterizer path.

### D6. Pruning Score Definition

- Conflict:
  - FasterGSFused already has gradient-based densification signals.
  - FastGS pruning is defined from multi-view high-error counts weighted by photometric loss.
- Proposed decision:
  - Keep FastGS pruning score semantics intact.
  - Do not mix pruning score with existing gradient statistics beyond candidate gating.
- Confirmed decision:
  - Preserve FastGS VCD/VCP score semantics.
  - Do not blend pruning score with FasterGSFused-specific gradient heuristics.

### D7. Scheduling Frequency

- Conflict:
  - Frequent multi-view scoring improves control over Gaussians.
  - It also eats into FasterGSFused's training speed advantage.
- Proposed decision:
  - Run multi-view scoring only on densification/pruning intervals, not every iteration.
- Confirmed decision:
  - Follow FastGS-style interval-triggered scoring.
  - Use FastGS base-style densification/pruning intervals rather than per-iteration scoring.

### D8. Late-Stage Pruning

- Conflict:
  - Current FasterGSFused cleanup is mostly opacity/degeneracy-based.
  - FastGS benefits from a separate late-stage aggressive multi-view prune.
- Proposed decision:
  - Add a dedicated late-stage VCP pruning callback after densification ends.
- Confirmed decision:
  - Keep a separate late-stage aggressive VCP prune schedule.

### D9. Invisible Momentum During Scoring

- Conflict:
  - FasterGSFused can apply momentum-only Adam updates to invisible Gaussians.
  - Score renders must be observational only.
- Proposed decision:
  - Disable all update and momentum behavior in score-only renders.
- Confirmed decision:
  - Score-only renders may write score outputs only.
  - They must not update parameters, moments, densification statistics, or any training state.

### D10. FastGS Dual-Gradient ADC vs Current Fused Densification Buffer

- Conflict:
  - FastGS clone/split selection depends on two distinct screen-space gradient statistics:
    - signed mean2D gradient accumulation for clone candidates
    - absolute mean2D gradient accumulation for split candidates
  - `FasterGSFusedfast` currently stores only a compact `(2, N)` densification buffer:
    - update count
    - one scalar gradient magnitude
- Options:
  - Option A: keep the current scalar statistic and reuse it for both clone and split.
    - Reason: smallest code change.
    - Consequence: clone/split stop matching FastGS semantics; `grad_abs_thresh` becomes mostly cosmetic.
  - Option B: extend densification stats to keep the full FastGS-style signals.
    - Reason: preserves FastGS ADC behavior much more faithfully.
    - Consequence: requires backend backward changes and model-side buffer/layout changes.
  - Option C: approximate the second signal from another fused quantity.
    - Reason: less backend work than Option B.
    - Consequence: behavior becomes harder to reason about and tune; threshold transfer from FastGS gets weaker.
- Recommendation:
  - Choose Option B.
- Confirmed decision:
  - Extend fused densification stats so clone and split can use separate FastGS-style gradient signals.
  - Practical target layout: count + signed x/y accumulators + absolute x/y accumulators.

### D11. FastGS Screen-Space Large-Gaussian Pruning vs Current Fused Observability

- Conflict:
  - FastGS tracks `max_radii2D` during training and can prune excessively large Gaussians in screen space.
  - `FasterGSFusedfast` training path currently does not expose per-Gaussian radii back to Python/model state.
- Options:
  - Option A: add a new radii export path to the fused backend and replicate FastGS screen-space pruning exactly.
    - Reason: most faithful to FastGS.
    - Consequence: more intrusive backend/API change on top of the current merge.
  - Option B: keep current fused world-space large-Gaussian pruning only.
    - Reason: preserves existing fused behavior and keeps the merge focused on VCD/VCP ownership.
    - Consequence: large-screen splats are controlled less aggressively than in original FastGS.
  - Option C: approximate screen size from score-only renders only.
    - Reason: avoids changing the training render API.
    - Consequence: pruning signal becomes sparse and less aligned with FastGS's per-iteration tracking.
- Recommendation:
  - Choose Option B for this merge.
- Confirmed decision:
  - Keep FastGS ownership over multi-view VCD/VCP scoring and densify/prune scheduling.
  - Retain fused world-space large-Gaussian pruning as the large-splat safety check unless a later pass explicitly adds radii export.
