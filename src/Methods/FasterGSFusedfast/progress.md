# Progress

## 2026-04-16 Session

- Initialized planning files in project root as requested by the user.
- Confirmed `FasterGSFusedfast` exists and is a direct copy of `FasterGSFused`.
- Confirmed the copied package still uses stale imports referencing `Methods.FasterGSFused`.
- Confirmed the merge should be implemented by preserving FasterGSFused's fused training path and adding FastGS scoring as a separate no-update path.
- Added explicit merge decision points to `task_plan.md` so the remaining design can be reviewed item by item before code changes continue.
- Wrote the confirmed decision outcomes into the planning files:
  - FasterGSFusedfast backend ownership is fixed.
  - FastGS governs densify/prune policy.
  - Score-only rendering is separate and side-effect free.

## Next

- Trigger a real extension rebuild/import to validate the new CUDA score kernel end to end.
- Run a short training smoke test once the user wants runtime verification.

## 2026-04-16 Implementation Update

- Expanded `task_plan.md` with the remaining real merge conflicts:
  - FastGS dual-gradient ADC vs fused compact densification buffer
  - FastGS `max_radii2D` pruning vs fused lack of radii export
- Completed the `FasterGSFusedfast` merge implementation:
  - model-side FastGS clone/split/prune methods were added
  - trainer-side FastGS densify/final-prune callbacks now target real model methods
  - fused backward densification stats were extended from `(2, N)` to FastGS-style signed/absolute gradient accumulators
  - fused backend score-only forward path was implemented with a dedicated `blend_score` kernel
- Cleaned remaining copied naming issues in touched files.
- Ran `python3 -m compileall /home/ubuntu/codes/nerficg_sht/src/Methods/FasterGSFusedfast` successfully.
- Remaining validation gap:
  - the CUDA backend source is patched, but this session has not yet forced a real extension rebuild/import and runtime execution.
- Wrote the current implementation risk assessment back into the local docs:
  - gradient-threshold scale mismatch risk
  - score-kernel equivalence risk
  - missing `max_radii2D` screen-space prune risk
  - randomness/variance risk
  - runtime validation gap
