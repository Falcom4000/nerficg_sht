# Repository Guidelines

## Project Structure & Module Organization
This repository is a 3DGS-focused research and development codebase. Core algorithms live in `src/Methods`, with one subdirectory per method such as `GaussianSplatting`, `FasterGSDash`, and `FasterGSFusedDash`. Shared training infrastructure is in `src/Framework.py`, `src/Optim`, `src/Datasets`, `src/Cameras`, and `src/Visual`. CUDA and extension code lives under `src/CudaUtils` and method-specific backend folders. Entry-point scripts are in `scripts/`, reusable configs in `configs/`, environment files in `environments/`, and experiment notes/results in `experiments/`. Treat `dataset/` and `output/` as local runtime data, not source.

## Build, Test, and Development Commands
Create the base environment with `conda env create -f environments/py311_cu128.yaml` and activate it with `conda activate nerficg`. Install method-specific extensions with `python scripts/install.py -m <METHOD_NAME>`. Generate a config with `python scripts/create_config.py -m FasterGSDash -d MipNeRF360 -o my_run`. Train with `python scripts/train.py -c configs/my_run.yaml`. Run batch training with `python scripts/sequential_train.py -d configs/test_dash_1`. Render or evaluate a trained model with `python scripts/inference.py -h`. Use `python scripts/benchmark.py -c <config>` or `bash scripts/benchmark.sh --methods fastergsdash --scenes bonsai` for repeatable comparisons.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for functions/variables, PascalCase for classes, and concise module docstrings. Keep new methods under `src/Methods/<MethodName>/` and match config-facing names to directory/class names. Preserve the current import pattern used by scripts (`with utils.DiscoverSourcePath():`). There is no repo-wide formatter config, so keep changes consistent with neighboring files and avoid large style-only diffs.

## Testing Guidelines
There is no dedicated `tests/` suite today. Validate changes with the smallest config or scene that exercises the touched path, then record metrics when behavior or performance changes. For algorithm work, prefer a reproducible command, relevant config path, and before/after PSNR, SSIM, LPIPS, timing, or VRAM numbers.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects, often with version tags, for example `V8: FP16 intermediate backward buffers` or `benchmark and result`. Keep commits focused and scoped to one change. PRs should include the problem statement, files/configs affected, exact reproduction commands, hardware/CUDA environment, and benchmark deltas. Add screenshots only for GUI-visible changes.
