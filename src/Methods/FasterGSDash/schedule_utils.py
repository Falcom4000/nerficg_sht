"""FasterGSDash/schedule_utils.py

DashGaussian TrainingScheduler ported to the NeRFICG framework.
Original: DashGaussian/utils/schedule_utils.py
(Copyright 2025 Harbin Institute of Technology, Huawei Noah's Ark Lab, CC-BY-NC-SA-4.0)

All DashGaussian-specific argument-class dependencies have been removed;
the constructor now accepts plain Python parameters.
"""

import math

import torch


class TrainingScheduler:
    """Resolution and primitive-count scheduler from DashGaussian.

    Controls:
      - render_scale: the integer downsampling factor applied to both the
        rendered image and the GT image each training iteration.
      - densify_rate: the maximum allowed fractional growth of the Gaussian
        count at each densification step.
    """

    def __init__(
        self,
        max_steps: int,
        init_n_gaussian: int,
        densify_until_iter: int,
        densification_interval: int,
        max_n_gaussian: int,
        initial_momentum_factor: float,
        densify_mode: str,
        resolution_mode: str,
        original_images: list,
        max_reso_scale: int = 8,
        start_significance_factor: float = 4.0,
    ) -> None:
        """
        Args:
            max_steps: total training iterations.
            init_n_gaussian: number of Gaussians at initialisation.
            densify_until_iter: iteration at which densification ends.
            densification_interval: iterations between densification steps.
            max_n_gaussian: hard cap on Gaussians (<=0 → momentum-based).
            initial_momentum_factor: momentum initial value = factor × init_n_gaussian
                (only used when max_n_gaussian <= 0).
            densify_mode: "freq" for scheduled densification, "free" for
                unconstrained (degrades to standard 3DGS behaviour).
            resolution_mode: "freq" for FFT-based resolution schedule,
                "const" to keep full resolution throughout.
            original_images: list of training-view RGB tensors (C×H×W,
                float32, on any device) used for FFT analysis.
            max_reso_scale: hard upper bound on the FFT-computed initial downsampling
                factor. Lowering this (e.g. 4) prevents extreme downsampling on
                indoor scenes, making the early densify_rate budget less restrictive.
            start_significance_factor: controls the FFT energy threshold that determines
                max_reso_scale. e_min = e_total / factor; lower values (e.g. 2) yield
                a smaller max_reso_scale for scenes with spread-out frequency content.
        """
        self.max_steps = max_steps
        self.init_n_gaussian = init_n_gaussian
        self.densify_mode = densify_mode
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.resolution_mode = resolution_mode

        self.start_significance_factor = start_significance_factor
        self.max_reso_scale = max_reso_scale
        self.reso_sample_num = 32  # must be >= 2
        self.max_densify_rate_per_step = 0.2
        self.reso_scales = None
        self.reso_level_significance = None
        self.reso_level_begin = None
        self.increase_reso_until = self.densify_until_iter
        self.next_i = 2

        if max_n_gaussian > 0:
            self.max_n_gaussian = max_n_gaussian
            self.momentum = -1  # disabled
        else:
            self.momentum = int(initial_momentum_factor * self.init_n_gaussian)
            self.max_n_gaussian = self.init_n_gaussian + self.momentum
            self.integrate_factor = 0.98
            self.momentum_step_cap = 1_000_000

        self.init_reso_scheduler(original_images)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_momentum(self, momentum_step: int) -> None:
        """Update the adaptive P_fin estimate after a densification step.

        Args:
            momentum_step: number of Gaussians that naturally passed the
                gradient threshold (before top-k budget capping).
        """
        if self.momentum == -1:
            return
        self.momentum = max(
            self.momentum,
            int(self.integrate_factor * self.momentum
                + min(self.momentum_step_cap, momentum_step)),
        )
        self.max_n_gaussian = self.init_n_gaussian + self.momentum

    def get_res_scale(self, iteration: int) -> int:
        """Return the current integer downsampling factor (1 = full resolution)."""
        if self.resolution_mode == "const":
            return 1
        elif self.resolution_mode == "freq":
            if iteration >= self.increase_reso_until:
                return 1
            if iteration < self.reso_level_begin[1]:
                return int(self.reso_scales[0])
            while iteration >= self.reso_level_begin[self.next_i]:
                self.next_i += 1
            i = self.next_i - 1
            i_now, i_nxt = self.reso_level_begin[i: i + 2]
            s_lst, s_now = self.reso_scales[i - 1: i + 1]
            scale = (
                1.0
                / (
                    (iteration - i_now) / (i_nxt - i_now)
                    * (1.0 / s_now ** 2 - 1.0 / s_lst ** 2)
                    + 1.0 / s_lst ** 2
                )
            ) ** 0.5
            return max(1, int(scale))
        else:
            raise NotImplementedError(
                f"Resolution mode '{self.resolution_mode}' is not implemented."
            )

    def get_densify_rate(
        self, iteration: int, cur_n_gaussian: int, cur_scale: int = None
    ) -> float:
        """Return the maximum allowed fractional growth for this densification step.

        Returns a value in [0, max_densify_rate_per_step].
        """
        if self.densify_mode == "free":
            return 1.0
        elif self.densify_mode == "freq":
            assert cur_scale is not None, "cur_scale required for densify_mode='freq'"
            if self.densification_interval + iteration < self.increase_reso_until:
                next_n_gaussian = int(
                    (self.max_n_gaussian - self.init_n_gaussian)
                    / cur_scale ** (2.0 - iteration / self.densify_until_iter)
                ) + self.init_n_gaussian
            else:
                next_n_gaussian = self.max_n_gaussian
            rate = (next_n_gaussian - cur_n_gaussian) / max(cur_n_gaussian, 1)
            return min(max(rate, 0.0), self.max_densify_rate_per_step)
        else:
            raise NotImplementedError(
                f"Densify mode '{self.densify_mode}' is not implemented."
            )

    def near_full_resolution(self, sh_unlock_scale_threshold: float = 4.0) -> bool:
        """Return True once the schedule has advanced past the given scale threshold.

        SH degree unlock is gated on this to avoid training high-order SH when there
        is no high-frequency supervision signal.  The default threshold of 4.0 means
        SH unlocking is allowed as soon as we have progressed past the 1/4-resolution
        level, which gives enough detail to benefit from view-dependent colour while
        still preventing noise-driven contamination at 1/8 resolution.
        (Using 2.0 was too conservative: it delayed SH until ~iter 20000 on bicycle,
        causing a ~1 dB PSNR regression.)
        """
        if self.resolution_mode == "const":
            return True
        if self.reso_scales is None:
            return False
        for idx, (begin, scale) in enumerate(zip(self.reso_level_begin, self.reso_scales)):
            if scale < sh_unlock_scale_threshold:
                return self.next_i - 1 >= idx
        return True

    def lr_decay_from_iter(self) -> int:
        """Return the first iteration at which render_scale first drops below 2."""
        if self.resolution_mode == "const":
            return 1
        for i, s in zip(self.reso_level_begin, self.reso_scales):
            if s < 2.0:
                return i
        return 1

    # ------------------------------------------------------------------
    # Internal: FFT-based resolution schedule initialisation
    # ------------------------------------------------------------------

    def init_reso_scheduler(self, original_images: list) -> None:
        if self.resolution_mode != "freq":
            print(
                f"[ INFO ] Skipped resolution scheduler initialisation "
                f"(resolution_mode={self.resolution_mode})"
            )
            return

        def compute_win_significance(significance_map: torch.Tensor, scale: float) -> float:
            h, w = significance_map.shape[-2:]
            c = ((h + 1) // 2, (w + 1) // 2)
            win_size = (int(h / scale), int(w / scale))
            return significance_map[
                ...,
                c[0] - win_size[0] // 2: c[0] + win_size[0] // 2,
                c[1] - win_size[1] // 2: c[1] + win_size[1] // 2,
            ].sum().item()

        def scale_solver(significance_map: torch.Tensor, target_significance: float) -> float:
            lo, hi = 0.0, 1.0
            for _ in range(64):
                mid = (lo + hi) / 2.0
                if compute_win_significance(significance_map, 1.0 / mid) < target_significance:
                    lo = mid
                else:
                    hi = mid
            return 1.0 / mid

        print("[ INFO ] Initialising resolution scheduler (FFT analysis)...")

        # max_reso_scale was set in __init__ from the constructor argument;
        # reset next_i but keep the configured cap.
        self.next_i = 2
        scene_freq_image = None

        for img in original_images:
            # img: C×H×W tensor (any device)
            img_fft = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
            img_mod = (img_fft.real.square() + img_fft.imag.square()).sqrt()
            scene_freq_image = img_mod if scene_freq_image is None else scene_freq_image + img_mod

            e_total = img_mod.sum().item()
            e_min = e_total / self.start_significance_factor
            self.max_reso_scale = min(
                self.max_reso_scale, scale_solver(img_mod, e_min)
            )

        modulation_func = math.log

        self.reso_scales = []
        self.reso_level_significance = []
        self.reso_level_begin = []

        scene_freq_image = scene_freq_image / len(original_images)
        E_total = scene_freq_image.sum().item()
        E_min = compute_win_significance(scene_freq_image, self.max_reso_scale)

        self.reso_level_significance.append(E_min)
        self.reso_scales.append(self.max_reso_scale)
        self.reso_level_begin.append(0)

        n = self.reso_sample_num
        for i in range(1, n - 1):
            sig = (E_total - E_min) * (i - 0) / (n - 1 - 0) + E_min
            self.reso_level_significance.append(sig)
            self.reso_scales.append(scale_solver(scene_freq_image, sig))
            self.reso_level_significance[-2] = modulation_func(
                self.reso_level_significance[-2] / E_min
            )
            self.reso_level_begin.append(
                int(
                    self.increase_reso_until
                    * self.reso_level_significance[-2]
                    / modulation_func(E_total / E_min)
                )
            )

        self.reso_level_significance.append(modulation_func(E_total / E_min))
        self.reso_scales.append(1.0)
        self.reso_level_significance[-2] = modulation_func(
            self.reso_level_significance[-2] / E_min
        )
        self.reso_level_begin.append(
            int(
                self.increase_reso_until
                * self.reso_level_significance[-2]
                / modulation_func(E_total / E_min)
            )
        )
        self.reso_level_begin.append(self.increase_reso_until)

        print(
            f"[ INFO ] Resolution scheduler ready: "
            f"max_scale={self.max_reso_scale:.1f}, "
            f"levels={len(self.reso_scales)}, "
            f"full-res from iter≈{self.lr_decay_from_iter()}"
        )
