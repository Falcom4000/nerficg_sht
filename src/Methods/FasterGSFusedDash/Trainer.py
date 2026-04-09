"""FasterGSFusedDash/Trainer.py"""

import torch
import torch.nn.functional as F

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud, apply_background_color
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.FasterGSFused.Loss import FasterGSFusedLoss
from Methods.FasterGSFused.utils import enable_expandable_segments, carve
from Methods.FasterGSDash.schedule_utils import TrainingScheduler
from Optim.Samplers.DatasetSamplers import DatasetSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    DENSIFICATION_START_ITERATION=600,
    DENSIFICATION_END_ITERATION=27_000,  # extended from FasterGSFused default (14900) to match DashGaussian
    DENSIFICATION_INTERVAL=100,
    DENSIFICATION_GRAD_THRESHOLD=0.0002,
    DENSIFICATION_PERCENT_DENSE=0.01,
    OPACITY_RESET_INTERVAL=3_000,
    EXTRA_OPACITY_RESET_ITERATION=500,
    MORTON_ORDERING_END_ITERATION=27_000,  # extended to match densification end
    MORTON_ORDERING_MIN_GAUSSIANS=50_000,  # conflict B fix: skip z-ordering when count is small
    USE_RANDOM_BACKGROUND_COLOR=False,
    MIN_OPACITY_AFTER_TRAINING=1 / 255,
    RANDOM_INITIALIZATION=Framework.ConfigParameterList(
        FORCE=False,
        N_POINTS=100_000,
        ENABLE_CARVING=True,
        CARVING_IN_ALL_FRUSTUMS=False,
        CARVING_ENFORCE_ALPHA=False,
    ),
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,
        LAMBDA_DSSIM=0.2,
    ),
    OPTIMIZER=Framework.ConfigParameterList(
        LEARNING_RATE_MEANS_INIT=0.00016,
        LEARNING_RATE_MEANS_FINAL=0.0000016,
        LEARNING_RATE_MEANS_MAX_STEPS=30_000,
    ),
    # ---- DashGaussian scheduling ----
    DASH=Framework.ConfigParameterList(
        DENSIFY_MODE="freq",            # "freq" = scheduled top-k | "free" = standard ADC
        RESOLUTION_MODE="freq",         # "freq" = FFT schedule    | "const" = full res always
        MAX_N_GAUSSIANS=-1,             # -1 = momentum-adaptive; >0 = hard cap
        INITIAL_MOMENTUM_FACTOR=5,      # initial momentum = factor × init_n_gaussian
        MAX_RESO_SCALE=8,               # hard cap on FFT-computed initial downsampling factor
        START_SIGNIFICANCE_FACTOR=4.0,  # FFT energy threshold: e_min = e_total / factor
    ),
)
class FasterGSFusedDashTrainer(GuiTrainer):
    """FasterGSFused trainer extended with DashGaussian resolution + primitive scheduling."""

    def __init__(self, **kwargs) -> None:
        self.requires_empty_cache = True
        if not Framework.config.TRAINING.GUI.ACTIVATE:
            if enable_expandable_segments():
                self.requires_empty_cache = False
                Logger.log_info('using "expandable_segments:True" with the torch cuda memory allocator')
        super().__init__(**kwargs)
        self.train_sampler = None
        self.loss = FasterGSFusedLoss(loss_config=self.LOSS)
        self.autograd_dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device='cuda'))
        self.dash_scheduler: TrainingScheduler | None = None
        self.current_render_scale: int = 1

    @pre_training_callback(priority=50)
    @torch.no_grad()
    def create_sampler(self, _, dataset: 'BaseDataset') -> None:
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setup_gaussians(self, _, dataset: 'BaseDataset') -> None:
        """Sets up the model and the DashGaussian scheduler."""
        dataset.train()
        camera_centers = torch.stack([view.position for view in dataset])
        radius = (1.1 * torch.max(torch.linalg.norm(camera_centers - torch.mean(camera_centers, dim=0), dim=1))).item()
        Logger.log_info(f'training cameras extent: {radius:.2f}')

        if dataset.point_cloud is not None and not self.RANDOM_INITIALIZATION.FORCE:
            point_cloud = dataset.point_cloud
        else:
            samples = torch.rand((self.RANDOM_INITIALIZATION.N_POINTS, 3), dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE)
            positions = samples * dataset.bounding_box.size + dataset.bounding_box.min
            if self.RANDOM_INITIALIZATION.ENABLE_CARVING:
                positions = carve(positions, dataset, self.RANDOM_INITIALIZATION.CARVING_IN_ALL_FRUSTUMS, self.RANDOM_INITIALIZATION.CARVING_ENFORCE_ALPHA)
            point_cloud = BasicPointCloud(positions)
        self.model.gaussians.initialize_from_point_cloud(point_cloud)
        self.model.gaussians.training_setup(self, radius)
        self.model.gaussians.reset_densification_info()

        # initialise DashGaussian scheduler
        init_n = self.model.gaussians.means.shape[0]
        self.dash_scheduler = TrainingScheduler(
            max_steps=self.NUM_ITERATIONS,
            init_n_gaussian=init_n,
            densify_until_iter=self.DENSIFICATION_END_ITERATION,
            densification_interval=self.DENSIFICATION_INTERVAL,
            max_n_gaussian=self.DASH.MAX_N_GAUSSIANS,
            initial_momentum_factor=self.DASH.INITIAL_MOMENTUM_FACTOR,
            densify_mode=self.DASH.DENSIFY_MODE,
            resolution_mode=self.DASH.RESOLUTION_MODE,
            original_images=[view.rgb for view in dataset.train()],
            max_reso_scale=self.DASH.MAX_RESO_SCALE,
            start_significance_factor=self.DASH.START_SIGNIFICANCE_FACTOR,
        )
        self.current_render_scale = self.dash_scheduler.get_res_scale(1)
        self._last_morton_n = 0
        Logger.log_info(f'DashGaussian scheduler ready — initial render_scale: {self.current_render_scale}')

    @training_callback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increase_sh_degree(self, *_) -> None:
        """Increase SH degree, but only once the resolution is near-full (Gap 2 fix)."""
        if self.dash_scheduler is not None and not self.dash_scheduler.near_full_resolution():
            return
        self.model.gaussians.increase_used_sh_degree()

    @training_callback(priority=100, start_iteration='DENSIFICATION_START_ITERATION', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, _) -> None:
        """Apply DashGaussian-scheduled densification."""
        n_gaussians = self.model.gaussians.means.shape[0]
        densify_rate = self.dash_scheduler.get_densify_rate(
            iteration, n_gaussians, self.current_render_scale
        )
        momentum_add = self.model.gaussians.dash_density_control_topk(
            grad_threshold=self.DENSIFICATION_GRAD_THRESHOLD,
            min_opacity=0.005,
            prune_large_gaussians=iteration > self.OPACITY_RESET_INTERVAL,
            render_scale=self.current_render_scale,
            densify_rate=densify_rate,
        )
        self.dash_scheduler.update_momentum(momentum_add)
        # update render_scale after densification (mirrors DashGaussian train_dash.py)
        self.current_render_scale = self.dash_scheduler.get_res_scale(iteration)
        if iteration < self.DENSIFICATION_END_ITERATION:
            self.model.gaussians.reset_densification_info()
        if self.requires_empty_cache:
            torch.cuda.empty_cache()

    @training_callback(priority=99, end_iteration='MORTON_ORDERING_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def morton_ordering(self, *_) -> None:
        """Apply Morton ordering adaptively: skip when count is small or growth < 20%."""
        n = self.model.gaussians.means.shape[0]
        if n < self.MORTON_ORDERING_MIN_GAUSSIANS:
            return
        if n < self._last_morton_n * 1.2:
            return
        self.model.gaussians.apply_morton_ordering()
        self._last_morton_n = n

    @training_callback(priority=90, start_iteration='OPACITY_RESET_INTERVAL', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='OPACITY_RESET_INTERVAL')
    @torch.no_grad()
    def reset_opacities(self, *_) -> None:
        """Reset opacities."""
        self.model.gaussians.reset_opacities()

    @training_callback(priority=90, start_iteration='EXTRA_OPACITY_RESET_ITERATION', end_iteration='EXTRA_OPACITY_RESET_ITERATION')
    @torch.no_grad()
    def reset_opacities_extra(self, _, dataset: 'BaseDataset') -> None:
        """Reset opacities one additional time when using a non-black background."""
        if dataset.default_camera.background_color.sum() != 0.0:
            Logger.log_info('resetting opacities one additional time because using non-black background')
            self.model.gaussians.reset_opacities()

    @training_callback(priority=80)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Training step with DashGaussian resolution scaling and fused Adam."""
        # init modes
        self.model.train()
        dataset.train()
        self.loss.train()
        # Conflict J fix: delay LR decay until near-full-resolution training.
        # Without this, position LR decays from iter 1 even while training at
        # low resolution (render_scale >> 1), starving mean positions of LR.
        optimization_step = iteration + 1
        lr_iter = max(1, iteration - self.dash_scheduler.lr_decay_from_iter() + 1)
        self.model.gaussians.update_learning_rate(lr_iter)
        # get random view
        view = self.train_sampler.get(dataset=dataset)['view']
        bg_color = torch.rand_like(view.camera.background_color) if self.USE_RANDOM_BACKGROUND_COLOR else view.camera.background_color

        render_scale = self.dash_scheduler.get_res_scale(iteration)
        self.current_render_scale = render_scale

        # render at reduced resolution (Conflict G fix: scaled dims passed to CUDA via Renderer)
        image, autograd_dummy = self.renderer.render_image_training(
            view=view,
            update_densification_info=iteration < self.DENSIFICATION_END_ITERATION,
            bg_color=bg_color,
            adam_step_count=optimization_step,  # Conflict D: global step for bias correction
            autograd_dummy=self.autograd_dummy,
            render_scale=render_scale,
        )

        # downsample GT with anti-aliased area interpolation (Gap 1)
        rgb_gt = view.rgb
        if (alpha_gt := view.alpha) is not None:
            rgb_gt = apply_background_color(rgb_gt, alpha_gt, bg_color)
        if render_scale > 1:
            rgb_gt = F.interpolate(
                rgb_gt.unsqueeze(0),
                scale_factor=1.0 / render_scale,
                mode='area',
            ).squeeze(0)

        # calculate loss
        loss = self.loss(image, rgb_gt) + 0.0 * autograd_dummy
        # backward (triggers fused Adam inside the CUDA kernel)
        loss.backward()

    @training_callback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def log_wandb(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds Gaussian count and render_scale to WandB logging."""
        Framework.wandb.log({
            '#Gaussians': self.model.gaussians.means.shape[0],
            'render_scale': self.current_render_scale,
        }, step=iteration)
        super().log_wandb(iteration, dataset)

    @post_training_callback(priority=1000)
    @torch.no_grad()
    def finalize(self, *_) -> None:
        """Clean up after training."""
        n_gaussians = self.model.gaussians.training_cleanup(min_opacity=self.MIN_OPACITY_AFTER_TRAINING)
        Logger.log_info(f'final number of Gaussians: {n_gaussians:,}')
        with open(str(self.output_directory / 'n_gaussians.txt'), 'w') as n_gaussians_file:
            n_gaussians_file.write(
                f'Final number of Gaussians: {n_gaussians:,}\n'
                f'\n'
                f'N_Gaussians:{n_gaussians}'
            )
