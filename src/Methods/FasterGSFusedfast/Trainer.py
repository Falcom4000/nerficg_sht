"""FasterGSFusedfast/Trainer.py"""

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.FasterGSFusedfast.Loss import FasterGSFusedLoss
from Methods.FasterGSFusedfast.utils import enable_expandable_segments, carve, compute_gaussian_scores_fastgs
from Optim.Samplers.DatasetSamplers import DatasetSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    DENSIFICATION_START_ITERATION=600,
    DENSIFICATION_END_ITERATION=14_900,  # while official code states 15000, densification actually stops at 14900 there
    DENSIFICATION_INTERVAL=100,
    DENSIFICATION_GRAD_THRESHOLD=0.0002,
    DENSIFICATION_GRAD_ABS_THRESHOLD=0.0012,
    DENSIFICATION_PERCENT_DENSE=0.001,
    FASTGS_DENSE_THRESHOLD=0.001,
    FASTGS_IMPORTANCE_THRESHOLD=5,
    FASTGS_SCORE_VIEWS=10,
    FASTGS_LOSS_THRESHOLD=0.1,
    OPACITY_RESET_INTERVAL=3_000,
    EXTRA_OPACITY_RESET_ITERATION=500,
    FASTGS_FINAL_PRUNE_START_ITERATION=18_000,
    FASTGS_FINAL_PRUNE_END_ITERATION=27_000,
    FASTGS_FINAL_PRUNE_INTERVAL=3_000,
    FASTGS_FINAL_PRUNE_SCORE_THRESHOLD=0.9,
    MORTON_ORDERING_INTERVAL=5000,  # lowering to 2500 or 1000 may improve performance when number of Gaussians is high
    MORTON_ORDERING_END_ITERATION=15000,
    USE_RANDOM_BACKGROUND_COLOR=False,  # prevents the model from overfitting to the background color
    MIN_OPACITY_AFTER_TRAINING=1 / 255,
    RANDOM_INITIALIZATION=Framework.ConfigParameterList(
        FORCE=False,  # if True, the point cloud from the dataset will be ignored
        N_POINTS=100_000,  # number of random points to be sampled within the scene bounding box
        ENABLE_CARVING=True,  # removes points that are never in-frustum in any training view
        CARVING_IN_ALL_FRUSTUMS=False,  # removes points not in-frustum in all views
        CARVING_ENFORCE_ALPHA=False,  # removes points that project to a pixel with alpha=0 in any view where the point is in-frustum
    ),
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,  # weight for the per-pixel L1 loss on the rgb image
        LAMBDA_DSSIM=0.2,  # weight for the DSSIM loss on the rgb image
    ),
    OPTIMIZER=Framework.ConfigParameterList(
        LEARNING_RATE_MEANS_INIT=0.00016,
        LEARNING_RATE_MEANS_FINAL=0.0000016,
        LEARNING_RATE_MEANS_MAX_STEPS=30_000,
        LEARNING_RATE_SH_COEFFICIENTS_0=0.0025,
        LEARNING_RATE_SH_COEFFICIENTS_REST=0.00025,
        LEARNING_RATE_OPACITIES=0.025,
        LEARNING_RATE_SCALES=0.005,
        LEARNING_RATE_ROTATIONS=0.001,
    ),
)
class FasterGSFusedTrainer(GuiTrainer):
    """Defines the trainer for the FasterGSFused variant."""

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
        self.fastgs_background_color = None
        self.fastgs_main_step_count = 0
        self.fastgs_sh_step_count = 0
        for optimization_step in range(1, self.model.num_iterations_trained + 1):
            update_main, update_sh = self._fastgs_update_schedule(optimization_step)
            self.fastgs_main_step_count += int(update_main)
            self.fastgs_sh_step_count += int(update_sh)

    @pre_training_callback(priority=50)
    @torch.no_grad()
    def create_sampler(self, _, dataset: 'BaseDataset') -> None:
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setup_gaussians(self, _, dataset: 'BaseDataset') -> None:
        """Sets up the model."""
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
        self.model.gaussians.reset_max_radii2D()
        default_background = dataset.default_camera.background_color.to(device='cuda', dtype=torch.float32)
        self.fastgs_background_color = torch.rand_like(default_background) if self.USE_RANDOM_BACKGROUND_COLOR else default_background

    @staticmethod
    def _fastgs_update_schedule(optimization_step: int) -> tuple[bool, bool]:
        """Return FastGS update gates for main params and SH params."""
        if optimization_step <= 15_000:
            return True, optimization_step % 16 == 0
        if optimization_step <= 20_000:
            update_now = optimization_step % 32 == 0
            return update_now, update_now
        update_now = optimization_step % 64 == 0
        return update_now, update_now

    @training_callback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increase_sh_degree(self, *_) -> None:
        """Increase the number of used SH coefficients up to a maximum degree."""
        self.model.gaussians.increase_used_sh_degree()

    @training_callback(priority=100, start_iteration='DENSIFICATION_START_ITERATION', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Apply densification."""
        if iteration <= self.DENSIFICATION_START_ITERATION or iteration % self.DENSIFICATION_INTERVAL != 0:
            return
        importance_score, pruning_score = compute_gaussian_scores_fastgs(
            dataset=dataset.train(),
            renderer=self.renderer,
            num_views=self.FASTGS_SCORE_VIEWS,
            loss_thresh=self.FASTGS_LOSS_THRESHOLD,
            lambda_dssim=self.LOSS.LAMBDA_DSSIM,
            bg_color=self.fastgs_background_color,
        )
        if importance_score is None or pruning_score is None:
            return
        self.model.gaussians.adaptive_density_control_fastgs(
            grad_threshold=self.DENSIFICATION_GRAD_THRESHOLD,
            grad_abs_threshold=self.DENSIFICATION_GRAD_ABS_THRESHOLD,
            min_opacity=0.005,
            importance_score=importance_score,
            pruning_score=pruning_score,
            importance_threshold=self.FASTGS_IMPORTANCE_THRESHOLD,
            max_screen_size=20.0 if iteration > self.OPACITY_RESET_INTERVAL else None,
        )
        if iteration < self.DENSIFICATION_END_ITERATION:
            self.model.gaussians.reset_densification_info()
        if self.requires_empty_cache:
            torch.cuda.empty_cache()

    @training_callback(priority=99, end_iteration='MORTON_ORDERING_END_ITERATION', iteration_stride='MORTON_ORDERING_INTERVAL')
    @torch.no_grad()
    def morton_ordering(self, *_) -> None:
        """Apply morton ordering to all Gaussian parameters and their optimizer states."""
        self.model.gaussians.apply_morton_ordering()

    @training_callback(priority=90, start_iteration='OPACITY_RESET_INTERVAL', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='OPACITY_RESET_INTERVAL')
    @torch.no_grad()
    def reset_opacities(self, *_) -> None:
        """Reset opacities."""
        self.model.gaussians.reset_opacities()

    @training_callback(priority=90, start_iteration='EXTRA_OPACITY_RESET_ITERATION', end_iteration='EXTRA_OPACITY_RESET_ITERATION')
    @torch.no_grad()
    def reset_opacities_extra(self, _, dataset: 'BaseDataset') -> None:
        """Reset opacities one additional time when using a white background."""
        if torch.allclose(dataset.default_camera.background_color, torch.ones_like(dataset.default_camera.background_color)):
            Logger.log_info('resetting opacities one additional time because using white background')
            self.model.gaussians.reset_opacities()

    @training_callback(
        priority=95,
        start_iteration='FASTGS_FINAL_PRUNE_START_ITERATION',
        end_iteration='FASTGS_FINAL_PRUNE_END_ITERATION',
        iteration_stride='FASTGS_FINAL_PRUNE_INTERVAL',
    )
    @torch.no_grad()
    def fastgs_final_prune(self, _, dataset: 'BaseDataset') -> None:
        """Apply the late-stage FastGS multi-view consistent pruning schedule."""
        _, pruning_score = compute_gaussian_scores_fastgs(
            dataset=dataset.train(),
            renderer=self.renderer,
            num_views=self.FASTGS_SCORE_VIEWS,
            loss_thresh=self.FASTGS_LOSS_THRESHOLD,
            lambda_dssim=self.LOSS.LAMBDA_DSSIM,
            bg_color=self.fastgs_background_color,
        )
        if pruning_score is None:
            return
        self.model.gaussians.final_prune_fastgs(0.1, pruning_score, self.FASTGS_FINAL_PRUNE_SCORE_THRESHOLD)
        if self.requires_empty_cache:
            torch.cuda.empty_cache()

    @training_callback(priority=105)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs a training step without actually doing the optimizer step."""
        # init modes
        self.model.train()
        dataset.train()
        self.loss.train()
        # update learning rate
        optimization_step = iteration + 1
        self.model.gaussians.update_learning_rate(optimization_step)
        # get random view
        view = self.train_sampler.get(dataset=dataset)['view']
        # render
        bg_color = self.fastgs_background_color
        image, radii, autograd_dummy = self.renderer.render_image_training(
            view=view,
            update_densification_info=iteration < self.DENSIFICATION_END_ITERATION,
            bg_color=bg_color,
            adam_step_count_main=0,
            adam_step_count_sh=0,
            apply_parameter_updates=False,
            update_sh_coefficients=False,
            autograd_dummy=self.autograd_dummy,
        )
        if iteration < self.DENSIFICATION_END_ITERATION:
            visibility_filter = radii > 0
            self.model.gaussians.max_radii2D[visibility_filter] = torch.maximum(
                self.model.gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter],
            )
        # calculate loss
        rgb_gt = view.rgb
        loss = self.loss(image, rgb_gt) + 0.0 * autograd_dummy
        # backward
        loss.backward()

    @training_callback(priority=85)
    @torch.no_grad()
    def optimizer_step_fastgs(self, iteration: int, _) -> None:
        """Apply FastGS optimizer updates after densify/prune/reset callbacks."""
        optimization_step = iteration + 1
        if optimization_step >= self.NUM_ITERATIONS:
            return
        apply_parameter_updates, update_sh_coefficients = self._fastgs_update_schedule(optimization_step)
        adam_step_count_main = None
        adam_step_count_sh = None
        if apply_parameter_updates:
            self.fastgs_main_step_count += 1
            adam_step_count_main = self.fastgs_main_step_count
        if update_sh_coefficients:
            self.fastgs_sh_step_count += 1
            adam_step_count_sh = self.fastgs_sh_step_count
        self.model.gaussians.optimizer_step_fastgs(adam_step_count_main, adam_step_count_sh)

    @training_callback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def log_wandb(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds Gaussian count to default Weights & Biases logging."""
        Framework.wandb.log({
            '#Gaussians': self.model.gaussians.means.shape[0]
        }, step=iteration)
        # default logging
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
