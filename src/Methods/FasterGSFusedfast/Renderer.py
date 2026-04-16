"""FasterGSFusedfast/Renderer.py"""

import math

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.FasterGSFusedfast.Model import FasterGSFusedModel
from Methods.FasterGSFusedfast.FasterGSFusedCudaBackend import diff_rasterize, score_rasterize, RasterizerSettings


def extract_settings(
    view: View,
    active_sh_bases: int,
    bg_color: torch.Tensor,
    current_mean_lr: float,
    adam_step_count: int,
) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError('FasterGSFused renderer only supports perspective cameras')
    if view.camera.distortion is not None:
        Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
    return RasterizerSettings(
        view.w2c,
        view.position,
        bg_color,
        active_sh_bases,
        view.camera.width,
        view.camera.height,
        view.camera.focal_x,
        view.camera.focal_y,
        view.camera.center_x,
        view.camera.center_y,
        view.camera.near_plane,
        view.camera.far_plane,
        current_mean_lr,
        adam_step_count,
    )


@Framework.Configurable.configure(
    SCALE_MODIFIER=1.0,
)
class FasterGSFusedRenderer(BaseRenderer):
    """Wrapper around the rasterization module from 3DGS."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [FasterGSFusedModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('FasterGSFused renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'FasterGSFused renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if self.model.training:
            raise Framework.RendererError('please directly call render_image_training() instead of render_image() during training')
        else:
            return self.render_image_inference(view, to_chw)

    def render_image_training(
        self,
        view: View,
        update_densification_info: bool,
        bg_color: torch.Tensor,
        adam_step_count_main: int,
        adam_step_count_sh: int,
        apply_parameter_updates: bool,
        update_sh_coefficients: bool,
        autograd_dummy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Renders an image for a given view."""
        image, radii, autograd_dummy = diff_rasterize(
            means=self.model.gaussians.means,
            grad_accum_means=self.model.gaussians.grad_accum_means,
            moments_means=self.model.gaussians.moments_means,
            scales=self.model.gaussians.raw_scales,
            grad_accum_scales=self.model.gaussians.grad_accum_scales,
            moments_scales=self.model.gaussians.moments_scales,
            rotations=self.model.gaussians.raw_rotations,
            grad_accum_rotations=self.model.gaussians.grad_accum_rotations,
            moments_rotations=self.model.gaussians.moments_rotations,
            opacities=self.model.gaussians.raw_opacities,
            grad_accum_opacities=self.model.gaussians.grad_accum_opacities,
            moments_opacities=self.model.gaussians.moments_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            grad_accum_sh_coefficients_0=self.model.gaussians.grad_accum_sh_coefficients_0,
            moments_sh_coefficients_0=self.model.gaussians.moments_sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            grad_accum_sh_coefficients_rest=self.model.gaussians.grad_accum_sh_coefficients_rest,
            moments_sh_coefficients_rest=self.model.gaussians.moments_sh_coefficients_rest,
            autograd_dummy=autograd_dummy,
            densification_info=self.model.gaussians.densification_info if update_densification_info else torch.empty(0),
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, bg_color, self.model.gaussians.lr_means, adam_step_count_main),
            adam_step_count_sh=adam_step_count_sh,
            apply_parameter_updates=apply_parameter_updates,
            update_sh_coefficients=update_sh_coefficients,
        )
        return image, radii, autograd_dummy

    @torch.no_grad()
    def render_image_scoring(self, view: View, bg_color: torch.Tensor, metric_map: torch.Tensor) -> torch.Tensor:
        """Render a view in score-only mode and return per-Gaussian FastGS metric counts."""
        _, accum_metric_counts = score_rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            metric_map=metric_map,
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, bg_color, 0.0, 0),
        )
        return accum_metric_counts

    @torch.no_grad()
    def render_image_inference(
        self,
        view: View,
        to_chw: bool = False,
        clamp_output: bool = True,
        bg_color: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        image, _, _ = diff_rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales + math.log(max(self.SCALE_MODIFIER, 1e-6)),
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            autograd_dummy=torch.empty(0),
            densification_info=torch.empty(0),
            rasterizer_settings=extract_settings(
                view,
                self.model.gaussians.active_sh_bases,
                view.camera.background_color if bg_color is None else bg_color,
                0.0,
                0,
            ),
            grad_accum_means=torch.empty(0),
            grad_accum_scales=torch.empty(0),
            grad_accum_rotations=torch.empty(0),
            grad_accum_opacities=torch.empty(0),
            grad_accum_sh_coefficients_0=torch.empty(0),
            grad_accum_sh_coefficients_rest=torch.empty(0),
        )
        if clamp_output:
            image = image.clamp(0.0, 1.0)
        return {'rgb': image if to_chw else image.permute(1, 2, 0)}

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {'rgb': outputs['rgb']}
