"""FasterGSFusedfast/utils.py"""

import io
import warnings
import contextlib
import random

import torch

from Datasets.Base import BaseDataset
from Datasets.utils import apply_background_color
from Logging import Logger
from Optim.Losses.DSSIM import fused_dssim


def enable_expandable_segments() -> bool:
    """Return True if 'expandable_segments' allocator feature is available on this device."""
    torch.cuda.memory._set_allocator_settings('expandable_segments:True')
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stderr(stderr_buffer), warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always')
        torch.empty(1, device='cuda')  # allocate gpu memory to trigger potential warning
    stderr_output = stderr_buffer.getvalue()
    for warning in caught_warnings:
        if 'expandable_segments' in str(warning.message):
            return False
    if 'expandable_segments' in stderr_output:
        return False
    return True


def carve(points: torch.Tensor, dataset: BaseDataset, in_all_frustums: bool, enforce_alpha: bool) -> torch.Tensor:
    """
    Carves away points based on visibility and alpha.
    - Points that are never in-frustum in any view are removed.
    - If in_all_frustums=True, points not in-frustum in all views are removed.
    - If enforce_alpha=True, points that project to a pixel with alpha=0 in any view (where the point is in-frustum) are removed.
    """
    Logger.log_info(f'removing points that would not be visible in any training view (in_all_frustums={in_all_frustums}, enforce_alpha={enforce_alpha})')
    in_frustum_any = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    in_frustum_all = torch.ones_like(in_frustum_any)
    in_alpha_all = torch.ones_like(in_frustum_any)
    dilation_kernel = torch.ones(1, 1, 3, 3) if enforce_alpha else None
    for view in dataset:
        xy_screen, _, in_frustum = view.project_points(points)
        in_frustum_any |= in_frustum
        if in_all_frustums:
            in_frustum_all &= in_frustum
        if enforce_alpha and in_frustum.any() and (alpha_gt := view.alpha) is not None:
            alpha_gt = torch.nn.functional.conv2d(alpha_gt[None], dilation_kernel, padding=1)[0] > 0
            xy_screen = torch.floor(xy_screen[in_frustum]).long()
            valid_alpha = alpha_gt[0, xy_screen[:, 1], xy_screen[:, 0]] > 0
            in_alpha_all[in_frustum] &= valid_alpha
    valid_mask = in_frustum_any & in_alpha_all & in_frustum_all
    return points[valid_mask].contiguous()


def sample_training_views(dataset: BaseDataset, num_views: int) -> list:
    """Randomly sample up to ``num_views`` training views from a dataset."""
    views = list(dataset)
    if not views:
        return []
    sample_count = min(num_views, len(views))
    return random.sample(views, sample_count)


def compute_photometric_loss(rendered_image: torch.Tensor, target_image: torch.Tensor, lambda_dssim: float) -> torch.Tensor:
    """Compute the scalar photometric loss used by FastGS for pruning scores."""
    l1_term = torch.nn.functional.l1_loss(rendered_image, target_image)
    dssim_term = fused_dssim(rendered_image, target_image)
    return (1.0 - lambda_dssim) * l1_term + lambda_dssim * dssim_term


def compute_metric_map(rendered_image: torch.Tensor, target_image: torch.Tensor, loss_thresh: float) -> torch.Tensor:
    """Create the FastGS-style high-error metric map from the normalized per-pixel L1 error."""
    l1_map = torch.mean(torch.abs(rendered_image - target_image), dim=0)
    l1_min = torch.min(l1_map)
    l1_max = torch.max(l1_map)
    denom = torch.clamp(l1_max - l1_min, min=1e-8)
    l1_map_norm = (l1_map - l1_min) / denom
    return (l1_map_norm > loss_thresh).to(dtype=torch.int32).contiguous().flatten()


@torch.no_grad()
def compute_gaussian_scores_fastgs(
    dataset: BaseDataset,
    renderer,
    num_views: int,
    loss_thresh: float,
    lambda_dssim: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute FastGS-style multi-view importance and pruning scores."""
    sampled_views = sample_training_views(dataset, num_views)
    if not sampled_views:
        return None, None

    full_metric_counts = None
    full_metric_score = None
    for view in sampled_views:
        bg_color = view.camera.background_color.to(device=view.rgb.device, dtype=torch.float32)
        rendered_image = renderer.render_image_inference(view, to_chw=True)['rgb']
        target_image = view.rgb
        if (alpha_gt := view.alpha) is not None:
            target_image = apply_background_color(target_image, alpha_gt, bg_color)

        photometric_loss = compute_photometric_loss(rendered_image, target_image, lambda_dssim)
        metric_map = compute_metric_map(rendered_image, target_image, loss_thresh)
        metric_counts = renderer.render_image_scoring(view, bg_color, metric_map).to(dtype=torch.float32)

        if full_metric_counts is None:
            full_metric_counts = metric_counts.clone()
            full_metric_score = photometric_loss * metric_counts
        else:
            full_metric_counts += metric_counts
            full_metric_score += photometric_loss * metric_counts

    score_min = torch.min(full_metric_score)
    score_max = torch.max(full_metric_score)
    if torch.allclose(score_min, score_max):
        pruning_score = torch.zeros_like(full_metric_score)
    else:
        pruning_score = (full_metric_score - score_min) / (score_max - score_min)
    importance_score = torch.div(full_metric_counts, len(sampled_views), rounding_mode='floor').to(dtype=torch.int32)
    return importance_score, pruning_score
