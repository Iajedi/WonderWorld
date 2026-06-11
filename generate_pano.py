"""
WonderWorld codebase analysis (Step 1)
======================================

run.py — step-by-step:
  1. Load OmegaConf (base-config.yaml + example YAML).
  2. Init OneFormer segmentation, RepViT masks, inpainter (BCDM/FLUX or SD2),
     Marigold depth/normals, KeyframeGen.
  3. Load 512x512 start image -> kf_gen.image_latest; bootstrap sky point cloud.
  4. recompose_image_latest_and_set_current_pc() builds first point cloud.
  5. Train sky 3DGS (convert_to_3dgs_traindata -> traindata_sky) when gen_sky.
  6. Train first scene 3DGS; set inscreen visibility at canonical pose.
  7. Interactive Flask loop: render at user view_matrix, inpaint, update PC,
     train next 3DGS — camera motion driven by rotation_path / kf_gen.cameras.

models.py — renderer:
  - KeyframeGen.render() uses PyTorch3D point clouds (not 3DGS).
  - Post-training views use gaussian_renderer.render(viewpoint_camera, pc, opt, bg).
  - Returns dict with 'render' tensor [3, H, W] in [0, 1]; ToPILImage() for saving.
  - Bridge: PerspectiveCameras -> scene.cameras.Camera via convert_pt3d_cam_to_3dgs_cam.

Camera format:
  - Generation: PyTorch3D PerspectiveCameras (kf_gen.get_camera_at_origin(), kf_gen.cameras[i]).
  - Rendering: scene.cameras.Camera(R, T, FoVx, FoVy) — COLMAP-style w2c storage.
  - Yaw: rotate_pytorch3d_camera(..., axis='y') around world Y (same as set_cameras).

Key imports reused:
  arguments.GSParams, arguments.CameraParams
  gaussian_renderer.render, scene.Scene, scene.GaussianModel, scene.cameras.Camera
  models.models.KeyframeGen
  util.utils.convert_pt3d_cam_to_3dgs_cam, rotate_pytorch3d_camera
  run_ww_worldscore.train_gaussian, bootstrap_sky_pointcloud, has_traindata_points

Reference: run_ww_worldscore.py batch pipeline; pano replaces WorldScore cameras with
kf_gen.cameras[i] and final render with horizontal yaw sweep.
"""

from __future__ import annotations

import copy
import gc
import json
import random
import shutil
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from kornia.morphology import dilation
from marigold_lcm.marigold_pipeline import MarigoldNormalsPipeline, MarigoldPipeline
from omegaconf import OmegaConf
from PIL import Image
from pytorch3d.renderer import PerspectiveCameras
from torchvision.transforms import ToPILImage, ToTensor
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

from arguments import CameraParams, GSParams
from backbone.pipeline import BackbonePipeline
from gaussian_renderer import render
from models.models import KeyframeGen
from scene import GaussianModel, Scene
from scene.cameras import Camera
from util.segment_utils import create_mask_generator_repvit
from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from util.utils import (
    convert_pt3d_cam_to_3dgs_cam,
    prepare_scheduler,
    rotate_pytorch3d_camera,
    soft_stitching,
)
from syncdiffusion.syncdiffusion_model import SyncDiffusion

import run_ww_worldscore as ww_module
from run_ww_worldscore import has_traindata_points, train_gaussian

XYZ_SCALE = 1000

# Wide overview camera (matches run.py render_current_scene viz / save path).
_VIEW_MATRIX_FIXED = np.array(
    [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0.2, 0.5, 1],
    ]
)
_PITCH_RAD = np.radians(-3)
_ROTATION_MATRIX_X = np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(_PITCH_RAD), -np.sin(_PITCH_RAD), 0],
        [0, np.sin(_PITCH_RAD), np.cos(_PITCH_RAD), 0],
        [0, 0, 0, 1],
    ]
)
VIEW_MATRIX_FIXED = np.dot(_VIEW_MATRIX_FIXED, _ROTATION_MATRIX_X).flatten().tolist()


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _pt3d_camera_to_c2w(pt3d_cam: PerspectiveCameras, xyz_scale: int = XYZ_SCALE) -> np.ndarray:
    """Extract 4x4 camera-to-world matrix from a PyTorch3D camera."""
    transform_matrix_pt3d = pt3d_cam.get_world_to_view_transform().get_matrix()[0]
    transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
    transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale
    transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()
    device = pt3d_cam.device
    opengl_to_pt3d = torch.diag(torch.tensor([-1.0, 1.0, -1.0, 1.0], device=device))
    transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d
    c2w = transform_matrix_c2w_opengl.cpu().numpy()
    c2w[:3, 1:3] *= -1
    return c2w


def _c2w_to_pt3d_camera(
    c2w: np.ndarray,
    camera_params: CameraParams,
    device: torch.device,
) -> PerspectiveCameras:
    """Build a PyTorch3D camera from c2w and CameraParams intrinsics."""
    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=device)
    c2w_gl = c2w_t.clone()
    c2w_gl[:3, 1:3] *= -1
    opengl_to_pt3d = torch.diag(torch.tensor([-1.0, 1.0, -1.0, 1.0], device=device))
    c2w_pt3d = c2w_gl @ opengl_to_pt3d.inverse()
    w2c_pt3d = c2w_pt3d.inverse()
    w2c_pt3d[:3, 3] /= XYZ_SCALE

    transform_matrix = w2c_pt3d.transpose(0, 1).unsqueeze(0)
    R = transform_matrix[:, :3, :3]
    T = transform_matrix[:, 3, :3]

    h, w = camera_params.H, camera_params.W
    focal_x, focal_y = camera_params.focal
    K = torch.zeros((1, 4, 4), device=device)
    K[0, 0, 0] = focal_x
    K[0, 1, 1] = focal_y
    K[0, 0, 2] = w / 2
    K[0, 1, 2] = h / 2
    K[0, 2, 3] = 1
    K[0, 3, 2] = 1

    return PerspectiveCameras(
        K=K,
        R=R,
        T=T,
        in_ndc=False,
        image_size=((h, w),),
        device=device,
    )


def build_horizontal_camera_poses(
    canonical_c2w: np.ndarray,
    yaw_offsets_deg: list[float],
    camera_params: CameraParams,
) -> list[Camera]:
    """
    Given a canonical camera-to-world matrix and yaw offsets in degrees,
    return 3DGS Camera objects rotated horizontally around world Y-axis.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    canonical_pt3d = _c2w_to_pt3d_camera(canonical_c2w, camera_params, device)
    tdgs_cameras: list[Camera] = []
    for yaw_deg in yaw_offsets_deg:
        rotated = rotate_pytorch3d_camera(
            canonical_pt3d,
            torch.tensor(float(np.radians(yaw_deg))),
            axis="y",
        )
        tdgs_cameras.append(convert_pt3d_cam_to_3dgs_cam(rotated, xyz_scale=XYZ_SCALE))
    return tdgs_cameras


def _camera_to_c2w_list(tdgs_cam: Camera) -> list[list[float]]:
    """Reconstruct c2w from a 3DGS Camera (inverse of convert_pt3d_cam_to_3dgs_cam storage)."""
    R = np.array(tdgs_cam.R)
    T = np.array(tdgs_cam.T)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = T
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1
    opengl_to_pt3d = np.diag([-1.0, 1.0, -1.0, 1.0])
    c2w_pt3d = c2w @ opengl_to_pt3d
    c2w_pt3d[:3, 3] /= XYZ_SCALE
    return c2w_pt3d.tolist()


def _save_gaussian_scene_overview(
    kf_gen: KeyframeGen,
    gaussians: GaussianModel,
    opt: GSParams,
    background: torch.Tensor,
    scene_out: Path,
    overview_width: int = 1536,
) -> Path:
    """Render and save a wide 3DGS overview (run.py render_current_scene save path)."""
    with torch.no_grad():
        pt3d_cam = kf_gen.get_camera_by_js_view_matrix(
            VIEW_MATRIX_FIXED,
            xyz_scale=XYZ_SCALE,
            big_view=True,
        )
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(pt3d_cam, xyz_scale=XYZ_SCALE)
        tdgs_cam.image_width = overview_width
        render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
        image = render_pkg["render"]
    out_path = scene_out / "rendered_img.png"
    ToPILImage()(image).save(out_path)
    return out_path


def _sky_images_dir(example_name: str) -> Path:
    return Path("examples") / "sky_images" / example_name


def _bootstrap_sky_pointcloud(
    kf_gen: KeyframeGen,
    inpainter_pipeline,
    config,
    *,
    example_name: str,
    style_prompt: str,
) -> None:
    """
    Regenerate sky panorama and point cloud for this pano scene.

    Unlike run_ww_worldscore.bootstrap_sky_pointcloud, always loads SyncDiffusion
    and forces gen_sky=True so partial cached sky_*.png cannot leave the model
    as None while generate_sky_pointcloud still needs to sample layer 1.
    """
    sky_dir = _sky_images_dir(example_name)
    if sky_dir.exists():
        shutil.rmtree(sky_dir)
    sky_dir.mkdir(parents=True, exist_ok=True)

    sync_device = (
        config.get("bcdm_device", config["device"])
        if config.get("use_flux")
        else config["device"]
    )
    syncdiffusion_model = SyncDiffusion(sync_device, sd_version="2.0-inpaint")

    sky_mask = kf_gen.generate_sky_mask().float()
    if not sky_mask.bool().any().item():
        print(
            "[WARN] No sky pixels found; using the top image band to initialize "
            "the sky point cloud."
        )
        sky_mask = sky_mask.clone()
        sky_mask[:128, :] = 1.0

    inpainter_home_device = (
        getattr(inpainter_pipeline, "device", None)
        if syncdiffusion_model is not None
        else None
    )
    if hasattr(inpainter_pipeline, "to"):
        inpainter_pipeline.to("cpu")
        _empty_cache()
    try:
        kf_gen.generate_sky_pointcloud(
            syncdiffusion_model,
            image=kf_gen.image_latest,
            mask=sky_mask,
            gen_sky=True,
            style=style_prompt,
        )
    finally:
        del syncdiffusion_model
        _empty_cache()
        if inpainter_home_device is not None and hasattr(inpainter_pipeline, "to"):
            inpainter_pipeline.to(inpainter_home_device)


def _save_panorama_strip(view_paths: list[Path], scene_out: Path) -> Path:
    """Stitch rendered yaw views into one horizontal panoramic image."""
    images = [Image.open(path).convert("RGB") for path in view_paths]
    total_w = sum(img.width for img in images)
    max_h = max(img.height for img in images)
    strip = Image.new("RGB", (total_w, max_h))
    x_offset = 0
    for img in images:
        strip.paste(img, (x_offset, 0))
        x_offset += img.width
    out_path = scene_out / "panorama_scene.png"
    strip.save(out_path)
    return out_path


def _train_sky_gaussians(
    kf_gen: KeyframeGen,
    config,
    background: torch.Tensor,
    save_dir: Path,
) -> GaussianModel:
    """Train sky 3DGS following run.py."""
    traindatas = kf_gen.convert_to_3dgs_traindata(
        xyz_scale=XYZ_SCALE,
        remove_threshold=None,
        use_no_loss_mask=False,
    )
    if config["gen_layer"]:
        _, traindata_sky, _ = traindatas
    else:
        _, traindata_sky = traindatas

    gaussians = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9)
    sky_opt = GSParams()
    sky_opt.max_screen_size = 100
    sky_opt.scene_extent = 1.5
    sky_opt.densify_from_iter = 200
    sky_opt.prune_from_iter = 200
    sky_opt.densify_grad_threshold = 1.0
    sky_opt.iterations = 399

    ww_module.kf_gen = kf_gen
    ww_module.background = background

    scene = Scene(traindata_sky, gaussians, sky_opt, is_sky=True)
    dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
    sky_save = save_dir / f"{dt_string}_gaussian_scene_sky"
    train_gaussian(gaussians, scene, sky_opt, sky_save, initialize_scaling=False)

    gaussians.visibility_filter_all = torch.zeros(
        gaussians.get_xyz_all.shape[0], dtype=torch.bool, device=config["device"]
    )
    gaussians.delete_mask_all = torch.zeros(
        gaussians.get_xyz_all.shape[0], dtype=torch.bool, device=config["device"]
    )
    gaussians.is_sky_filter = torch.ones(
        gaussians.get_xyz_all.shape[0], dtype=torch.bool, device=config["device"]
    )
    return gaussians


def _run_expansion_step(
    kf_gen: KeyframeGen,
    gaussians: GaussianModel,
    opt: GSParams,
    config,
    background: torch.Tensor,
    inpainting_prompt: str,
    camera_idx: int,
    scene_index: int,
    save_dir: Path,
    adaptive_negative_prompt: str = "",
) -> GaussianModel:
    """Single scene-expansion iteration (adapted from run_ww_worldscore.py)."""
    kf_gen.set_kf_param(
        inpainting_resolution=config["inpainting_resolution_gen"],
        inpainting_prompt=inpainting_prompt,
        adaptive_negative_prompt=adaptive_negative_prompt,
    )
    current_pt3d_cam = kf_gen.cameras[camera_idx]
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=XYZ_SCALE)
    kf_gen.set_current_camera(current_pt3d_cam, archive_camera=True)

    with torch.no_grad():
        render_pkg = render(tdgs_cam, gaussians, opt, background)
        render_pkg_nosky = render(tdgs_cam, gaussians, opt, background, exclude_sky=True)

    side_sky_height = 128
    sky_cond_width = 40

    inpaint_mask_0p5_nosky = render_pkg_nosky["final_opacity"] < 0.6
    inpaint_mask_0p0_nosky = render_pkg_nosky["final_opacity"] < 0.01
    inpaint_mask_0p5 = render_pkg["final_opacity"] < 0.6
    inpaint_mask_0p0 = render_pkg["final_opacity"] < 0.01
    fg_mask_0p5_nosky = ~inpaint_mask_0p5_nosky.clone()
    foreground_cols = torch.sum(fg_mask_0p5_nosky == 1, dim=1) > 150
    foreground_cols_idx = torch.nonzero(foreground_cols, as_tuple=True)[1]

    mask_using_full_render = torch.zeros(1, 1, 512, 512, device=config["device"])
    if foreground_cols_idx.numel() > 0:
        min_index = foreground_cols_idx.min().item()
        max_index = foreground_cols_idx.max().item()
        mask_using_full_render[:, :, :, min_index : max_index + 1] = 1
    mask_using_full_render[:, :, :sky_cond_width, :] = 1
    mask_using_full_render[:, :, :side_sky_height, :sky_cond_width] = 1
    mask_using_full_render[:, :, :side_sky_height, -sky_cond_width:] = 1

    mask_using_nosky_render = 1 - mask_using_full_render
    outpaint_condition_image = (
        render_pkg_nosky["render"] * mask_using_nosky_render
        + render_pkg["render"] * mask_using_full_render
    )
    fill_mask = (
        inpaint_mask_0p5_nosky * mask_using_nosky_render
        + inpaint_mask_0p5 * mask_using_full_render
    )
    outpaint_mask = (
        inpaint_mask_0p0_nosky * mask_using_nosky_render
        + inpaint_mask_0p0 * mask_using_full_render
    )
    outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7, device=config["device"]))

    kf_gen.inpaint(
        outpaint_condition_image,
        inpaint_mask=outpaint_mask,
        fill_mask=fill_mask,
        inpainting_prompt=inpainting_prompt,
        mask_strategy=np.max,
        diffusion_steps=50,
    )

    sem_seg = kf_gen.update_sky_mask()
    recomposed = soft_stitching(
        render_pkg["render"], kf_gen.image_latest, kf_gen.sky_mask_latest
    )

    depth_should_be = render_pkg["median_depth"][0:1].unsqueeze(0) / XYZ_SCALE
    mask_to_align_depth = (depth_should_be < 0.006 * 0.8) & (depth_should_be > 0.001)

    ground_mask = kf_gen.generate_ground_mask(sem_map=sem_seg)[None, None]
    depth_should_be_ground = kf_gen.compute_ground_depth(camera_height=0.0003)
    ground_outputable_mask = (depth_should_be_ground > 0.001) & (
        depth_should_be_ground < 0.006 * 0.8
    )

    joint_mask = mask_to_align_depth | (ground_mask & ground_outputable_mask)
    depth_should_be_joint = torch.where(
        mask_to_align_depth, depth_should_be, depth_should_be_ground
    )

    with torch.no_grad():
        kf_gen.get_depth(
            kf_gen.image_latest,
            target_depth=depth_should_be_joint,
            mask_align=joint_mask,
            archive_output=True,
            diffusion_steps=30,
            guidance_steps=30,
        )

    kf_gen.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())
    kf_gen.image_latest = recomposed

    if config["gen_layer"]:
        kf_gen.generate_layer(pred_semantic_map=sem_seg, scene_name=None)
        depth_should_be = kf_gen.depth_latest_init
        mask_to_align_depth = ~(kf_gen.mask_disocclusion.bool()) & (depth_should_be < 0.006 * 0.8)
        mask_to_farther_depth = kf_gen.mask_disocclusion.bool() & (depth_should_be < 0.006 * 0.8)
        with torch.no_grad():
            kf_gen.depth, kf_gen.disparity = kf_gen.get_depth(
                kf_gen.image_latest,
                archive_output=True,
                target_depth=depth_should_be,
                mask_align=mask_to_align_depth,
                mask_farther=mask_to_farther_depth,
                diffusion_steps=30,
                guidance_steps=30,
            )
        kf_gen.refine_disp_with_segments(
            no_refine_mask=ground_mask.squeeze().cpu().numpy(),
            existing_mask=~(kf_gen.mask_disocclusion).bool().squeeze().cpu().numpy(),
            existing_disp=kf_gen.disparity_latest_init.squeeze().cpu().numpy(),
        )
        wrong_depth_mask = kf_gen.depth_latest < kf_gen.depth_latest_init
        kf_gen.depth_latest[wrong_depth_mask] = (
            kf_gen.depth_latest_init[wrong_depth_mask] + 0.0001
        )
        kf_gen.depth_latest = (
            kf_gen.mask_disocclusion * kf_gen.depth_latest
            + (1 - kf_gen.mask_disocclusion) * kf_gen.depth_latest_init
        )
        kf_gen.update_sky_mask()
        valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)
        kf_gen.update_current_pc_by_kf(
            image=kf_gen.image_latest,
            depth=kf_gen.depth_latest,
            valid_mask=valid_px_mask,
        )
        kf_gen.update_current_pc_by_kf(
            image=kf_gen.image_latest_init,
            depth=kf_gen.depth_latest_init,
            valid_mask=kf_gen.mask_disocclusion * outpaint_mask,
            gen_layer=True,
        )
    else:
        valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)
        kf_gen.update_current_pc_by_kf(
            image=kf_gen.image_latest,
            depth=kf_gen.depth_latest,
            valid_mask=valid_px_mask,
        )
    kf_gen.archive_latest()

    if config["gen_layer"]:
        traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(
            xyz_scale=XYZ_SCALE
        )
        if has_traindata_points(traindata_layer):
            gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
            scene = Scene(traindata_layer, gaussians, opt)
            dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            layer_save = save_dir / f"{dt_string}_gaussian_scene_layer{scene_index:02d}"
            train_gaussian(gaussians, scene, opt, layer_save)
    else:
        traindata = kf_gen.convert_to_3dgs_traindata_latest(
            xyz_scale=XYZ_SCALE, use_no_loss_mask=False
        )

    if not has_traindata_points(traindata):
        gaussians.set_inscreen_points_to_visible(tdgs_cam)
        kf_gen.increment_kf_idx()
        return gaussians

    gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
    scene = Scene(traindata, gaussians, opt)
    dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
    scene_save = save_dir / f"{dt_string}_gaussian_scene{scene_index:02d}"
    train_gaussian(gaussians, scene, opt, scene_save)
    gaussians.set_inscreen_points_to_visible(tdgs_cam)
    kf_gen.increment_kf_idx()
    return gaussians


def generate_and_render_panorama(args: Namespace) -> dict:
    """
    1. Load input image
    2. Initialise WonderWorld scene from image + prompt
    3. Build horizontal camera poses
    4. Render each view
    5. Save PNGs and views_manifest.json
    """
    if args.num_views % 2 == 0:
        raise ValueError("--num-views must be odd so a central view exists.")

    base_config = OmegaConf.load(args.base_config)
    config = OmegaConf.merge(base_config, {})
    config.seed = args.seed
    config.device = args.device
    save_dir = Path(args.output_dir)
    # Isolate sky cache per batch output (e.g. pano_flux vs pano_sd15) and scene.
    sky_example_name = f"{args.scene_id}__{save_dir.name}"
    config.runs_dir = str(save_dir)
    config.example_name = sky_example_name
    config.use_gpt = False

    _seed_all(args.seed)
    device = config["device"]
    background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device=device)

    save_dir.mkdir(parents=True, exist_ok=True)
    scene_out = save_dir / args.scene_id
    scene_out.mkdir(parents=True, exist_ok=True)

    rotation_path = list(config["rotation_path"])
    num_scenes = int(config["num_scenes"])
    if len(rotation_path) >= num_scenes:
        rotation_path = rotation_path[:num_scenes]
    else:
        rotation_path = [rotation_path[i % len(rotation_path)] for i in range(num_scenes)]

    segment_processor = OneFormerProcessor.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large"
    )
    segment_model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large"
    ).to(device)
    mask_generator = create_mask_generator_repvit()

    if config["use_flux"]:
        inpainter_pipeline = BackbonePipeline(
            offload=False,
            device=str(config.get("bcdm_device", device)),
        )

    depth_model = MarigoldPipeline.from_pretrained(
        "prs-eth/marigold-v1-0", torch_dtype=torch.bfloat16
    ).to(device)
    depth_model.scheduler = EulerDiscreteScheduler.from_config(
        depth_model.scheduler.config
    )
    depth_model.scheduler = prepare_scheduler(depth_model.scheduler)

    normal_estimator = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v0-1", torch_dtype=torch.bfloat16
    ).to(device)

    kf_gen = KeyframeGen(
        config=config,
        inpainter_pipeline=inpainter_pipeline,
        mask_generator=mask_generator,
        depth_model=depth_model,
        segment_model=segment_model,
        segment_processor=segment_processor,
        normal_estimator=normal_estimator,
        rotation_path=rotation_path,
        inpainting_resolution=config["inpainting_resolution_gen"],
    ).to(device)

    ww_module.kf_gen = kf_gen
    ww_module.background = background

    start_keyframe = Image.open(args.input_image).convert("RGB").resize((512, 512))
    kf_gen.image_latest = ToTensor()(start_keyframe).unsqueeze(0).to(device)

    _bootstrap_sky_pointcloud(
        kf_gen,
        inpainter_pipeline,
        config,
        example_name=sky_example_name,
        style_prompt=args.prompt,
    )
    kf_gen.recompose_image_latest_and_set_current_pc(scene_name=args.scene_id)

    gaussians = _train_sky_gaussians(kf_gen, config, background, save_dir)

    opt = GSParams()
    if config["gen_layer"]:
        traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(
            xyz_scale=XYZ_SCALE
        )
        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        if has_traindata_points(traindata_layer):
            scene = Scene(traindata_layer, gaussians, opt)
            dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            train_gaussian(
                gaussians,
                scene,
                opt,
                save_dir / f"{dt_string}_gaussian_scene_layer00",
            )
    else:
        traindata = kf_gen.convert_to_3dgs_traindata_latest(
            xyz_scale=XYZ_SCALE, use_no_loss_mask=False
        )

    if has_traindata_points(traindata):
        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata, gaussians, opt)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        train_gaussian(
            gaussians,
            scene,
            opt,
            save_dir / f"{dt_string}_gaussian_scene00",
        )

    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(
        kf_gen.get_camera_at_origin(), xyz_scale=XYZ_SCALE
    )
    gaussians.set_inscreen_points_to_visible(tdgs_cam)
    kf_gen.increment_kf_idx()

    inpainting_prompt = args.prompt
    for step in range(num_scenes):
        camera_idx = step + 1
        if camera_idx >= len(kf_gen.cameras):
            print(
                f"[WARN] Camera index {camera_idx} out of range "
                f"({len(kf_gen.cameras)} cameras); stopping expansion."
            )
            break
        gaussians = _run_expansion_step(
            kf_gen=kf_gen,
            gaussians=gaussians,
            opt=opt,
            config=config,
            background=background,
            inpainting_prompt=inpainting_prompt,
            camera_idx=camera_idx,
            scene_index=step + 1,
            save_dir=save_dir,
        )

    yaw_offsets = np.linspace(-args.yaw_range, args.yaw_range, args.num_views).tolist()
    camera_params = CameraParams()
    canonical_c2w = _pt3d_camera_to_c2w(kf_gen.get_camera_at_origin())
    tdgs_cameras = build_horizontal_camera_poses(
        canonical_c2w, yaw_offsets, camera_params
    )

    central_index = args.num_views // 2
    views_meta: list[dict] = []
    view_paths: list[Path] = []
    to_pil = ToPILImage()

    with torch.no_grad():
        for view_idx, (yaw_deg, tdgs_cam_view) in enumerate(
            zip(yaw_offsets, tdgs_cameras)
        ):
            render_pkg = render(
                tdgs_cam_view, gaussians, opt, background, render_visible=True
            )
            image = render_pkg["render"]
            filename = f"view_{view_idx:04d}.png"
            out_path = scene_out / filename
            to_pil(image).save(out_path)
            view_paths.append(out_path)

            views_meta.append(
                {
                    "view_index": view_idx,
                    "yaw_deg": float(yaw_deg),
                    "pitch_deg": 0.0,
                    "image_path": filename,
                    "c2w": _camera_to_c2w_list(tdgs_cam_view),
                }
            )

    panorama_path = _save_panorama_strip(view_paths, scene_out)
    overview_path = _save_gaussian_scene_overview(
        kf_gen, gaussians, opt, background, scene_out
    )

    manifest = {
        "scene_id": args.scene_id,
        "prompt": args.prompt,
        "input_image": str(Path(args.input_image).resolve()),
        "num_views": args.num_views,
        "central_view_index": central_index,
        "yaw_range_deg": args.yaw_range,
        "panorama_scene_path": panorama_path.name,
        "rendered_img_path": overview_path.name,
        "views": views_meta,
    }
    manifest_path = scene_out / "views_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _empty_cache()
    return manifest


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate panoramic views from a single image.")
    parser.add_argument("--input-image", required=True, help="Path to conditioning image.")
    parser.add_argument("--prompt", required=True, help="Fixed text prompt for generation.")
    parser.add_argument("--output-dir", required=True, help="Output directory for views.")
    parser.add_argument("--scene-id", required=True, help="Scene identifier for output naming.")
    parser.add_argument(
        "--yaw-range",
        type=float,
        default=60.0,
        help="Half-range of horizontal sweep in degrees (default: 60).",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=9,
        help="Number of views to render (must be odd, default: 9).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda or cpu).")
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="WonderWorld base config YAML.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    manifest = generate_and_render_panorama(args)
    print(f"Saved {manifest['num_views']} views for scene {manifest['scene_id']}.")


if __name__ == "__main__":
    main()
