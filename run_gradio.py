"""WonderWorld Gradio demo - stream-viewer adapted UI with session reset and presets."""
import gc
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from datetime import datetime
import threading
import gradio as gr
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from diffusers import AutoencoderKL, DDIMScheduler, EulerDiscreteScheduler

# TEST FLUX
from diffusers import FluxFillPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from diffusers import Flux2KleinPipeline
from backbone.edit.controller import BCDMPipeline
from backbone.edit.geom_controller import EditPipeline
from backbone.edit.socket_edit import (
    EditPayloadError,
    build_socket_geometry_spec,
    decode_edit_payload,
    image_to_png_data_url,
    masked_source_for_caption,
)

from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldPipelineNormal, MarigoldNormalsPipeline

from models.models import KeyframeGen, save_point_cloud_as_ply, BCDM_MASK_GAUSSIAN_BLUR_RADIUS
from util.gs_utils import save_pc_as_3dgs, convert_pc_to_splat
from util.internlm import TextpromptGen as GeminiTextpromptGen, TextpromptGen
# from util.gemini_prompt_gen import GeminiTextpromptGen as TextpromptGen, GeminiTextpromptGen
from util.general_utils import apply_depth_colormap, save_video
from util.utils import save_depth_map, prepare_scheduler, soft_stitching
from util.utils import load_example_yaml, convert_pt3d_cam_to_3dgs_cam
from util.segment_utils import create_mask_generator_repvit
from util.free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
 
from arguments import GSParams, CameraParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.loss import l1_loss, ssim
from scene.cameras import Camera
from random import randint
import time
import cv2
from syncdiffusion.syncdiffusion_model import SyncDiffusion
from kornia.morphology import dilation
import warnings
import os
import copy
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS on the Flask app
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for SocketIO

xyz_scale = 1000
client_id = None
scene_name = None
view_matrix = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_wonder = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_delete = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

view_matrix_fixed = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0.2, 0.5, 1]
])
theta = np.radians(-3)
rotation_matrix_x = np.array([
    [1, 0, 0, 0],
    [0, np.cos(theta), -np.sin(theta), 0],
    [0, np.sin(theta), np.cos(theta), 0],
    [0, 0, 0, 1]
])
view_matrix_fixed = np.dot(view_matrix_fixed, rotation_matrix_x)
view_matrix_fixed = view_matrix_fixed.flatten().tolist()

background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device='cuda')
latest_frame = None
latest_viz = None
keep_rendering = True
iter_number = None
kf_gen = None
gaussians = None
opt = None
scene_dict = None
style_prompt = None
pt_gen = None
change_scene_name_by_user = False
undo = False
save = False
delete = False
exclude_sky = False
runtime_config = None
edit_pipeline = None
edit_pipeline_config = None
edit_lock = threading.Lock()

# Event object used to control the synchronization
start_event = threading.Event()
gen_event = threading.Event()

# Gradio session control
uploaded_image = {"path": None, "example_name": None, "is_preset": False}
new_image_event = threading.Event()
reset_event = threading.Event()
status_message = "Starting server..."

PRESET_CONFIGS = {
    "venice": {
        "example_name": "venice",
        "runs_dir": "output/venice",
        "gen_sky_image": False,
        "gen_sky": False,
        "gen_layer": True,
        "use_gpt": False,
        "load_gen": False,
    },
    "main_square": {
        "example_name": "main_square",
        "runs_dir": "output/main_square",
        "gen_sky_image": False,
        "gen_sky": False,
        "gen_layer": True,
        "use_gpt": False,
        "load_gen": False,
    },
    "real_campus_2": {
        "example_name": "real_campus_2",
        "runs_dir": "output/real_campus_2",
        "gen_sky_image": False,
        "gen_sky": False,
        "gen_layer": True,
        "use_gpt": False,
        "load_gen": False,
    },
}


def _set_status(msg):
    global status_message
    status_message = msg
    if client_id is not None:
        socketio.emit("server-state", msg, room=client_id)


def build_session_config(base_config, upload_info):
    if upload_info.get("is_preset") and upload_info.get("example_name") in PRESET_CONFIGS:
        extra = OmegaConf.create(PRESET_CONFIGS[upload_info["example_name"]])
        return OmegaConf.merge(base_config, extra)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"custom_{ts}"
    extra = OmegaConf.create({
        "example_name": name,
        "runs_dir": f"output/{name}",
        "gen_sky_image": True,
        "gen_sky": False,
        "gen_layer": True,
        "use_gpt": False,
        "load_gen": False,
    })
    return OmegaConf.merge(base_config, extra)


def reset_session_state():
    global view_matrix, view_matrix_wonder, view_matrix_delete, keep_rendering
    global undo, save, delete, exclude_sky, change_scene_name_by_user
    global gaussians, kf_gen, pt_gen, scene_dict, scene_name, iter_number
    view_matrix = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    view_matrix_wonder = list(view_matrix)
    view_matrix_delete = list(view_matrix)
    keep_rendering = True
    undo = save = delete = False
    exclude_sky = False
    change_scene_name_by_user = False
    gaussians = None
    kf_gen = None
    pt_gen = None
    scene_dict = None
    scene_name = None
    iter_number = None


def _session_aborted():
    return reset_event.is_set()

def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"running with seed: {seed}.")


def run_one_session(
    config,
    img_path,
    yaml_data,
    inpainter_pipeline,
    mask_generator,
    depth_model,
    segment_model,
    segment_processor,
    normal_estimator,
):
    global client_id, view_matrix, scene_name, latest_frame, keep_rendering, kf_gen, latest_viz, gaussians, opt, background, scene_dict, style_prompt, pt_gen, change_scene_name_by_user, undo, save, delete, exclude_sky, view_matrix_delete, runtime_config, edit_pipeline, edit_pipeline_config

    runtime_config = config
    example = config["example_name"]
    rotation_path = config["rotation_path"][: config["num_scenes"]]
    assert len(rotation_path) == config["num_scenes"]

    reset_session_state()
    keep_rendering = True

    print("###### ------------------ Keyframe generation (session) ------------------ ######")
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
    ).to(config["device"])

    content_prompt = yaml_data["content_prompt"]
    style_prompt = yaml_data["style_prompt"]
    adaptive_negative_prompt = yaml_data["negative_prompt"]
    background_prompt = yaml_data.get("background", None)
    control_text = yaml_data.get("control_text", None)
    outdoor = yaml_data.get("outdoor", False)
    if adaptive_negative_prompt != "":
        adaptive_negative_prompt += ", "

    start_keyframe = Image.open(img_path).convert("RGB").resize((512, 512))
    kf_gen.image_latest = ToTensor()(start_keyframe).unsqueeze(0).to(config["device"])

    if _session_aborted():
        return

    needs_sync = config["gen_sky_image"] or (
        not os.path.exists(f"examples/sky_images/{example}/sky_0.png")
        and not os.path.exists(f"examples/sky_images/{example}/sky_1.png")
    )
    if needs_sync:
        _set_status("SyncDiffusion generating sky - please wait...")
    else:
        _set_status("Loading cached sky - please wait...")

    if config['gen_sky_image'] or (not os.path.exists(f'examples/sky_images/{example}/sky_0.png') and not os.path.exists(f'examples/sky_images/{example}/sky_1.png')):
        syncdiffusion_model = SyncDiffusion(config["bcdm_device"], sd_version='2.0-inpaint')
    else:
        syncdiffusion_model = None
    sky_mask = kf_gen.generate_sky_mask().float()

    # Free cuda:1 while SyncDiffusion generates all sky images, then restore.
    # inpainter_pipeline (BCDMPipeline) and edit_pipeline share the same
    # underlying wrapper, so moving one moves both.  For the SD case the
    # inpainter lives on a different GPU (config["device"]) so offloading it
    # is a no-op cost-wise but still safe.
    _inpainter_home_device = (
        inpainter_pipeline.device          # BCDMPipeline: stored str, never changes
        if config["use_flux"]
        else str(inpainter_pipeline.device)  # HF pipeline: reflect before offload
    ) if syncdiffusion_model is not None else None
    if syncdiffusion_model is not None:
        inpainter_pipeline.to('cpu')
        empty_cache()
    try:
        kf_gen.generate_sky_pointcloud(syncdiffusion_model, image=kf_gen.image_latest, mask=sky_mask, gen_sky=config['gen_sky_image'], style=style_prompt)
    finally:
        # Release SyncDiffusion memory before restoring the inpainter so the
        # restore itself does not trigger another OOM.
        syncdiffusion_model = None
        empty_cache()
        if _inpainter_home_device is not None:
            inpainter_pipeline.to(_inpainter_home_device)

    if _session_aborted():
        return

    kf_gen.recompose_image_latest_and_set_current_pc(scene_name=scene_name)

    pt_gen = TextpromptGen(kf_gen.run_dir, isinstance(control_text, list))
    
    content_list = content_prompt.split(',')
    if config["use_flux"]:
        scene_name = content_prompt
        entities = content_list[1:]
    else:
        scene_name = content_list[0]
        entities = content_list[1:]
    scene_dict = {'scene_name': scene_name, 'entities': entities, 'style': style_prompt, 'background': background_prompt}
    inpainting_prompt = content_prompt
    if isinstance(pt_gen, GeminiTextpromptGen):
        pt_gen.set_initial_resolved_state(scene_dict, style=style_prompt)
    socketio.emit('scene-prompt', scene_name, room=client_id)

    kf_gen.increment_kf_idx()
    ###### ------------------ Main loop ------------------ ######

    if config['gen_sky'] or not os.path.exists(f'examples/sky_images/{example}/finished_3dgs_sky_tanh.ply'):
        traindatas = kf_gen.convert_to_3dgs_traindata(xyz_scale=xyz_scale, remove_threshold=None, use_no_loss_mask=False)
        if config['gen_layer']:
            traindata, traindata_sky, traindata_layer = traindatas
        else:
            traindata, traindata_sky = traindatas
        gaussians = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9)
        opt = GSParams()
        opt.max_screen_size = 100  # Sky is supposed to be big; set a high max screen size
        opt.scene_extent = 1.5  # Sky is supposed to be big; set a high scene extent
        opt.densify_from_iter = 200  # Need to do some densify
        opt.prune_from_iter = 200  # Don't prune for sky because sky 3DGS are supposed to be big; prevent it by setting a high prune iter
        opt.densify_grad_threshold = 1.0  # Do not need to densify; Set a high threshold to prevent densifying
        opt.iterations = 399  # More iterations than 100 needed for sky
        scene = Scene(traindata_sky, gaussians, opt, is_sky=True)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_sky"
        train_gaussian(gaussians, scene, opt, save_dir, initialize_scaling=False)
        gaussians.save_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs_sky_tanh.ply')
    else:
        gaussians = GaussianModel(sh_degree=0)
        gaussians.load_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs_sky_tanh.ply')  # pure sky

    gaussians.visibility_filter_all = torch.zeros(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
    gaussians.delete_mask_all = torch.zeros(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
    gaussians.is_sky_filter = torch.ones(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
    
    if config['load_gen'] and os.path.exists(f'examples/sky_images/{example}/finished_3dgs.ply') and os.path.exists(f'examples/sky_images/{example}/visibility_filter_all.pth') and os.path.exists(f'examples/sky_images/{example}/is_sky_filter.pth') and os.path.exists(f'examples/sky_images/{example}/delete_mask_all.pth'):
        print("Loading existing 3DGS...")
        gaussians = GaussianModel(sh_degree=0)
        gaussians.load_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs.ply')
        gaussians.visibility_filter_all = torch.load(f'examples/sky_images/{example}/visibility_filter_all.pth').to('cuda')
        gaussians.is_sky_filter = torch.load(f'examples/sky_images/{example}/is_sky_filter.pth').to('cuda')
        gaussians.delete_mask_all = torch.load(f'examples/sky_images/{example}/delete_mask_all.pth').to('cuda')
    opt = GSParams()

    ### First scene 3DGS
    if config['gen_layer']:
        traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata_layer, gaussians, opt)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_layer{0:02d}"
        train_gaussian(gaussians, scene, opt, save_dir)  # Base layer training
    else:
        traindata = kf_gen.convert_to_3dgs_traindata_latest(xyz_scale=xyz_scale, use_no_loss_mask=False)

    gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
    scene = Scene(traindata, gaussians, opt)
    dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
    i = 0
    save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene{i:02d}"
    train_gaussian(gaussians, scene, opt, save_dir)

    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale)
    gaussians.set_inscreen_points_to_visible(tdgs_cam)

    if _session_aborted():
        return

    def llm_prompt_generation(event):
        global scene_dict, style_prompt, pt_gen, change_scene_name_by_user, scene_name
        while True:
            event.wait()
            print("-- start llm...")
            scene_dict = pt_gen.wonder_next_scene(scene_name=scene_name, entities=scene_dict['entities'], style=style_prompt, background=scene_dict['background'], change_scene_name_by_user=change_scene_name_by_user)
            change_scene_name_by_user = False
            print("-- llm done.")
            event.clear()
        
    if config['use_gpt']:
        llm_event = threading.Event()
        llm_thread = threading.Thread(target=llm_prompt_generation, args=(llm_event, ))
        llm_thread.daemon = True
        llm_thread.start()
    
    gaussians_tmp = copy.deepcopy(gaussians)
    _set_status("Ready - navigate the scene. Use Outpaint scene to expand.")
    while True:
        if _session_aborted():
            empty_cache()
            return

        inpainting_prompt = pt_gen.generate_prompt(style=style_prompt, entities=scene_dict['entities'], background=scene_dict['background'], scene_name=scene_dict['scene_name'])
        scene_name = scene_dict['scene_name'] if isinstance(scene_dict['scene_name'], str) else scene_dict['scene_name'][0]
        i += 1
        
        socketio.emit('scene-prompt', scene_name, room=client_id)
        print('Waiting for scene gen signal...')
        _set_status('Waiting to generate new scenes...')

        while keep_rendering:
            if _session_aborted():
                empty_cache()
                return
            time.sleep(0.05)
            if delete:
                print("Deleting...")
                current_pt3d_cam_delete = kf_gen.get_camera_by_js_view_matrix(view_matrix_delete, xyz_scale=xyz_scale)
                tdgs_cam_delete = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam_delete, xyz_scale=xyz_scale)
                gaussians.delete_points(tdgs_cam_delete)
                delete = False
            if save:
                print("Saving...")
                gaussians.save_ply_all_with_filter(f'examples/sky_images/{example}/finished_3dgs.ply')
                torch.save(gaussians.visibility_filter_all, f'examples/sky_images/{example}/visibility_filter_all.pth')
                torch.save(gaussians.is_sky_filter, f'examples/sky_images/{example}/is_sky_filter.pth')
                torch.save(gaussians.delete_mask_all, f'examples/sky_images/{example}/delete_mask_all.pth')
                gaussians.yield_splat_data(f'examples/sky_images/{example}/{example}_finished_3dgs.splat')
                save = False

        if _session_aborted():
            empty_cache()
            return
        
        if undo:
            print("Undoing...")
            gaussians = copy.deepcopy(gaussians_tmp)
            undo = False
        else:
            print("Not undo...")
            gaussians_tmp = copy.deepcopy(gaussians)
             
        _set_status('Generating new scene...')
        if _session_aborted():
            empty_cache()
            return
        
        # LLM prompt generation
        if config['use_gpt']:
            llm_event.set()
        
        if config['use_gpt']:
            scene_dict = pt_gen.wonder_next_scene(scene_name=scene_name, entities=scene_dict['entities'], style=style_prompt, background=scene_dict['background'], change_scene_name_by_user=change_scene_name_by_user)
            change_scene_name_by_user = False

        inpainting_prompt = pt_gen.generate_prompt(style=style_prompt, entities=scene_dict['entities'], background=scene_dict['background'], scene_name=scene_dict['scene_name'])
        scene_name = scene_dict['scene_name'] if isinstance(scene_dict['scene_name'], str) else scene_dict['scene_name'][0]

        ###### ------------------ Keyframe (the major part of point clouds) generation ------------------ ######        
        
        # Keyframe generation
        kf_gen.set_kf_param(inpainting_resolution=config['inpainting_resolution_gen'],
                            inpainting_prompt=inpainting_prompt, adaptive_negative_prompt=adaptive_negative_prompt)
        current_pt3d_cam = kf_gen.get_camera_by_js_view_matrix(view_matrix, xyz_scale=xyz_scale)
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=xyz_scale)
        kf_gen.set_current_camera(current_pt3d_cam, archive_camera=True)
        
        if exclude_sky:
            with torch.no_grad():
                render_pkg = render(tdgs_cam, gaussians, opt, background)
                render_pkg_nosky = render(tdgs_cam, gaussians, opt, background, exclude_sky=True)
            
            side_sky_height = 128

            inpaint_mask_0p5_nosky = (render_pkg_nosky["final_opacity"]<0.6)
            inpaint_mask_0p0_nosky = (render_pkg_nosky["final_opacity"]<0.01)  # Should not have holes in existing regions
            inpaint_mask_0p5 = (render_pkg["final_opacity"]<0.6)
            inpaint_mask_0p0 = (render_pkg["final_opacity"]<0.01)  # Should not have holes in existing regions

            mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config['device'])
            mask_using_full_render[:, :, :side_sky_height, :] = 1
            
            mask_using_nosky_render = 1 - mask_using_full_render
                
            outpaint_condition_image = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render
            viz = outpaint_condition_image[0].permute(1, 2, 0).detach().cpu().numpy()
            viz = (viz * 255).astype(np.uint8)
            viz = viz[..., ::-1]
            # latest_viz = viz
            fill_mask = inpaint_mask_0p5_nosky * mask_using_nosky_render + inpaint_mask_0p5 * mask_using_full_render
            outpaint_mask = inpaint_mask_0p0_nosky * mask_using_nosky_render + inpaint_mask_0p0 * mask_using_full_render
            outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())
            exclude_sky = False
        else:
            with torch.no_grad():
                render_pkg = render(tdgs_cam, gaussians, opt, background)
                render_pkg_nosky = render(tdgs_cam, gaussians, opt, background, exclude_sky=True)
            
            side_sky_height = 128
            sky_cond_width = 40

            inpaint_mask_0p5_nosky = (render_pkg_nosky["final_opacity"]<0.6)
            inpaint_mask_0p0_nosky = (render_pkg_nosky["final_opacity"]<0.01)  # Should not have holes in existing regions
            inpaint_mask_0p5 = (render_pkg["final_opacity"]<0.6)
            inpaint_mask_0p0 = (render_pkg["final_opacity"]<0.01)  # Should not have holes in existing regions
            fg_mask_0p5_nosky = ~inpaint_mask_0p5_nosky.clone()
            foreground_cols = torch.sum(fg_mask_0p5_nosky == 1, dim=1)>150  # [1, 512]
            foreground_cols_idx = torch.nonzero(foreground_cols, as_tuple=True)[1]

            mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config['device'])
            if foreground_cols_idx.numel() > 0:
                min_index = foreground_cols_idx.min().item()
                max_index = foreground_cols_idx.max().item()
                mask_using_full_render[:, :, :, min_index:max_index+1] = 1
            mask_using_full_render[:, :, :sky_cond_width, :] = 1
            mask_using_full_render[:, :, :side_sky_height, :sky_cond_width] = 1
            mask_using_full_render[:, :, :side_sky_height, -sky_cond_width:] = 1
            
            mask_using_nosky_render = 1 - mask_using_full_render
                
            outpaint_condition_image = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render
            viz = outpaint_condition_image[0].permute(1, 2, 0).detach().cpu().numpy()
            viz = (viz * 255).astype(np.uint8)
            viz = viz[..., ::-1]
            # latest_viz = viz
            fill_mask = inpaint_mask_0p5_nosky * mask_using_nosky_render + inpaint_mask_0p5 * mask_using_full_render
            outpaint_mask = inpaint_mask_0p0_nosky * mask_using_nosky_render + inpaint_mask_0p0 * mask_using_full_render
            outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())

        # Widen the region for adding Gaussians under FLUX/BCDM: matches BCDM mask blur edge (PIL) in
        # models.models.BCDM_MASK_GAUSSIAN_BLUR_RADIUS and KeyframeGen.inpaint.
        if config["use_flux"]:
            k = 5 * int(np.ceil(BCDM_MASK_GAUSSIAN_BLUR_RADIUS)) + 1
            k = min(k, 31)
            outpaint_mask_for_new_points = (dilation(
                outpaint_mask.float(), kernel=torch.ones(k, k, device=config["device"])
            ) > 0.5).to(outpaint_mask.dtype)
        else:
            outpaint_mask_for_new_points = outpaint_mask
        
        bcdm_src, bcdm_tgt = None, None
        if isinstance(pt_gen, GeminiTextpromptGen) and config["use_flux"]:
            bcdm_src, bcdm_tgt = pt_gen.build_bcdm_inpaint_pair_from_conditioning_image(
                outpaint_condition_image, style_prompt, scene_dict
            )

        # Content inpainting
        # Measure time for inpainting
        time_start = time.time()
        inpaint_output = kf_gen.inpaint(outpaint_condition_image, inpaint_mask=outpaint_mask, fill_mask=fill_mask, inpainting_prompt=inpainting_prompt, mask_strategy=np.max, diffusion_steps=50, bcdm_prompt_src=bcdm_src, bcdm_prompt_tgt=bcdm_tgt)
        time_end = time.time()
        print(f"Inpainting time: {time_end - time_start} seconds")

        # TODO: Prune edited regions (mask union of all edited regions)
        # TODO: Edit output (equivalent to inpainting entire image)

        sem_seg = kf_gen.update_sky_mask()
        recomposed = soft_stitching(render_pkg["render"], kf_gen.image_latest, kf_gen.sky_mask_latest)  # Replace generated sky with rendered sky

        depth_should_be = render_pkg['median_depth'][0:1].unsqueeze(0) / xyz_scale
        mask_to_align_depth = (depth_should_be < 0.006 * 0.8) & (depth_should_be > 0.001)  # If opacity < 0.5, then median_depth = -1

        ground_mask = kf_gen.generate_ground_mask(sem_map=sem_seg)[None, None]
        depth_should_be_ground = kf_gen.compute_ground_depth(camera_height=0.0003)
        ground_outputable_mask = (depth_should_be_ground > 0.001) & (depth_should_be_ground < 0.006 * 0.8)

        joint_mask = mask_to_align_depth | (ground_mask & ground_outputable_mask)
        depth_should_be_joint = torch.where(mask_to_align_depth, depth_should_be, depth_should_be_ground)

        with torch.no_grad():
            depth_guide_joint, _ = kf_gen.get_depth(kf_gen.image_latest, target_depth=depth_should_be_joint, mask_align=joint_mask, archive_output=True, 
                                                    diffusion_steps=30, guidance_steps=8)

        kf_gen.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())

        kf_gen.image_latest = recomposed
        if config['gen_layer']:
            # Measure time for layer generation
            time_start = time.time()
            kf_gen.generate_layer(pred_semantic_map=sem_seg, scene_name=scene_name)
            time_end = time.time()
            print(f"Layer generation time: {time_end - time_start} seconds")

            depth_should_be = kf_gen.depth_latest_init
            mask_to_align_depth = ~(kf_gen.mask_disocclusion.bool()) & (depth_should_be < 0.006 * 0.8)
            mask_to_farther_depth = kf_gen.mask_disocclusion.bool() & (depth_should_be < 0.006 * 0.8)
            # Measure time for depth generation
            time_start = time.time()
            with torch.no_grad():
                kf_gen.depth, kf_gen.disparity = kf_gen.get_depth(kf_gen.image_latest, archive_output=True, target_depth=depth_should_be, mask_align=mask_to_align_depth, mask_farther=mask_to_farther_depth,
                                                                  diffusion_steps=30, guidance_steps=8)
            time_end = time.time()
            print(f"Depth generation time: {time_end - time_start} seconds")

            kf_gen.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy(),
                                             existing_mask=~(kf_gen.mask_disocclusion).bool().squeeze().cpu().numpy(),
                                             existing_disp=kf_gen.disparity_latest_init.squeeze().cpu().numpy())
            wrong_depth_mask = kf_gen.depth_latest<kf_gen.depth_latest_init
            kf_gen.depth_latest[wrong_depth_mask] = kf_gen.depth_latest_init[wrong_depth_mask] + 0.0001
            kf_gen.depth_latest = kf_gen.mask_disocclusion * kf_gen.depth_latest + (1-kf_gen.mask_disocclusion) * kf_gen.depth_latest_init
            kf_gen.update_sky_mask()
            valid_px_mask = outpaint_mask_for_new_points * (~kf_gen.sky_mask_latest)
            kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest, depth=kf_gen.depth_latest, valid_mask=valid_px_mask)  # Base only
            kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest_init, depth=kf_gen.depth_latest_init, valid_mask=kf_gen.mask_disocclusion*outpaint_mask_for_new_points, gen_layer=True)  # Object layer
        else:
            valid_px_mask = outpaint_mask_for_new_points * (~kf_gen.sky_mask_latest)
            kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest, depth=kf_gen.depth_latest, valid_mask=valid_px_mask)
        kf_gen.archive_latest()

        gaussian_training_time = 0
        if config['gen_layer']:
            traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
            gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
            scene = Scene(traindata_layer, gaussians, opt)
            dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_layer{i+1:02d}"

            # Measure time for training
            time_start = time.time()
            train_gaussian(gaussians, scene, opt, save_dir)  # Base layer training
            time_end = time.time()
            gaussian_training_time += time_end - time_start
        else:
            traindata = kf_gen.convert_to_3dgs_traindata_latest(xyz_scale=xyz_scale, use_no_loss_mask=False)

        if traindata['pcd_points'].shape[-1] == 0:
            gaussians.set_inscreen_points_to_visible(tdgs_cam)

            kf_gen.increment_kf_idx()
            keep_rendering = True
            continue
        
        mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config['device'])
        x = torch.sum(fg_mask_0p5_nosky == 1, dim=2)>0  # [1, 512]
        x_idx = torch.nonzero(x, as_tuple=True)[1]
        if foreground_cols_idx.numel() > 0:
            min_index = foreground_cols_idx.min().item()
            max_index = foreground_cols_idx.max().item()
            mask_using_full_render[:, :, :x_idx.max().item(), min_index:max_index+1] = 1
        # mask_using_full_render[:, :, :sky_cond_width, :] = 1
        # mask_using_full_render[:, :, :side_sky_height, :sky_cond_width] = 1
        # mask_using_full_render[:, :, :side_sky_height, -sky_cond_width:] = 1
        
        mask_using_nosky_render = 1 - mask_using_full_render
        image_tmp = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render
        
        
        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata, gaussians, opt)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene{i+1:02d}"

        # Measure time for training
        time_start = time.time()
        train_gaussian(gaussians, scene, opt, save_dir)
        time_end = time.time()
        gaussian_training_time += time_end - time_start

        print(f"Gaussian training time: {gaussian_training_time} seconds")
        gaussians.set_inscreen_points_to_visible(tdgs_cam)

        kf_gen.increment_kf_idx()
        keep_rendering = True
        empty_cache()
        if _session_aborted():
            return


def run_gradio(base_config):
    """Load models once, then run scene sessions until process exit."""
    global edit_pipeline, edit_pipeline_config, kf_gen

    edit_pipeline_config = _load_edit_pipeline_config()
    seeding(base_config["seed"])

    segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    segment_model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large"
    ).to("cuda")
    mask_generator = create_mask_generator_repvit()

    if base_config["use_flux"]:
        inpainter_pipeline = BCDMPipeline(offload=False, model="klein", device=base_config["bcdm_device"])
        edit_pipeline = EditPipeline(base_pipeline=inpainter_pipeline)
    else:
        inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            base_config["stable_diffusion_checkpoint"],
            safety_checker=None,
            torch_dtype=torch.bfloat16,
        ).to(base_config["device"])
        inpainter_pipeline.scheduler = DDIMScheduler.from_config(inpainter_pipeline.scheduler.config)
        inpainter_pipeline.unet.set_attn_processor(AttnProcessor2_0())
        inpainter_pipeline.vae.set_attn_processor(AttnProcessor2_0())
        edit_pipeline = None

    depth_model = MarigoldPipeline.from_pretrained(
        "prs-eth/marigold-depth-v1-0", torch_dtype=torch.bfloat16
    ).to(base_config["device"])
    depth_model.scheduler = EulerDiscreteScheduler.from_config(depth_model.scheduler.config)
    depth_model.scheduler = prepare_scheduler(depth_model.scheduler)

    normal_estimator = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v0-1", torch_dtype=torch.bfloat16
    ).to(base_config["device"])

    print("###### ------------------ Gradio demo ready ------------------ ######")

    while True:
        reset_event.clear()
        reset_session_state()
        _set_status("Ready - drop an image or pick an example.")

        new_image_event.wait()
        new_image_event.clear()
        if reset_event.is_set():
            continue

        upload = dict(uploaded_image)
        if not upload.get("path"):
            continue

        session_config = build_session_config(base_config, upload)
        example_name = session_config["example_name"]

        if upload.get("is_preset"):
            yaml_data = load_example_yaml(example_name, "examples/examples.yaml")
            status = "Loading preset (cached sky) - please wait..."
        else:
            yaml_data = {
                "content_prompt": "A scenic outdoor view, landscape and architecture, natural lighting",
                "style_prompt": "DSLR 35mm landscape",
                "negative_prompt": "",
                "background": "",
                "control_text": None,
                "outdoor": True,
            }
            status = "SyncDiffusion generating sky - please wait..."

        _set_status(status)
        print(f"Starting session: {example_name} from {upload['path']}")

        try:
            run_one_session(
                session_config,
                upload["path"],
                yaml_data,
                inpainter_pipeline,
                mask_generator,
                depth_model,
                segment_model,
                segment_processor,
                normal_estimator,
            )
        except Exception as exc:
            print(f"Session error: {exc}")
            _set_status(f"Session error: {exc}")
            empty_cache()

        if reset_event.is_set():
            _set_status("Reset - drop an image or pick an example.")
        empty_cache()


def train_gaussian(gaussians: GaussianModel, scene: Scene, opt: GSParams, save_dir: Path, initialize_scaling=True):
    global latest_frame, iter_number, view_matrix, latest_viz
    iterable_gauss = range(1, opt.iterations + 1)
    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=trainCameras, initialize_scaling=initialize_scaling)

    for iteration in iterable_gauss:
        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # import pdb; pdb.set_trace()
        # Render
        render_pkg = render(viewpoint_cam, gaussians, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if iteration == opt.iterations:
        # if iteration % 5 == 0 or iteration == 1:
            time.sleep(0.1)
            print(f'Iteration {iteration}, Loss: {loss.item()}')
            with torch.no_grad():
                tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_by_js_view_matrix(view_matrix, xyz_scale=xyz_scale), xyz_scale=xyz_scale)
                render_pkg = render(tdgs_cam, gaussians, opt, background)
                image = render_pkg['render']
                # rendered_normal = render_pkg['render_normal']
                # rendered_normal_map = rendered_normal/2-0.5
            rendered_image = image.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            rendered_image = rendered_image[..., ::-1]
            latest_frame = rendered_image
        loss.backward()
        if iteration == opt.iterations:
            print(f'Final loss: {loss.item()}')

        # Use variables that related to the trainable GS
        n_trainable = gaussians.get_xyz.shape[0]
        viewspace_point_tensor_grad, visibility_filter, radii = viewspace_point_tensor.grad[:n_trainable], visibility_filter[:n_trainable], radii[:n_trainable]

        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    max_screen_size = opt.max_screen_size if iteration >= opt.prune_from_iter else None
                    camera_height = 0.0003 * xyz_scale
                    scene_extent = camera_height * 2 if opt.scene_extent is None else opt.scene_extent
                    opacity_lowest = 0.05
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opacity_lowest, scene_extent, max_screen_size)
                    gaussians.compute_3D_filter(cameras=trainCameras)
                
                # if (iteration % opt.opacity_reset_interval == 0 
                #     or (opt.white_background and iteration == opt.densify_from_iter)
                # ):
                #     gaussians.reset_opacity()

            # if iteration % 100 == 0 and iteration > opt.densify_until_iter:
            #     if iteration < opt.iterations - 100:
            #         # don't update in the end of training
            #         gaussians.compute_3D_filter(cameras=trainCameras)
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)


def _load_edit_pipeline_config():
    config_path = Path(__file__).resolve().parent / "backbone" / "configs" / "geom_edit_pipeline.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return {k: v for k, v in raw.items() if k not in ("geometry", "inputs")}


def _emit_edit_status(room, stage, message):
    socketio.emit("edit_status", {"stage": stage, "message": message}, room=room)


def _emit_edit_error(room, message, details=None):
    payload = {"message": message}
    if details is not None:
        payload["details"] = details
    socketio.emit("edit_error", payload, room=room)


def _process_edit_submit(data, room):
    global pt_gen, edit_pipeline, edit_pipeline_config, runtime_config

    if not edit_lock.acquire(blocking=False):
        _emit_edit_error(room, "Another edit is already running. Please wait for it to finish.")
        return

    try:
        if edit_pipeline is None:
            raise RuntimeError("Edit geometry requires config['use_flux'] so the BCDM/FLUX pipeline is available.")
        if pt_gen is None:
            raise RuntimeError("Gemini prompt generator is not ready yet.")
        if runtime_config is None or edit_pipeline_config is None:
            raise RuntimeError("Wonderworld runtime is not initialized yet.")

        _emit_edit_status(room, "payload_received", "Edit payload received.")
        decoded = decode_edit_payload(data)
        _emit_edit_status(room, "images_decoded", "All edit images decoded as 512x512 PNGs.")
        _emit_edit_status(room, "mask_processed", "Source and target masks normalized to binary masks.")

        masked_source = masked_source_for_caption(decoded.source_image, decoded.source_mask)
        _emit_edit_status(room, "original_region_removed", "Original selected region removed for captioning.")

        source_caption = pt_gen.describe_edit_source_without_mask(masked_source)
        _emit_edit_status(room, "caption_generated", "Source-scene caption generated.")

        composed_caption = None

        def describe_composed(composed_image):
            nonlocal composed_caption
            _emit_edit_status(room, "flux_pass_complete", "First edit composition pass complete.")
            composed_caption = pt_gen.describe_composed_edit_image(composed_image)
            _emit_edit_status(room, "composed_image_described", "Composed image description generated.")
            return composed_caption

        spec = build_socket_geometry_spec(decoded, source_caption, source_caption)
        output_dir = str(Path(kf_gen.run_dir) / "edit_geom") if kf_gen is not None else "outputs/edit_geom"
        result = edit_pipeline.run(
            src_image=decoded.source_image,
            tgt_image=decoded.target_image,
            spec=spec,
            config=edit_pipeline_config,
            output_dir=output_dir,
            composition_prompt_callback=describe_composed,
        )

        _apply_edit_result_to_scene(
            result,
            decoded,
            room,
            runtime_config,
            composed_caption or source_caption,
            source_caption,
        )
        socketio.emit(
            "edit_result",
            {
                "image": image_to_png_data_url(result),
                "metadata": {
                    "edit_type": decoded.edit_type,
                    "prompt_src": source_caption,
                    "prompt_tgt": composed_caption or source_caption,
                },
            },
            room=room,
        )
    except EditPayloadError as exc:
        _emit_edit_error(room, str(exc), {"type": exc.__class__.__name__})
    except Exception as exc:
        _emit_edit_error(room, "Edit processing failed.", {"type": exc.__class__.__name__, "message": str(exc)})
        raise
    finally:
        edit_lock.release()


def _apply_edit_result_to_scene(result_image, decoded, room, config, scene_description, layer_scene_description=None):
    global kf_gen, gaussians, opt, background, view_matrix_wonder

    if kf_gen is None or gaussians is None or opt is None:
        raise RuntimeError("Cannot apply edit before the Wonderworld scene is initialized.")

    device = config["device"]
    current_pt3d_cam = kf_gen.get_camera_by_js_view_matrix(view_matrix_wonder, xyz_scale=xyz_scale)
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=xyz_scale)
    kf_gen.set_current_camera(current_pt3d_cam, archive_camera=True)
    with torch.no_grad():
        edit_render_pkg = render(tdgs_cam, gaussians, opt, background)
    kf_gen.image_latest = ToTensor()(result_image.convert("RGB")).unsqueeze(0).to(device)

    source_region = decoded.source_mask_tensor if decoded.removes_source_region else torch.zeros_like(decoded.source_mask_tensor)
    edit_mask = torch.maximum(source_region, decoded.target_mask_tensor).unsqueeze(0).to(device)
    edit_mask_float = edit_mask.float()
    if config["use_flux"]:
        k = 5 * int(np.ceil(BCDM_MASK_GAUSSIAN_BLUR_RADIUS)) + 1
        k = min(k, 31)
        dilated_edit_mask = (dilation(edit_mask_float, kernel=torch.ones(k, k, device=device)) > 0.5).to(edit_mask_float.dtype)
    else:
        dilated_edit_mask = edit_mask_float

    gaussians.delete_regions(tdgs_cam, edit_mask)
    _emit_edit_status(room, "existing_region_removed", "Existing Gaussians removed from the edited region.")

    _emit_edit_status(room, "depth_prediction_started", "Depth prediction started.")
    with torch.no_grad():
        kf_gen.get_depth(kf_gen.image_latest, archive_output=True, diffusion_steps=30, guidance_steps=8)
    sem_seg = kf_gen.update_sky_mask()
    recomposed = soft_stitching(edit_render_pkg["render"], kf_gen.image_latest, kf_gen.sky_mask_latest)
    depth_should_be = edit_render_pkg["median_depth"][0:1].unsqueeze(0) / xyz_scale
    mask_to_align_depth = (depth_should_be < 0.006 * 0.8) & (depth_should_be > 0.001)
    ground_mask = kf_gen.generate_ground_mask(sem_map=sem_seg)[None, None]
    depth_should_be_ground = kf_gen.compute_ground_depth(camera_height=0.0003)
    ground_outputable_mask = (depth_should_be_ground > 0.001) & (depth_should_be_ground < 0.006 * 0.8)
    joint_mask = mask_to_align_depth | (ground_mask & ground_outputable_mask)
    depth_should_be_joint = torch.where(mask_to_align_depth, depth_should_be, depth_should_be_ground)
    with torch.no_grad():
        kf_gen.get_depth(
            kf_gen.image_latest,
            target_depth=depth_should_be_joint,
            mask_align=joint_mask,
            archive_output=True,
            diffusion_steps=30,
            guidance_steps=8,
        )
    kf_gen.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())
    kf_gen.image_latest = recomposed
    _emit_edit_status(room, "depth_prediction_completed", "Depth prediction completed.")

    if config["gen_layer"]:
        _emit_edit_status(room, "layer_generation_started", "Layer generation started.")
        layer_prompt = layer_scene_description or scene_description
        kf_gen.generate_layer(
            pred_semantic_map=sem_seg,
            scene_name=layer_prompt,
            force_mask_disocclusion=dilated_edit_mask.bool(),
        )

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
                guidance_steps=8,
            )
        kf_gen.refine_disp_with_segments(
            no_refine_mask=ground_mask.squeeze().cpu().numpy(),
            existing_mask=~(kf_gen.mask_disocclusion).bool().squeeze().cpu().numpy(),
            existing_disp=kf_gen.disparity_latest_init.squeeze().cpu().numpy(),
        )
        wrong_depth_mask = kf_gen.depth_latest < kf_gen.depth_latest_init
        kf_gen.depth_latest[wrong_depth_mask] = kf_gen.depth_latest_init[wrong_depth_mask] + 0.0001
        kf_gen.depth_latest = (
            kf_gen.mask_disocclusion * kf_gen.depth_latest
            + (1 - kf_gen.mask_disocclusion) * kf_gen.depth_latest_init
        )
        kf_gen.update_sky_mask()
        valid_px_mask = dilated_edit_mask * (~kf_gen.sky_mask_latest)
        kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest, depth=kf_gen.depth_latest, valid_mask=valid_px_mask)
        kf_gen.update_current_pc_by_kf(
            image=kf_gen.image_latest_init,
            depth=kf_gen.depth_latest_init,
            valid_mask=kf_gen.mask_disocclusion * dilated_edit_mask,
            gen_layer=True,
        )
        kf_gen.archive_latest()

        traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        layer_scene = Scene(traindata_layer, gaussians, opt)
        save_dir = Path(config["runs_dir"]) / f"{datetime.now().strftime('%d-%m_%H-%M-%S')}_gaussian_scene_edit_layer"
        train_gaussian(gaussians, layer_scene, opt, save_dir)
    else:
        valid_px_mask = dilated_edit_mask * (~kf_gen.sky_mask_latest)
        kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest, depth=kf_gen.depth_latest, valid_mask=valid_px_mask)
        kf_gen.archive_latest()
        traindata = kf_gen.convert_to_3dgs_traindata_latest(xyz_scale=xyz_scale, use_no_loss_mask=False)
        _emit_edit_status(room, "layer_generation_started", "Gaussian scene update started.")

    if traindata["pcd_points"].shape[-1] == 0:
        gaussians.set_inscreen_points_to_visible(tdgs_cam)
    else:
        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata, gaussians, opt)
        save_dir = Path(config["runs_dir"]) / f"{datetime.now().strftime('%d-%m_%H-%M-%S')}_gaussian_scene_edit"
        train_gaussian(gaussians, scene, opt, save_dir)
        gaussians.set_inscreen_points_to_visible(tdgs_cam)

    _emit_edit_status(room, "layer_generation_completed", "Layer generation and Gaussian scene update completed.")

    kf_gen.increment_kf_idx()
    empty_cache()

def start_server(port):
    # Bind loopback only; Gradio proxies /socket.io on the public port (7860).
    socketio.run(app, host="127.0.0.1", port=port, allow_unsafe_werkzeug=True)

@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)
    global client_id
    client_id = request.sid

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)
    global client_id
    client_id = None

@socketio.on('start')
def handle_start(data):
    print("Client connected:", request.sid)
    print('Received start signal.')
    start_event.set()  # Signal the main program to proceed

@socketio.on('gen')
def handle_gen(data):
    print('Received gen signal. Camera matrix: ', data)
    global view_matrix, keep_rendering
    keep_rendering = False
    view_matrix = data

@socketio.on('render-pose')
def handle_render_pose(data):
    global view_matrix_wonder, keep_rendering
    view_matrix_wonder = data

@socketio.on('scene-prompt')
def handle_new_prompt(data):
    assert isinstance(data, str)
    print('Received new scene prompt: ' + data)
    global scene_name, change_scene_name_by_user
    scene_name = data
    change_scene_name_by_user = True

@socketio.on('undo')
def handle_undo():
    print('Received undo signal.')
    global undo
    undo = True

@socketio.on('save')
def handle_save():
    print('Received save signal.')
    global save
    save = True

@socketio.on('delete')
def handle_delete(data):
    print('Received delete signal.')
    global delete, view_matrix_delete
    delete = True
    view_matrix_delete = data

@socketio.on('fill_hole')
def handle_fill_hole():
    print('Received fill hole signal.')
    global exclude_sky
    exclude_sky = True 

@socketio.on('edit_submit')
def handle_edit_submit(data):
    print('Received edit submit signal.')
    socketio.start_background_task(_process_edit_submit, data, request.sid)
    
    
# opt_render = GSParams()
def render_current_scene():
    global latest_frame, client_id, iter_number, latest_viz, kf_gen, gaussians, opt, background, view_matrix_wonder, save
    while True:
        time.sleep(0.05)
        try:
            with torch.no_grad():
                tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_by_js_view_matrix(view_matrix_wonder, xyz_scale=xyz_scale), xyz_scale=xyz_scale)
                render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
            rendered_img = render_pkg['render']
            rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            rendered_image = rendered_image[..., ::-1]
            latest_frame = rendered_image

            with torch.no_grad():
                tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_by_js_view_matrix(view_matrix_fixed, xyz_scale=xyz_scale, big_view=True), xyz_scale=xyz_scale)
                tdgs_cam.image_width = 1536
                # tdgs_cam.image_height = 1024
                render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
            rendered_img = render_pkg['render']
            rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            rendered_image = rendered_image[..., ::-1]
            latest_viz = rendered_image
            if save:
                ToPILImage()(rendered_img).save(kf_gen.run_dir / 'rendered_img.png')
        except Exception as e:
            pass

        if latest_frame is not None:
            image_bytes = cv2.imencode('.jpg', latest_frame)[1].tobytes()
            if client_id is not None:
                socketio.emit('frame', image_bytes, room=client_id)
                socketio.emit('iter-number', f'Iter: {iter_number}', room=client_id)
            else:
                socketio.emit('frame', image_bytes)
                socketio.emit('iter-number', f'Iter: {iter_number}')
        if latest_viz is not None:
            image_bytes = cv2.imencode('.jpg', latest_viz)[1].tobytes()
            if client_id is not None:
                socketio.emit('viz', image_bytes, room=client_id)
            else:
                socketio.emit('viz', image_bytes)

def on_image_upload(img_path):
    if not img_path:
        return status_message
    reset_event.set()
    uploaded_image.update({"path": img_path, "example_name": None, "is_preset": False})
    reset_event.clear()
    new_image_event.set()
    return "SyncDiffusion generating sky - please wait..."


def on_preset(example_name):
    path = f"examples/images/{example_name}.png"
    if not os.path.exists(path):
        return f"Example image not found: {path}"
    reset_event.set()
    uploaded_image.update({"path": path, "example_name": example_name, "is_preset": True})
    reset_event.clear()
    new_image_event.set()
    return "Loading preset (cached sky) - please wait..."


def on_reset():
    reset_event.set()
    uploaded_image.update({"path": None, "example_name": None, "is_preset": False})
    new_image_event.set()  # wake session loop if blocked
    return "Reset - drop an image or pick an example."


def poll_status():
    return status_message


def build_viewer_html():
    return _VIEWER_HTML


_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
    }
)


def create_combined_app(demo, internal_sio_port: int) -> FastAPI:
    """Gradio UI + viewer page + Socket.IO proxy on one port (e.g. 7860)."""
    fastapi_app = FastAPI()
    fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/")

    @fastapi_app.get("/ww/viewer")
    async def ww_viewer_page():
        return HTMLResponse(build_viewer_html())

    @fastapi_app.get("/ww-api/frame")
    async def ww_api_frame():
        if latest_frame is None:
            return Response(status_code=204)
        ok, buf = cv2.imencode(".jpg", latest_frame)
        if not ok:
            return Response(status_code=204)
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    @fastapi_app.get("/ww-api/viz")
    async def ww_api_viz():
        if latest_viz is None:
            return Response(status_code=204)
        ok, buf = cv2.imencode(".jpg", latest_viz)
        if not ok:
            return Response(status_code=204)
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    async def _proxy_socketio(request: Request, path: str = ""):
        qs = str(request.url.query)
        suffix = f"/{path}" if path else "/"
        target = f"http://127.0.0.1:{internal_sio_port}/socket.io{suffix}"
        if qs:
            target = f"{target}?{qs}"
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS
        }
        async with httpx.AsyncClient() as client:
            upstream = await client.request(
                request.method,
                target,
                headers=headers,
                content=await request.body(),
                timeout=120.0,
            )
        out_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in _HOP_HEADERS and k.lower() != "content-length"
        }
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=out_headers,
        )

    for route_path in ("/socket.io", "/socket.io/"):
        fastapi_app.add_api_route(
            route_path,
            _proxy_socketio,
            methods=["GET", "POST", "OPTIONS", "HEAD"],
        )
    fastapi_app.add_api_route(
        "/socket.io/{path:path}",
        _proxy_socketio,
        methods=["GET", "POST", "OPTIONS", "HEAD"],
    )

    return fastapi_app


def create_gradio_demo():
    with gr.Blocks(title="WonderWorld Demo", css=".viewer-frame iframe { min-height: 720px; }") as demo:
        gr.Markdown(
            "## WonderWorld Interactive Demo\n"
            "Drop an image to generate a 3D scene (SyncDiffusion runs for custom uploads). "
            "Presets use cached sky. Click the viewer, then navigate with WASD / arrows; "
            "use **Outpaint scene** instead of R."
        )
        with gr.Row():
            image_in = gr.Image(
                label="Drop image to generate scene",
                type="filepath",
                sources=["upload", "clipboard"],
                height=200,
            )
            status_tb = gr.Textbox(label="Status", value=status_message, interactive=False)
            reset_btn = gr.Button("Reset", variant="stop")
        gr.Markdown("**Example scenes** (pre-generated sky, no SyncDiffusion):")
        with gr.Row():
            btn_venice = gr.Button("Venice")
            btn_main = gr.Button("Main Square")
            btn_campus = gr.Button("Real Campus 2")
        btn_venice.click(fn=lambda: on_preset("venice"), outputs=status_tb)
        btn_main.click(fn=lambda: on_preset("main_square"), outputs=status_tb)
        btn_campus.click(fn=lambda: on_preset("real_campus_2"), outputs=status_tb)
        # Full viewer at /ww/viewer (scripts run); Gradio embeds via same-origin iframe.
        viewer_html = gr.HTML(
            '<iframe src="/ww/viewer" title="WonderWorld viewer" '
            'style="width:100%;min-height:780px;height:80vh;border:1px solid #ccc;border-radius:8px;"></iframe>',
            elem_classes=["viewer-frame"],
        )

        image_in.upload(on_image_upload, inputs=image_in, outputs=status_tb)
        reset_btn.click(on_reset, outputs=status_tb)

        # Gradio 4+: periodic status sync (demo.load(..., every=) was removed)
        if hasattr(gr, "Timer"):
            gr.Timer(1).tick(fn=poll_status, outputs=status_tb)

    return demo


_VIEWER_HTML = r"""
<div id="ww-viewer-root" tabindex="-1" style="outline:none;font-family:sans-serif;position:relative;min-height:700px;background:#fff;">
<style>
#ww-viewer-root{overflow:hidden}
#ww-prompt-box{position:absolute;top:10px;left:130px;z-index:10;border:2px solid #ccc;border-radius:5px;padding:4px 8px;width:280px}
#ww-send-button,#ww-outpaint-btn{position:absolute;top:10px;z-index:10;background:#007bff;color:#fff;border:2px solid #ccc;border-radius:5px;padding:6px 10px;cursor:pointer}
#ww-send-button{left:10px}
#ww-outpaint-btn{left:420px}
#ww-main-wrap{position:absolute;top:40px;left:20px;z-index:10}
#ww-main-stack{position:relative;display:inline-block;line-height:0}
#ww-canvas,#ww-mask-overlay{display:block;border:3px solid #000;touch-action:none}
#ww-mask-overlay{position:absolute;left:0;top:0;pointer-events:none}
#ww-fg-wrap{position:absolute;inset:0;pointer-events:none;z-index:2}
#ww-fg{position:absolute;border:2px dashed rgba(0,100,200,.9);background:rgba(0,120,255,.12);pointer-events:auto;transform-origin:center;cursor:move}
#ww-viz-col{position:absolute;top:40px;left:560px;display:flex;flex-direction:column;gap:10px}
#ww-canvas-viz{border:2px solid #333}
#ww-edit-panel{background:#fff;border:1px solid #ccc;border-radius:8px;padding:10px;width:300px;max-height:400px;overflow:auto}
#ww-server-state{position:absolute;bottom:10px;left:10px;font-size:13px;color:#333}
#ww-quality{position:absolute;bottom:10px;right:10px;font-size:13px}
#ww-caminfo{position:absolute;top:10px;right:10px;font-size:12px}
#ww-mask-dialog{position:fixed;inset:0;background:rgba(0,0,0,.45);display:none;align-items:center;justify-content:center;z-index:9999}
#ww-mask-dialog.open{display:flex}
#ww-mask-inner{background:#fff;padding:12px;border-radius:8px}
</style>
<input id="ww-prompt-box" type="text" placeholder="Scene prompt..." />
<button type="button" id="ww-send-button">Next scene is ..</button>
<button type="button" id="ww-outpaint-btn">Outpaint scene</button>
<div id="ww-main-wrap">
  <button type="button" id="ww-edit-btn" style="margin-bottom:6px">Edit view</button>
  <div id="ww-main-stack">
    <canvas id="ww-canvas" width="512" height="512"></canvas>
    <canvas id="ww-mask-overlay" width="512" height="512" style="display:none"></canvas>
    <div id="ww-fg-wrap"><div id="ww-fg" style="display:none;width:140px;height:88px"></div></div>
  </div>
</div>
<div id="ww-viz-col">
  <canvas id="ww-canvas-viz" width="768" height="256"></canvas>
  <div id="ww-edit-panel" style="display:none">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <b>Edit mode</b><button type="button" id="ww-edit-close">Close</button>
    </div>
    <label>Mode</label>
    <select id="ww-edit-mode">
      <option value="manipulation">Manipulation</option>
      <option value="addition">Addition</option>
      <option value="copy">Copy</option>
      <option value="replacement">Replacement</option>
    </select>
    <div style="margin-top:8px"><button type="button" id="ww-draw-mask">Draw mask</button></div>
    <div style="margin-top:8px"><button type="button" id="ww-choose-target">Choose target image</button>
      <input type="file" id="ww-target-file" accept="image/*" style="display:none" /></div>
    <div style="margin-top:8px"><button type="button" id="ww-manual-mask">Manual target mask</button>
      <button type="button" id="ww-upload-mask">Upload mask</button>
      <input type="file" id="ww-mask-file" accept="image/*" style="display:none" /></div>
    <div style="margin-top:10px"><button type="button" id="ww-edit-submit" style="background:#28a745;color:#fff;padding:6px 12px;border:none;border-radius:4px">Submit</button></div>
  </div>
</div>
<div id="ww-server-state"></div>
<div id="ww-quality"><span id="ww-fps"></span></div>
<div id="ww-caminfo"><span id="ww-iter"></span><br><span id="ww-camid"></span></div>
<div id="ww-mask-dialog"><div id="ww-mask-inner">
  <h3 id="ww-mask-title">Draw mask</h3>
  <button type="button" id="ww-mask-brush">Brush</button>
  <button type="button" id="ww-mask-eraser">Eraser</button>
  <div style="position:relative;margin:8px 0"><img id="ww-mask-bg" style="max-width:480px;display:block" /><canvas id="ww-mask-canvas"></canvas></div>
  <button type="button" id="ww-mask-cancel">Cancel</button>
  <button type="button" id="ww-mask-apply">Apply mask</button>
</div></div>
<script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
<script>
(() => {
const SOCKET_URL = window.location.origin;
const SIZE = 512;
const defaultViewMatrix = [-1,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,1];
function invert4(a){const b00=a[0]*a[5]-a[1]*a[4],b01=a[0]*a[6]-a[2]*a[4],b02=a[0]*a[7]-a[3]*a[4],b03=a[1]*a[6]-a[2]*a[5],b04=a[1]*a[7]-a[3]*a[5],b05=a[2]*a[7]-a[3]*a[6],b06=a[8]*a[13]-a[9]*a[12],b07=a[8]*a[14]-a[10]*a[12],b08=a[8]*a[15]-a[11]*a[12],b09=a[9]*a[14]-a[10]*a[13],b10=a[9]*a[15]-a[11]*a[13],b11=a[10]*a[15]-a[11]*a[14],det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;if(!det)return null;return[(a[5]*b11-a[6]*b10+a[7]*b09)/det,(a[2]*b10-a[1]*b11-a[3]*b09)/det,(a[13]*b05-a[14]*b04+a[15]*b03)/det,(a[10]*b04-a[9]*b05-a[11]*b03)/det,(a[6]*b08-a[4]*b11-a[7]*b07)/det,(a[0]*b11-a[2]*b08+a[3]*b07)/det,(a[14]*b02-a[12]*b05-a[15]*b01)/det,(a[8]*b05-a[10]*b02+a[11]*b01)/det,(a[4]*b10-a[5]*b08+a[7]*b06)/det,(a[1]*b08-a[0]*b10-a[3]*b06)/det,(a[12]*b04-a[13]*b02+a[15]*b00)/det,(a[9]*b02-a[8]*b04-a[11]*b00)/det,(a[5]*b07-a[4]*b09-a[6]*b06)/det,(a[0]*b09-a[1]*b07+a[2]*b06)/det,(a[13]*b01-a[12]*b03-a[14]*b00)/det,(a[8]*b03-a[9]*b01+a[10]*b00)/det]}
function rotate4(a,rad,x,y,z){let len=Math.hypot(x,y,z);x/=len;y/=len;z/=len;const s=Math.sin(rad),c=Math.cos(rad),t=1-c,b00=x*x*t+c,b01=y*x*t+z*s,b02=z*x*t-y*s,b10=x*y*t-z*s,b11=y*y*t+c,b12=z*y*t+x*s,b20=x*z*t+y*s,b21=y*z*t-x*s,b22=z*z*t+c;return[a[0]*b00+a[4]*b01+a[8]*b02,a[1]*b00+a[5]*b01+a[9]*b02,a[2]*b00+a[6]*b01+a[10]*b02,a[3]*b00+a[7]*b01+a[11]*b02,a[0]*b10+a[4]*b11+a[8]*b12,a[1]*b10+a[5]*b11+a[9]*b12,a[2]*b10+a[6]*b11+a[10]*b12,a[3]*b10+a[7]*b11+a[11]*b12,a[0]*b20+a[4]*b21+a[8]*b22,a[1]*b20+a[5]*b21+a[9]*b22,a[2]*b20+a[6]*b21+a[10]*b22,a[3]*b20+a[7]*b21+a[11]*b22,...a.slice(12,16)]}
function translate4(a,x,y,z){return[...a.slice(0,12),a[0]*x+a[4]*y+a[8]*z+a[12],a[1]*x+a[5]*y+a[9]*z+a[13],a[2]*x+a[6]*y+a[10]*z+a[14],a[3]*x+a[7]*y+a[11]*z+a[15]]}
const root=document.getElementById('ww-viewer-root');
const mainCanvas=document.getElementById('ww-canvas');
const vizCanvas=document.getElementById('ww-canvas-viz');
const ctx=mainCanvas.getContext('2d');
const ctxViz=vizCanvas.getContext('2d');
const promptBox=document.getElementById('ww-prompt-box');
const serverState=document.getElementById('ww-server-state');
const fpsEl=document.getElementById('ww-fps');
const iterEl=document.getElementById('ww-iter');
const camEl=document.getElementById('ww-camid');
let yaw=0,pitch=0,movement=[0,0,0],viewMatrix=[...defaultViewMatrix],activeKeys=[],editMode=false,editBlock=false;
let fgTransform='translate(186px, 212px) rotate(0deg) scale(1, 1)';
const fgEl=document.getElementById('ww-fg');
const maskOverlay=document.getElementById('ww-mask-overlay');
let editOp='manipulation',targetUrl=null,targetMaskUrl=null,manualMaskUrl=null,sourceBounds=null,mainMaskCommitted=false;
const socket=io(SOCKET_URL,{path:'/socket.io',transports:['polling']});
socket.on('connect',()=>{serverState.textContent='Connected. Waiting for scene...'});
socket.on('connect_error',()=>{serverState.textContent='Connection failed (retrying)...'});
async function pollFrameApi(){try{const r=await fetch('/ww-api/frame');if(r.ok&&r.status!==204){const b=await r.blob();const u=URL.createObjectURL(b);const img=new Image();img.onload=()=>{ctx.drawImage(img,0,0,512,512);URL.revokeObjectURL(u)};img.src=u}}catch(_){}}
async function pollVizApi(){try{const r=await fetch('/ww-api/viz');if(r.ok&&r.status!==204){const b=await r.blob();const u=URL.createObjectURL(b);const img=new Image();img.onload=()=>{ctxViz.drawImage(img,0,0,768,256);URL.revokeObjectURL(u)};img.src=u}}catch(_){}}
setInterval(pollFrameApi,100);setInterval(pollVizApi,200);
socket.on('frame',(data)=>{const b=new Blob([data],{type:'image/jpeg'});const u=URL.createObjectURL(b);const img=new Image();img.onload=()=>{ctx.drawImage(img,0,0,512,512);URL.revokeObjectURL(u)};img.src=u});
socket.on('viz',(data)=>{const b=new Blob([data],{type:'image/jpeg'});const u=URL.createObjectURL(b);const img=new Image();img.onload=()=>{ctxViz.drawImage(img,0,0,768,256);URL.revokeObjectURL(u)};img.src=u});
socket.on('server-state',(m)=>{serverState.textContent=m});
socket.on('scene-prompt',(m)=>{promptBox.value=m});
socket.on('iter-number',(m)=>{iterEl.textContent=m});
setInterval(()=>{if(socket.connected)socket.emit('render-pose',viewMatrix)},1000/60);
function buildViewEmitGen(extraMove,extraYaw){pitch=0;let inv=invert4(defaultViewMatrix);const mov=[movement[0]+(extraMove?.[0]||0),movement[1]+(extraMove?.[1]||0),movement[2]+(extraMove?.[2]||0)];const y=yaw+(extraYaw||0);inv=translate4(inv,...mov);inv=rotate4(inv,y,0,1,0);inv=rotate4(inv,pitch,1,0,0);viewMatrix=invert4(inv);socket.emit('gen',viewMatrix)}
document.getElementById('ww-outpaint-btn').onclick=()=>buildViewEmitGen();
document.getElementById('ww-send-button').onclick=()=>socket.emit('scene-prompt',promptBox.value);
function onKeyDown(e){if(editBlock||document.activeElement===promptBox)return;const k=e.code;
if(k==='KeyZ')socket.emit('undo');if(k==='KeyX')socket.emit('save');if(k==='KeyE')socket.emit('fill_hole');
if(k==='KeyC'){let inv=invert4(defaultViewMatrix);inv=translate4(inv,...movement);inv=rotate4(inv,yaw,0,1,0);inv=rotate4(inv,pitch,1,0,0);socket.emit('delete',invert4(inv))}
if(k==='KeyT')buildViewEmitGen([0,0,-0.8]);if(k==='KeyY')buildViewEmitGen(null,-20*Math.PI/180);if(k==='KeyU')buildViewEmitGen(null,20*Math.PI/180);
if(k==='KeyI')buildViewEmitGen([0,0,-0.5],-15*Math.PI/180);if(k==='KeyO')buildViewEmitGen([0,0,-0.5],15*Math.PI/180);
if(k==='KeyK')buildViewEmitGen([0,0,0.5],-15*Math.PI/180);if(k==='KeyL')buildViewEmitGen([0,0,0.5],15*Math.PI/180);
if(!activeKeys.includes(k))activeKeys.push(k)}
function onKeyUp(e){activeKeys=activeKeys.filter(x=>x!==e.code)}
window.addEventListener('keydown',onKeyDown);window.addEventListener('keyup',onKeyUp);
root.addEventListener('mousedown',()=>root.focus());
let last=0,avgFps=0;
function frame(now){if(editBlock){activeKeys=[];requestAnimationFrame(frame);return}
const sf=0.2;if(activeKeys.includes('KeyA'))yaw-=0.02*sf;if(activeKeys.includes('KeyD'))yaw+=0.02*sf;
if(activeKeys.includes('KeyW'))pitch+=0.005*sf;if(activeKeys.includes('KeyS'))pitch-=0.005*sf;
pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,pitch));
let dx=0,dz=0,dy=0;
if(activeKeys.includes('ArrowUp'))dz+=0.02*sf;if(activeKeys.includes('ArrowDown'))dz-=0.02*sf;
if(activeKeys.includes('ArrowLeft'))dx-=0.02*sf;if(activeKeys.includes('ArrowRight'))dx+=0.02*sf;
if(activeKeys.includes('KeyN'))dy-=0.02*sf;if(activeKeys.includes('KeyM'))dy+=0.02*sf;
const fwd=[Math.sin(yaw)*dz,0,Math.cos(yaw)*dz],rt=[Math.sin(yaw+Math.PI/2)*dx,0,Math.cos(yaw+Math.PI/2)*dx];
movement[0]+=fwd[0]+rt[0];movement[1]+=fwd[1]+rt[1]+dy;movement[2]+=fwd[2]+rt[2];
let inv=invert4(defaultViewMatrix);inv=translate4(inv,...movement);inv=rotate4(inv,yaw,0,1,0);inv=rotate4(inv,pitch,1,0,0);viewMatrix=invert4(inv);
const cfps=last?1000/(now-last):0;avgFps=avgFps*0.9+cfps*0.1;fpsEl.textContent=Math.round(avgFps)+' fps';last=now;requestAnimationFrame(frame)}
requestAnimationFrame(frame);
document.getElementById('ww-edit-btn').onclick=()=>{editMode=true;editBlock=true;document.getElementById('ww-edit-panel').style.display='block';maskOverlay.style.display='block'};
document.getElementById('ww-edit-close').onclick=()=>{editMode=false;editBlock=false;document.getElementById('ww-edit-panel').style.display='none';maskOverlay.style.display='none';fgEl.style.display='none'};
document.getElementById('ww-edit-mode').onchange=(e)=>{editOp=e.target.value;targetUrl=null;sourceBounds=null;mainMaskCommitted=false};
document.getElementById('ww-choose-target').onclick=()=>document.getElementById('ww-target-file').click();
document.getElementById('ww-target-file').onchange=(e)=>{const f=e.target.files?.[0];if(!f)return;targetUrl=URL.createObjectURL(f);fgEl.style.display='block';const img=new Image();img.onload=()=>{const s=Math.min(1,512/img.naturalWidth,512/img.naturalHeight);const w=Math.max(1,Math.round(img.naturalWidth*s)),h=Math.max(1,Math.round(img.naturalHeight*s));fgEl.style.width=w+'px';fgEl.style.height=h+'px';fgEl.innerHTML='';fgEl.appendChild(img);fgTransform='translate('+Math.round((512-w)/2)+'px,'+Math.round((512-h)/2)+'px) rotate(0deg) scale(1,1)';fgEl.style.transform=fgTransform};img.src=targetUrl};
let maskTool='brush',maskDrawing=false,maskLast={x:0,y:0};
const maskDlg=document.getElementById('ww-mask-dialog'),maskCanvas=document.getElementById('ww-mask-canvas'),maskBg=document.getElementById('ww-mask-bg');
function openMaskDlg(kind){const snap=document.createElement('canvas');snap.width=512;snap.height=512;const s=snap.getContext('2d');s.fillStyle='#cdcdcd';s.fillRect(0,0,512,512);try{s.drawImage(mainCanvas,0,0,512,512)}catch(_){}
maskBg.src=kind==='target'&&targetUrl?targetUrl:snap.toDataURL();maskDlg.classList.add('open');const img=new Image();img.onload=()=>{const mw=Math.min(480,img.naturalWidth),mh=Math.min(360,img.naturalHeight);maskCanvas.width=mw;maskCanvas.height=mh;maskCanvas.getContext('2d').clearRect(0,0,mw,mh)};img.src=maskBg.src}
document.getElementById('ww-draw-mask').onclick=()=>openMaskDlg('main');
document.getElementById('ww-manual-mask').onclick=()=>{if(targetUrl)openMaskDlg('target')};
document.getElementById('ww-mask-brush').onclick=()=>maskTool='brush';
document.getElementById('ww-mask-eraser').onclick=()=>maskTool='eraser';
function maskLocal(c,x,y){const r=c.getBoundingClientRect();return{x:(x-r.left)/r.width*c.width,y:(y-r.top)/r.height*c.height}}
maskCanvas.onpointerdown=(e)=>{maskDrawing=true;maskCanvas.setPointerCapture(e.pointerId);const p=maskLocal(maskCanvas,e.clientX,e.clientY);maskLast=p;const mc=maskCanvas.getContext('2d');mc.lineWidth=28;mc.lineCap='round';if(maskTool==='eraser'){mc.globalCompositeOperation='destination-out';mc.strokeStyle='rgba(0,0,0,1)'}else{mc.globalCompositeOperation='source-over';mc.strokeStyle='rgba(255,255,255,.95)'}mc.beginPath();mc.arc(p.x,p.y,14,0,Math.PI*2);mc.fill()};
maskCanvas.onpointermove=(e)=>{if(!maskDrawing)return;const p=maskLocal(maskCanvas,e.clientX,e.clientY);const mc=maskCanvas.getContext('2d');mc.beginPath();mc.moveTo(maskLast.x,maskLast.y);mc.lineTo(p.x,p.y);mc.stroke();maskLast=p};
maskCanvas.onpointerup=()=>{maskDrawing=false};
document.getElementById('ww-mask-cancel').onclick=()=>maskDlg.classList.remove('open');
document.getElementById('ww-mask-apply').onclick=()=>{maskCanvas.toBlob((blob)=>{if(!blob)return;const url=URL.createObjectURL(blob);if(maskBg.src.includes('data:image')||document.getElementById('ww-mask-title').textContent==='Draw mask'){paintMaskOnOverlay(blob)}else{manualMaskUrl=url}maskDlg.classList.remove('open')},'image/png')};
function paintMaskOnOverlay(blob){const url=URL.createObjectURL(blob);const img=new Image();img.onload=()=>{const o=maskOverlay.getContext('2d');o.clearRect(0,0,512,512);o.fillStyle='rgb(255,60,60)';o.fillRect(0,0,512,512);o.globalCompositeOperation='destination-in';o.drawImage(img,0,0,512,512);o.restore();mainMaskCommitted=true;if(editOp==='manipulation'||editOp==='copy'){sourceBounds=getMaskBBox(maskOverlay);if(sourceBounds){fgEl.style.display='block';fgEl.style.left=sourceBounds.x+'px';fgEl.style.top=sourceBounds.y+'px';fgEl.style.width=sourceBounds.w+'px';fgEl.style.height=sourceBounds.h+'px'}}URL.revokeObjectURL(url)};img.src=url}
function getMaskBBox(c){const d=c.getContext('2d').getImageData(0,0,512,512).data;let minX=512,minY=512,maxX=-1,maxY=-1;for(let y=0;y<512;y++)for(let x=0;x<512;x++){const i=(y*512+x)*4;if(d[i+3]>8&&((d[i]+d[i+1]+d[i+2])/3)>32){minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x);maxY=Math.max(maxY,y)}}if(maxX<minX)return null;return{x:minX,y:minY,w:maxX-minX+1,h:maxY-minY+1}}
let dragFg=false,dragStart={x:0,y:0},fgTx=0,fgTy=0;
fgEl.onmousedown=(e)=>{if(!editMode)return;dragFg=true;dragStart={x:e.clientX,y:e.clientY};const m=/translate\(([-\d.]+)px,\s*([-\d.]+)px\)/.exec(fgEl.style.transform||fgTransform);fgTx=parseFloat(m?.[1]||0);fgTy=parseFloat(m?.[2]||0);e.preventDefault()};
window.onmousemove=(e)=>{if(!dragFg)return;const nx=fgTx+e.clientX-dragStart.x,ny=fgTy+e.clientY-dragStart.y;fgTransform='translate('+nx+'px,'+ny+'px) rotate(0deg) scale(1,1)';fgEl.style.transform=fgTransform};
window.onmouseup=()=>{dragFg=false};
function blackMask(){const c=document.createElement('canvas');c.width=512;c.height=512;const x=c.getContext('2d');x.fillStyle='#000';x.fillRect(0,0,512,512);return c.toDataURL('image/png')}
function emptyPng(){const c=document.createElement('canvas');c.width=512;c.height=512;return c.toDataURL('image/png')}
function mainMaskBw(){const c=document.createElement('canvas');c.width=512;c.height=512;const x=c.getContext('2d');x.fillStyle='#000';x.fillRect(0,0,512,512);try{x.drawImage(maskOverlay,0,0)}catch(_){}const d=x.getImageData(0,0,512,512);for(let i=0;i<d.data.length;i+=4){const sel=d.data[i+3]>8&&(d.data[i]+d.data[i+1]+d.data[i+2])/3>32;const v=sel?255:0;d.data[i]=d.data[i+1]=d.data[i+2]=v;d.data[i+3]=255}x.putImageData(d,0,0);return c.toDataURL('image/png')}
document.getElementById('ww-edit-submit').onclick=async()=>{const payload={edit_type:editOp,source_image:mainCanvas.toDataURL('image/png'),source_mask:mainMaskBw(),target_image:emptyPng(),target_mask:blackMask()};
if(editOp==='addition'||editOp==='replacement'){if(!targetUrl){alert('Choose target image');return}payload.target_image=await renderTargetFg();payload.target_mask=manualMaskUrl?await renderTargetMask():blackMask()}
if((editOp==='manipulation'||editOp==='copy')&&sourceBounds){payload.target_image=renderSourceFg();payload.target_mask=renderSourceMask()}
socket.emit('edit_submit',payload);alert('Edit submitted')};
function renderTargetFg(){return new Promise((res)=>{const c=document.createElement('canvas');c.width=512;c.height=512;const o=c.getContext('2d');const img=new Image();img.onload=()=>{const w=fgEl.offsetWidth,h=fgEl.offsetHeight;const masked=document.createElement('canvas');masked.width=w;masked.height=h;masked.getContext('2d').drawImage(img,0,0,w,h);const m=/translate\(([-\d.]+)px,\s*([-\d.]+)px\)/.exec(fgEl.style.transform||'');o.drawImage(masked,parseFloat(m?.[1]||0)+256-w/2,parseFloat(m?.[2]||0)+256-h/2);res(c.toDataURL('image/png'))};img.src=targetUrl})}
async function renderTargetMask(){return blackMask()}
function renderSourceFg(){const c=document.createElement('canvas');c.width=512;c.height=512;const o=c.getContext('2d');const{b}=sourceBounds;const tmp=document.createElement('canvas');tmp.width=b.w;tmp.height=b.h;const t=tmp.getContext('2d');t.drawImage(mainCanvas,b.x,b.y,b.w,b.h,0,0,b.w,b.h);t.globalCompositeOperation='destination-in';t.drawImage(maskOverlay,b.x,b.y,b.w,b.h,0,0,b.w,b.h);o.drawImage(tmp,b.x,b.y);return c.toDataURL('image/png')}
function renderSourceMask(){const c=document.createElement('canvas');c.width=512;c.height=512;const o=c.getContext('2d');o.fillStyle='#000';o.fillRect(0,0,512,512);const{b}=sourceBounds;const tmp=document.createElement('canvas');tmp.width=b.w;tmp.height=b.h;const t=tmp.getContext('2d');t.drawImage(maskOverlay,b.x,b.y,b.w,b.h,0,0,b.w,b.h);const d=t.getImageData(0,0,b.w,b.h);for(let i=0;i<d.data.length;i+=4){const v=d.data[i+3]>8?255:0;d.data[i]=d.data[i+1]=d.data[i+2]=v;d.data[i+3]=255}t.putImageData(d,0,0);o.drawImage(tmp,b.x,b.y);return c.toDataURL('image/png')}
})();
</script>
</div>
"""


if __name__ == "__main__":
    parser = ArgumentParser(description="WonderWorld Gradio demo")
    parser.add_argument("--base-config", default="./config/base-config.yaml", help="Config path")
    parser.add_argument(
        "--port",
        default=7777,
        type=int,
        help="Internal Socket.IO port (loopback only; proxied via Gradio port)",
    )
    parser.add_argument(
        "--gradio-port",
        default=7860,
        type=int,
        help="Public UI port (Gradio + viewer + Socket.IO proxy)",
    )
    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)

    server_thread = threading.Thread(target=start_server, args=(args.port,), daemon=True)
    render_thread = threading.Thread(target=render_current_scene, daemon=True)
    server_thread.start()
    render_thread.start()

    demo = create_gradio_demo()
    demo.queue()
    combined_app = create_combined_app(demo, args.port)

    def _run_uvicorn():
        uvicorn.run(combined_app, host="0.0.0.0", port=args.gradio_port, log_level="info")

    gradio_thread = threading.Thread(target=_run_uvicorn, daemon=True)
    gradio_thread.start()
    print(f"WonderWorld UI: http://127.0.0.1:{args.gradio_port}/  (viewer: /ww/viewer)")

    POSTMORTEM = base_config.get("debug", False)
    if POSTMORTEM:
        try:
            run_gradio(base_config)
        except Exception as e:
            print(e)
            import ipdb
            ipdb.post_mortem()
    else:
        run_gradio(base_config)
