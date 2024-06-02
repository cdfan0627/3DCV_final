#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, DeformModel, SpecModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, safe_normalize, reflect, flip_align_view,  rotation_matrix_from_vectors
from utils.pose_utils import pose_spherical, render_wander_path
from utils.rigid_utils import from_homogenous, to_homogenous
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
from PIL import Image



def render_set(model_path, opt, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, specdecoder):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    # depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    # diffuse_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diffuse")
    # specular_path = os.path.join(model_path, name, "ours_{}".format(iteration), "specular")
    # specular_tint_path = os.path.join(model_path, name, "ours_{}".format(iteration), "specular_tint")
    # new_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "new_normal")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    # makedirs(depth_path, exist_ok=True)
    # makedirs(diffuse_path, exist_ok=True)
    # makedirs(specular_path, exist_ok=True)
    # makedirs(specular_tint_path, exist_ok=True)
    # makedirs(new_normal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        view_pos = view.camera_center.repeat(gaussians.get_opacity.shape[0], 1)
        wo = safe_normalize(view_pos - gaussians.get_xyz.detach())
        dir_pp_normalized = -wo
        normal = gaussians.get_normal(dir_pp_normalized=dir_pp_normalized)
        deform_wo = safe_normalize(view_pos - (gaussians.get_xyz.detach() + d_xyz.detach()))
        deform_dir_pp_normalized = -deform_wo
        deform_normal = gaussians.get_deformnormal(d_rotation, d_scaling, dir_pp_normalized=deform_dir_pp_normalized)
        rotation_matrix = rotation_matrix_from_vectors(normal, deform_normal)
        deform_deltanormal = torch.matmul(rotation_matrix, gaussians.get_delta_normal.unsqueeze(-1)).squeeze(-1)
        #norm_deform_deltanormal = safe_normalize(deform_deltanormal)
        new_normal = safe_normalize(deform_normal + deform_deltanormal)
        reflvec = safe_normalize(reflect(deform_wo, new_normal))
        spat = torch.cat([gaussians.get_xyz.detach() + d_xyz.detach(), gaussians.get_roughness * torch.abs(gaussians.get_scaling.detach() + d_scaling.detach()), gaussians.get_rotation.detach() + d_rotation.detach()], dim=-1)
        spec_color = specdecoder.step(spat, reflvec)
        results = render(view, gaussians, opt, pipeline, background, d_xyz, d_rotation, d_scaling, spec_color, new_normal, iteration, is_6dof)
        rendering = results["render"]
        # depth = results["depth"]
        # diffuse = results["diffuse"]
        # specular = results["specular_color"]
        # specular_tint = results["specular_tint"]
        #new_render_normal = results["new_normal"]

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(diffuse, os.path.join(diffuse_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(specular, os.path.join(specular_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(specular_tint, os.path.join(specular_tint_path, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(new_render_normal, os.path.join(new_normal_path, '{0:05d}'.format(idx) + ".png"))



def render_sets(dataset: ModelParams, opt: OptimizationParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path, iteration)
        specdecoder = SpecModel()
        specdecoder.load_weights(dataset.model_path, iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set

        if not skip_train:
            render_func(dataset.model_path, opt, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, deform, specdecoder)

        if not skip_test:
            render_func(dataset.model_path, opt, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform, specdecoder)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), op.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode)
