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
import flow_viz
from PIL import Image



def render_set(model_path, opt, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, specdecoder):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    diffuse_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diffuse")
    specular_path = os.path.join(model_path, name, "ours_{}".format(iteration), "specular")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    flow_path = os.path.join(model_path, name, "ours_{}".format(iteration), "flow")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(diffuse_path, exist_ok=True)
    makedirs(specular_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(flow_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:
            view1 = view
            continue
        if load2gpu_on_the_fly:
            view.load2device()
        fid1 = view1.fid
        fid2 = view.fid
        xyz = gaussians.get_xyz
        time_input1 = fid1.unsqueeze(0).expand(xyz.shape[0], -1)
        time_input2 = fid2.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input1)
        d_xyz2, d_rotation2, d_scaling2 = deform.step(xyz.detach(), time_input2)
        view_pos = view.camera_center.repeat(gaussians.get_opacity.shape[0], 1)
        wo = safe_normalize(view_pos - gaussians.get_xyz.detach())
        dir_pp_normalized = -wo
        normal = gaussians.get_normal(dir_pp_normalized=dir_pp_normalized)
        deform_wo = safe_normalize(view_pos - (gaussians.get_xyz.detach() + d_xyz.detach()))
        deform_dir_pp_normalized = -deform_wo
        deform_normal = gaussians.get_deformnormal(d_rotation, d_scaling, dir_pp_normalized=deform_dir_pp_normalized)
        rotation_matrix = rotation_matrix_from_vectors(normal, deform_normal)
        deform_deltanormal = torch.matmul(rotation_matrix, gaussians.get_delta_normal.unsqueeze(-1))
        new_normal = safe_normalize(deform_normal + deform_deltanormal.squeeze(-1))
        reflvec = safe_normalize(reflect(deform_wo, new_normal))
        spat = torch.cat([gaussians.get_xyz.detach() + d_xyz.detach(), gaussians.get_roughness * (gaussians.get_scaling.detach() + d_scaling.detach()), gaussians.get_rotation.detach() + d_rotation.detach()], dim=-1)
        spec_color = specdecoder.step(spat, reflvec)
        results = render(view, gaussians, opt, pipeline, background, d_xyz, d_rotation, d_scaling, spec_color, new_normal, iteration, is_6dof)
        # rendering = results["render"]
        # depth = results["depth"]
        # diffuse = results["diffuse"]
        # specular = results["specular_color"]
        # normal = results["normal"]
        #print(rendering.shape)
        # depth = depth / (depth.max() + 1e-5)
        # Gaussian flow
        render_t_1 = render(view, gaussians, opt, pipeline, background, d_xyz, d_rotation, d_scaling, spec_color, new_normal, iteration, is_6dof)
        render_t_2 = render(view, gaussians, opt, pipeline, background, d_xyz2, d_rotation2, d_scaling2, spec_color, new_normal, iteration, is_6dof)
            # Gaussian parameters at t_1
        proj_2D_t_1 = render_t_1["proj_2D"]
        gs_per_pixel = render_t_1["gs_per_pixel"].long() 
        weight_per_gs_pixel = render_t_1["weight_per_gs_pixel"]
        x_mu = render_t_1["x_mu"]
        cov2D_inv_t_1 = render_t_1["conic_2D"].detach()
                # Gaussian parameters at t_2
        proj_2D_t_2 = render_t_2["proj_2D"]
        cov2D_inv_t_2 = render_t_2["conic_2D"]
        cov2D_t_2 = render_t_2["conic_2D_inv"]
                # calculate cov2D_t_2*cov2D_inv_t_1
        cov2D_t_2cov2D_inv_t_1 = torch.zeros([cov2D_inv_t_2.shape[0],2,2]).cuda()
        cov2D_t_2cov2D_inv_t_1[:, 0, 0] = cov2D_t_2[:, 0] * cov2D_inv_t_1[:, 0] + cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 1]
        cov2D_t_2cov2D_inv_t_1[:, 0, 1] = cov2D_t_2[:, 0] * cov2D_inv_t_1[:, 1] + cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 2]
        cov2D_t_2cov2D_inv_t_1[:, 1, 0] = cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 0] + cov2D_t_2[:, 2] * cov2D_inv_t_1[:, 1]
        cov2D_t_2cov2D_inv_t_1[:, 1, 1] = cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 1] + cov2D_t_2[:, 2] * cov2D_inv_t_1[:, 2]

                # full formulation of GaussianFlow
        cov_multi = (cov2D_t_2cov2D_inv_t_1[gs_per_pixel] @ x_mu.permute(0,2,3,1).unsqueeze(-1).detach()).squeeze()

        predicted_flow_by_gs = (cov_multi + proj_2D_t_2[gs_per_pixel] - proj_2D_t_1[gs_per_pixel].detach() - x_mu.permute(0,2,3,1).detach()) * weight_per_gs_pixel.unsqueeze(-1).detach()

        gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(diffuse, os.path.join(diffuse_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(specular, os.path.join(specular_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(normal, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(flow_viz.flow_to_image(predicted_flow_by_gs.sum(0).cpu().numpy())).save(os.path.join(flow_path, '{0:05d}'.format(idx) + ".png"))
        view1 = view


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
