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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence, compute_depth_loss, mean_angular_error
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel, SpecModel
from utils.general_utils import safe_state, get_linear_noise_func, safe_normalize, reflect,  rotation_matrix_from_vectors
from utils.rigid_utils import from_homogenous, to_homogenous
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import copy

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    torch.autograd.set_detect_anomaly(True)

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    specdecoder = SpecModel()
    deform.train_setting(opt)
    specdecoder.train_setting(opt)
    
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    white_color = [1, 1, 1]
    white_background = torch.tensor(white_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_flow_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    if scene.use_loader:
        cam_loader = torch.utils.data.DataLoader(scene.getTrainCameras(), batch_size=1, shuffle=True, collate_fn=list)
        loader = iter(cam_loader)
        viewpoint_cam = next(loader)[0]
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if scene.use_loader:
            try:
                viewpoint_cam2 = next(loader)[0]
            except:
                loader = iter(cam_loader)
                viewpoint_cam = next(loader)[0]
                viewpoint_cam2 = next(loader)[0]
        # Pick a random Camera
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        total_frame = len(scene.getTrainCameras())
        time_interval = 1 / total_frame
        
        if iteration >= opt.warm_up:
            decay_iteration = total_frame
            Temp = 1. / (10 ** ((iteration - 3000) // (decay_iteration * 1000)))
        else:
            Temp = 0
        
        if dataset.load2gpu_on_the_fly:
            if viewpoint_cam2 is not None:
                viewpoint_cam2.load2device()
            viewpoint_cam.load2device()

        if viewpoint_cam2 is not None:
            fid2 = viewpoint_cam2.fid
        fid = viewpoint_cam.fid
        

        if iteration < opt.warm_up:
            gaussians.set_requires_grad("features_dc", state=True)
            gaussians.set_requires_grad("features_rest", state=True)
            gaussians.set_requires_grad("diffuse_color", state=False)
            gaussians.set_requires_grad("roughness", state=False)
            gaussians.set_requires_grad("specular", state=False)
            gaussians.set_requires_grad("normal", state=False)
            d_xyz, d_rotation, d_scaling, spec_color, new_normal = 0.0, 0.0, 0.0, 0.0, 0.0
        elif iteration < opt.warm_up2:
            gaussians.set_requires_grad("features_dc", state=True)
            gaussians.set_requires_grad("features_rest", state=True)
            gaussians.set_requires_grad("diffuse_color", state=False)
            gaussians.set_requires_grad("roughness", state=False)
            gaussians.set_requires_grad("specular", state=False)
            gaussians.set_requires_grad("normal", state=False)
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)
            if viewpoint_cam2 is not None:
                time_input2 = fid2.unsqueeze(0).expand(N, -1)
                d_xyz2, d_rotation2, d_scaling2 = deform.step(gaussians.get_xyz.detach(), time_input2)
            
            spec_color, new_normal = 0.0, 0.0
            
        else:
            gaussians.set_requires_grad("features_dc", state=True)
            gaussians.set_requires_grad("features_rest", state=False)
            gaussians.set_requires_grad("diffuse_color", state=False)
            gaussians.set_requires_grad("roughness", state=True)
            gaussians.set_requires_grad("specular", state=True)
            gaussians.set_requires_grad("normal", state=True)
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            d_xyz, d_rotation, d_scaling  = deform.step(gaussians.get_xyz.detach(), time_input)
            if viewpoint_cam2 is not None:
                time_input2 = fid2.unsqueeze(0).expand(N, -1)
                d_xyz2, d_rotation2, d_scaling2 = deform.step(gaussians.get_xyz.detach(), time_input2)
            view_pos = viewpoint_cam.camera_center.repeat(gaussians.get_opacity.shape[0], 1)
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
            
            

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, opt, pipe, background, d_xyz, d_rotation, d_scaling, spec_color, new_normal, iteration, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        #depth = render_pkg_re["depth"]
        Lflow = 0
        #depth_loss = 0
        
        if iteration >= opt.warm_up and viewpoint_cam.kwargs['fwd_flow'] is not None:
            # Gaussian flow
            render_t_1 = render(viewpoint_cam, gaussians, opt, pipe, background, d_xyz, d_rotation, d_scaling, spec_color, new_normal, iteration, dataset.is_6dof)
            render_t_2 = render(viewpoint_cam, gaussians, opt, pipe, background, d_xyz2, d_rotation2, d_scaling2, spec_color, new_normal, iteration, dataset.is_6dof)
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
            optical_flow = viewpoint_cam.kwargs['fwd_flow'].cuda()
            #large_motion_msk = torch.norm(optical_flow, p=2, dim=-1) >= 0.1
            Lflow = torch.norm((optical_flow - predicted_flow_by_gs.sum(0)), p=2, dim=-1).mean()
            #depth_loss = compute_depth_loss(depth, viewpoint_cam.kwargs['depth_map'].permute(2, 0, 1).to('cuda')) 
        
        # Loss
        if iteration < opt.warm_up2:
            normal_loss, reg_loss = 0.0, 0.0
        else:
            reg_loss = torch.sum(torch.norm(gaussians.get_delta_normal, p=2, dim=1))
            pred_normal = render_pkg_re["normal"].to('cuda')
            #pred_normal = pred_normal / torch.norm(pred_normal, p=2, dim=0, keepdim=True)
            gt_normal = viewpoint_cam.kwargs['normal_map'].permute(2, 0, 1).to('cuda')
            #gt_normal = gt_normal / torch.norm(gt_normal, p=2, dim=0, keepdim=True)
            normal_loss = torch.abs((gt_normal - pred_normal)).mean()
            #normal_loss = mean_angular_error(pred=(pred_normal.permute(2, 0, 1) - 1) / 2, gt=(gt_normal.permute(2, 0, 1) - 1) / 2).mean()
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))+ (opt.lambda_normal * normal_loss) +(opt.lambda_reg * reg_loss)
        loss += opt.lambda_flow * Lflow
        loss.backward()

        viewpoint_cam = viewpoint_cam2
        iter_end.record()

        '''
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')
        '''
        with torch.no_grad():
            # Progress bar
            # ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # fl = Lflow.item() if hasattr(Lflow, 'item') else Lflow
            # ema_flow_loss_for_log = 0.4 * fl + 0.6 * ema_flow_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{Ll1:.{4}f}", "Flow loss": f"{Lflow:.{4}f}", "normal loss": f"{normal_loss:.{4}f}", "reg loss": f"{reg_loss:.{4}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])
  
                
            # Log and save
            cur_psnr = training_report(tb_writer, iteration, opt, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform, specdecoder,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration
                    scene.save(iteration)
                    deform.save_weights(args.model_path, iteration)
                    specdecoder.save_weights(args.model_path, iteration)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                specdecoder.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                specdecoder.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                specdecoder.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, opt, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, specdecoder, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:

            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    
                    if iteration < opt.warm_up:
                        d_xyz, d_rotation, d_scaling, spec_color, new_normal = 0.0, 0.0, 0.0, 0.0, 0.0
                    elif iteration < opt.warm_up2:
                        xyz = scene.gaussians.get_xyz
                        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                        spec_color, new_normal = 0.0, 0.0
                    else:
                        xyz = scene.gaussians.get_xyz
                        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                        view_pos = viewpoint.camera_center.repeat(scene.gaussians.get_opacity.shape[0], 1)
                        wo = safe_normalize(view_pos - scene.gaussians.get_xyz.detach())
                        dir_pp_normalized = -wo
                        normal = scene.gaussians.get_normal(dir_pp_normalized=dir_pp_normalized)
                        deform_wo = safe_normalize(view_pos - (scene.gaussians.get_xyz.detach() + d_xyz.detach()))
                        deform_dir_pp_normalized = -deform_wo
                        deform_normal = scene.gaussians.get_deformnormal(d_rotation, d_scaling, dir_pp_normalized=deform_dir_pp_normalized)
                        rotation_matrix = rotation_matrix_from_vectors(normal, deform_normal)
                        deform_deltanormal = torch.matmul(rotation_matrix, scene.gaussians.get_delta_normal.unsqueeze(-1))
                        new_normal = safe_normalize(deform_normal + deform_deltanormal.squeeze(-1))
                        reflvec = safe_normalize(reflect(deform_wo, new_normal))
                        spat = torch.cat([scene.gaussians.get_xyz.detach() + d_xyz.detach(), scene.gaussians.get_roughness * (scene.gaussians.get_scaling.detach() + d_scaling.detach()), scene.gaussians.get_rotation.detach() + d_rotation.detach()], dim=-1)
                        spec_color = specdecoder.step(spat, reflvec)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, opt, *renderArgs, d_xyz, d_rotation, d_scaling,  spec_color, new_normal, iteration, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 100):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration >= opt.warm_up2:
                            normal_image = renderFunc(viewpoint, scene.gaussians, opt, *renderArgs, d_xyz, d_rotation, d_scaling, spec_color, new_normal, iteration, is_6dof)["normal"]
                            tb_writer.add_images(config['name'] + "_view_{}/normal".format(viewpoint.image_name),
                                                normal_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr

if __name__ == "__main__":
    # Set up command line argument parser
    torch.manual_seed(6000)
    np.random.seed(6000)
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=list(range(12000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(range(18000, 40001, 1000)))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
