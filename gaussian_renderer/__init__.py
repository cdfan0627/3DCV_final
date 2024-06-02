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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import numpy as np
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render(viewpoint_camera, pc: GaussianModel, opt, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling,  spec_color, new_normal, iteration, is_6dof=False, 
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = torch.abs(pc.get_scaling + d_scaling)
        rotations = pc.get_rotation + d_rotation


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if iteration < opt.warm_up2:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            #diffuse   = pc.get_diffuse # (N, 3) 
            specular  = pc.get_specular # (N, 3) 
            #diffuse_linear = torch.sigmoid(diffuse - np.log(3.0))
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            diffuse = eval_sh(0, shs_view, dirs = None)
            diffuse_linear = torch.sigmoid(diffuse)
            specular_color = torch.mul(specular, spec_color)
            colors_precomp = diffuse_linear + specular_color
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    render_extras = {}
    out_extras = {}
    
    if iteration >= opt.warm_up2:
        # p_hom = torch.cat([pc.get_xyz, torch.ones_like(pc.get_xyz[...,:1])], -1).unsqueeze(-1)
        # p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0,1), p_hom)
        # p_view = p_view[...,:3,:]
        # depth = p_view.squeeze()[...,2:3]
        # depth = depth.repeat(1,3)

        new_normal_normed = 0.5*new_normal + 0.5
        render_extras.update({
                            "normal": new_normal_normed,
                            # "diffuse": diffuse_linear,
                            # "specular_color": spec_color,
                            # "specular_tint": specular,
                            # "shader depth": depth
                            })
        
        for k in render_extras.keys():
                if render_extras[k] is None: continue
                image = rasterizer(
                    means3D = means3D.detach(),
                    means2D = means2D.detach(),
                    shs = None,
                    colors_precomp = render_extras[k],
                    opacities = opacity.detach(),
                    scales = scales.detach(),
                    rotations = rotations.detach(),
                    cov3D_precomp = cov3D_precomp)[0]
                out_extras[k] = image
        # for k in["normal"]:
        #         if k in out_extras.keys():
        #             out_extras[k] = (out_extras[k] - 0.5) * 2.

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": rendered_depth,
            # "alpha": rendered_alpha,
            # "proj_2D": proj_means_2D,
            # "conic_2D": conic_2D,
            # "conic_2D_inv": conic_2D_inv,
            # "gs_per_pixel": gs_per_pixel,
            # "weight_per_gs_pixel": weight_per_gs_pixel,
            # "x_mu": x_mu
            }
    out.update(out_extras)
    return out

