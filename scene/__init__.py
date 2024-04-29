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
import random
import torch
import json
import numpy as np
from PIL import Image
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel, SpecModel
from arguments import ModelParams
from scene.cameras import Camera
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cams, args):
        self.cams = cams
        self.args = args

    def __getitem__(self, index):
        cam_info = self.cams[index]
        # image = cam_info.image
        image = Image.open(cam_info.image_path)
        resized_image = torch.from_numpy(np.array(image)) / 255.0

        if len(resized_image.shape) == 3:
            resized_image = resized_image.permute(2, 0, 1)
        else:
            resized_image = resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
        
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                      FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                      image=resized_image, gt_alpha_mask=None,
                      image_name=cam_info.image_name, uid=cam_info.uid, data_device=self.args.data_device, fid=cam_info.fid)        

    def __len__(self):
        return len(self.cams)


class FlowDataset(torch.utils.data.Dataset):
    def __init__(self, cams, args):
        self.cams = cams
        self.args = args

    def __getitem__(self, index):
        cam_info = self.cams[index]
        # image = cam_info.image
        image = Image.open(cam_info.image_path)
        data_root = '/'.join(cam_info.image_path.split('/')[:-2])
        folder = cam_info.image_path.split('/')[-2]
        image_name =  cam_info.image_path.split('/')[-1]
        fwd_flow_path = os.path.join(data_root, f'{folder}_flow', f'{os.path.splitext(image_name)[0]}_fwd.npz')
        bwd_flow_path = os.path.join(data_root, f'{folder}_flow', f'{os.path.splitext(image_name)[0]}_bwd.npz')
        normal_path = os.path.join(data_root, f'normals_from_pretrain', f'{os.path.splitext(image_name)[0]}.png')
        normal_map = np.array(Image.open(normal_path), dtype="uint8")[..., :3]
        normal_map = torch.from_numpy(normal_map.astype("float32") / 255.0).float()
        depth_path = os.path.join(data_root, f'1x_depth', f'{os.path.splitext(image_name)[0]}_depth.png')
        depth_map = np.array(Image.open(depth_path), dtype="uint8")[..., :1]
        depth_map = torch.from_numpy(depth_map.astype("float32")/255).float()
        if os.path.exists(fwd_flow_path):
            fwd_data = np.load(fwd_flow_path)
            fwd_flow = torch.from_numpy(fwd_data['flow'])
            fwd_flow_mask = torch.from_numpy(fwd_data['mask'])
        else:
            fwd_flow, fwd_flow_mask  = None, None
        if os.path.exists(bwd_flow_path):
            bwd_data = np.load(bwd_flow_path)
            bwd_flow = torch.from_numpy(bwd_data['flow'])
            bwd_flow_mask = torch.from_numpy(bwd_data['mask'])
        else:
            bwd_flow, bwd_flow_mask  = None, None
        
        # image = np.zeros((3, 128, 128))
        resized_image = torch.from_numpy(np.array(image)) / 255.0

        if len(resized_image.shape) == 3:
            resized_image = resized_image.permute(2, 0, 1)
        else:
            resized_image = resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
        
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                      FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                      image=resized_image, gt_alpha_mask=None,
                      image_name=cam_info.image_name, uid=cam_info.uid,
                      data_device=self.args.data_device, fid=cam_info.fid,
                      fwd_flow=fwd_flow, fwd_flow_mask=fwd_flow_mask,
                      bwd_flow=bwd_flow, bwd_flow_mask=bwd_flow_mask, depth_map= depth_map, normal_map=normal_map)
    def __len__(self):
        return len(self.cams)


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.use_loader = False

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval)
            self.use_loader = True
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found calibration_full.json, assuming Neu3D data set!")
            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, 24)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if not self.use_loader:
            if shuffle:
                random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
                random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        if self.use_loader:
            self.train_cameras[resolution_scales[0]] = FlowDataset(scene_info.train_cameras, args)
            self.test_cameras[resolution_scales[0]] = FlowDataset(scene_info.test_cameras, args)

        else:
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
