# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Union
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate, get_original_cwd
import models
import time
from functools import partial
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.ops import corresponding_cameras_alignment
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.structures import Pointclouds
from pytorch3d.utils import opencv_from_cameras_projection
from pytorch3d.vis.plotly_vis import plot_scene

from util.utils import seed_all_random_engines
from util.match_extraction import extract_match
from util.load_img_folder import load_and_preprocess_images
from util.geometry_guided_sampling import geometry_guided_sampling
from util.metric import compute_ARE

def triangulate_sparse_points(kp1, kp2, i12, pred_cameras, image_size, reproj_thresh=0.5):
    """
    Triangulate 3D points from 2D keypoint matches and predicted cameras.

    kp1, kp2      : (N, 2) pixel coords in the cropped+resized image
    i12           : (N, 2) image-pair indices (0-based)
    reproj_thresh : max reprojection error in pixels to keep a point
    Returns       : (M, 3) numpy array of filtered 3D points, or None
    """
    N_cams = pred_cameras.R.shape[0]
    img_size_t = torch.LongTensor([[image_size, image_size]]).repeat(N_cams, 1).to(pred_cameras.device)
    R_cv, t_cv, K_cv = opencv_from_cameras_projection(pred_cameras, img_size_t)
    R_cv = R_cv.cpu().numpy()   # (N, 3, 3)
    t_cv = t_cv.cpu().numpy()   # (N, 3)
    K_cv = K_cv.cpu().numpy()   # (N, 3, 3)

    # Camera centers in world space: C = -R^T @ t
    cam_centers = np.array([-R_cv[i].T @ t_cv[i] for i in range(N_cams)])
    scene_center = cam_centers.mean(axis=0)
    cam_spread = np.linalg.norm(cam_centers - scene_center, axis=1).mean()
    max_scene_dist = cam_spread * 10  # points beyond this are almost certainly noise

    # Group matches by image pair
    pairs = {}
    for n in range(len(kp1)):
        key = (int(i12[n, 0]), int(i12[n, 1]))
        pairs.setdefault(key, ([], []))
        pairs[key][0].append(kp1[n])
        pairs[key][1].append(kp2[n])

    points_3d = []
    for (idx1, idx2), (pts1, pts2) in pairs.items():
        P1 = K_cv[idx1] @ np.hstack([R_cv[idx1], t_cv[idx1].reshape(3, 1)])
        P2 = K_cv[idx2] @ np.hstack([R_cv[idx2], t_cv[idx2].reshape(3, 1)])

        pts1_arr = np.array(pts1, dtype=np.float64).T  # (2, N)
        pts2_arr = np.array(pts2, dtype=np.float64).T

        X_hom = cv2.triangulatePoints(P1, P2, pts1_arr, pts2_arr)  # (4, N)
        X = (X_hom[:3] / (X_hom[3:] + 1e-9)).T  # (N, 3)

        for i, x in enumerate(X):
            # Must be in front of both cameras
            z1 = (R_cv[idx1] @ x + t_cv[idx1])[2]
            z2 = (R_cv[idx2] @ x + t_cv[idx2])[2]
            if z1 <= 0 or z2 <= 0:
                continue

            # Reprojection error filter
            x1_h = P1 @ np.append(x, 1.0)
            x2_h = P2 @ np.append(x, 1.0)
            x1_r = x1_h[:2] / (x1_h[2] + 1e-9)
            x2_r = x2_h[:2] / (x2_h[2] + 1e-9)
            err1 = np.linalg.norm(x1_r - pts1_arr[:, i])
            err2 = np.linalg.norm(x2_r - pts2_arr[:, i])
            if err1 > reproj_thresh or err2 > reproj_thresh:
                continue

            # Distance from scene center filter
            if np.linalg.norm(x - scene_center) > max_scene_dist:
                continue

            points_3d.append(x)

    if not points_3d:
        return None

    pts = np.array(points_3d, dtype=np.float32)

    # Final MAD-based outlier removal (more robust than std)
    median = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - median, axis=1)
    mad = np.median(dist)
    if mad > 0:
        pts = pts[dist < np.median(dist) + 5 * mad]

    return pts if len(pts) > 0 else None


@hydra.main(config_path="../cfgs/", config_name="default")
def demo(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print("Model Config:")
    print(OmegaConf.to_yaml(cfg))

    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = instantiate(cfg.MODEL, _recursive_=False)

    # Load and preprocess images
    original_cwd = get_original_cwd()  # Get original working directory
    folder_path = os.path.join(original_cwd, cfg.image_folder)
    images, image_info = load_and_preprocess_images(folder_path, cfg.image_size)

    # Load checkpoint
    ckpt_path = os.path.join(original_cwd, cfg.ckpt)
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        raise ValueError(f"No checkpoint found at: {ckpt_path}")

    # Move model and images to the GPU
    model = model.to(device)
    images = images.to(device)

    # Evaluation Mode
    model.eval()

    # Seed random engines
    seed_all_random_engines(cfg.seed)

    # Start the timer
    start_time = time.time()

    # Perform match extraction
    kp1, kp2, i12 = None, None, None
    if cfg.GGS.enable:
        # Optional TODO: remove the keypoints outside the cropped region?

        kp1, kp2, i12 = extract_match(image_folder_path=folder_path, image_info=image_info)

        if kp1 is not None:
            keys = ["kp1", "kp2", "i12", "img_shape"]
            values = [kp1, kp2, i12, images.shape]
            matches_dict = dict(zip(keys, values))

            cfg.GGS.pose_encoding_type = cfg.MODEL.pose_encoding_type
            GGS_cfg = OmegaConf.to_container(cfg.GGS)

            cond_fn = partial(geometry_guided_sampling, matches_dict=matches_dict, GGS_cfg=GGS_cfg)
            print("[92m=====> Sampling with GGS <=====[0m")
        else:
            cond_fn = None
    else:
        cond_fn = None
        print("[92m=====> Sampling without GGS <=====[0m")

    images = images.unsqueeze(0)

    # Forward
    with torch.no_grad():
        # Obtain predicted camera parameters
        # pred_cameras is a PerspectiveCameras object with attributes
        # pred_cameras.R, pred_cameras.T, pred_cameras.focal_length

        # The poses and focal length are defined as
        # NDC coordinate system in
        # https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md
        predictions = model(image=images, cond_fn=cond_fn, cond_start_step=cfg.GGS.start_step, training=False)

    pred_cameras = predictions["pred_cameras"]

    # Stop the timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken: {:.4f} seconds".format(elapsed_time))

    # Compute metrics if gt is available

    # Load gt poses
    if os.path.exists(os.path.join(folder_path, "gt_cameras.npz")):
        gt_cameras_dict = np.load(os.path.join(folder_path, "gt_cameras.npz"))
        gt_cameras = PerspectiveCameras(
            focal_length=gt_cameras_dict["gtFL"], R=gt_cameras_dict["gtR"], T=gt_cameras_dict["gtT"], device=device
        )

        # 7dof alignment, using Umeyama's algorithm
        pred_cameras_aligned = corresponding_cameras_alignment(
            cameras_src=pred_cameras, cameras_tgt=gt_cameras, estimate_scale=True, mode="extrinsics", eps=1e-9
        )

        # Compute the absolute rotation error
        ARE = compute_ARE(pred_cameras_aligned.R, gt_cameras.R).mean()
        print(f"For {folder_path}: the absolute rotation error is {ARE:.6f} degrees.")
    else:
        print(f"No GT provided. No evaluation conducted.")


    # Triangulate sparse point cloud from 2D matches if available
    # Use aligned cameras if GT is available so the point cloud matches the visualization
    sparse_points = None
    if kp1 is not None:
        cams_for_triangulation = pred_cameras_aligned if os.path.exists(os.path.join(folder_path, "gt_cameras.npz")) else pred_cameras
        sparse_points = triangulate_sparse_points(kp1, kp2, i12, cams_for_triangulation, images.shape[-1])
        if sparse_points is not None:
            print(f"Triangulated {len(sparse_points)} 3D points.")

    # Visualization
    if os.path.exists(os.path.join(folder_path, "gt_cameras.npz")):
        cams_show = {"ours_pred": pred_cameras, "ours_pred_aligned": pred_cameras_aligned, "gt_cameras": gt_cameras}
    else:
        cams_show = {"ours_pred": pred_cameras}

    scene_dict = dict(cams_show)
    if sparse_points is not None:
        pts_tensor = torch.from_numpy(sparse_points).float().unsqueeze(0)
        scene_dict["sparse_points"] = Pointclouds(points=pts_tensor.to("cpu"))

    fig = plot_scene({f"{folder_path}": scene_dict})

    html_path = os.path.join(original_cwd, "camera_vis.html")
    fig.write_html(html_path)
    print(f"Saved interactive visualization to: {html_path}")



if __name__ == "__main__":
    demo()
