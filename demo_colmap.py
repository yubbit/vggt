# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

import argparse
from pathlib import Path
import trimesh
import pycolmap

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track

VGGT_FIXED_RESOLUTION = 518
IMG_LOAD_RESOLUTION = 1024
MAX_POINTS_FOR_COLMAP = 100000

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=3.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    return parser.parse_args()


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = int(max(real_image_size) / img_size)
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction

if __name__ == "__main__":
    with torch.no_grad():
        args = parse_args()

        # Print configuration
        print("Arguments:", vars(args))

        # Set seed for reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
        print(f"Seed set: {args.seed}")

        # Set device and dtype
        if int(torch.cuda.get_device_capability()[0]) >= 80:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        print(f"Using device: {device}")
        print(f"Using dtype: {dtype}")

        # Load images
        image_dir = os.path.join(args.scene_dir, "images")
        image_path_list = glob.glob(os.path.join(image_dir, "*"))
        if len(image_path_list) == 0:
            raise ValueError(f"No images found in {image_dir}")
        base_image_path_list = [os.path.basename(path) for path in image_path_list]

        images, original_coords = load_and_preprocess_images_square(
            image_path_list, 
            IMG_LOAD_RESOLUTION)
        print(f"Loaded {len(images)} images from {image_dir}")

        # Resize images to follow VGGT input format
        images = F.interpolate(
            images, 
            size=(VGGT_FIXED_RESOLUTION, VGGT_FIXED_RESOLUTION), 
            mode="bilinear", 
            align_corners=False)
        images = images[None]  # add batch dimension

        # Load model
        print(f"Loading VGGT model")
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        model.eval()
        model = model.to(device)
        print(f"Model loaded")

        # Run VGGT aggregator, camera head, and depth head
        # TODO: Look into optimizing aggregator performance
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images.to(device)

            print("Running VGGT Aggregator")
            aggregated_tokens_list, ps_idx = model.aggregator(images)
            print("Running VGGT Camera Head")
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            print("Running VGGT Depth Head")
            depth_map, depth_conf = model.depth_head(
                aggregated_tokens_list, 
                images, 
                ps_idx, 
                frames_chunk_size=1)
            print("VGGT outputs calculated")

        # Extrinsic and intrinsic matrices, following OpenCV convention 
        # (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, 
            images.shape[-2:])

        # Remove batch dimensions from VGGT inputs and outputs for further
        # processing
        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()
        images = images.squeeze(0).cpu().numpy()
        original_coords = original_coords.numpy()

        # Free GPU memory
        del model
        del aggregated_tokens_list, ps_idx, pose_enc

    # Create 3D point cloud
    points_3d = unproject_depth_map_to_point_map(
        depth_map, 
        extrinsic, 
        intrinsic)
    points_rgb = (images * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    # (S, H, W, 3), with x, y coordinates and frame indices
    points_xyf = create_pixel_coordinate_grid(
        points_3d.shape[0], 
        points_3d.shape[1], 
        points_3d.shape[2])

    # Filter items by confidence mask, then reduce to 100000 points. Randomly
    # select instead of sorting. In cases where some areas have much stronger
    # observations than others, promotes an even distribution of points
    conf_mask = depth_conf >= args.conf_thres_value
    conf_mask = randomly_limit_trues(conf_mask, MAX_POINTS_FOR_COLMAP)

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    # Convert co COLMAP format
    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        (VGGT_FIXED_RESOLUTION, VGGT_FIXED_RESOLUTION),
        shared_camera=False,
        camera_type="PINHOLE",
    )

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords,
        img_size=VGGT_FIXED_RESOLUTION,
        shift_point2d_to_original_res=True,
        shared_camera=False,
    )

    # Save the reconstruction
    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
