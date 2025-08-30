"""
Mask visualization
"""

import os
import cv2
import numpy as np
from pathlib import Path
import torch

# > ImageNet mean/std for de-normalizing tensors back to [0,1]
# If training normalization isn’t ImageNet’s mean/std, replace IMNET_MEAN/STD accordingly to avoid color shifts
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

def unnormalize_to_uint8(tensor_cthw, mean=IMNET_MEAN, std=IMNET_STD):
    """
    Convert a normalized (C,T,H,W) float tensor into (T,H,W,3) uint8 frames for saving/visualization
    tensor_cthw: (C, T, H, W), float tensor, normalized by ImageNet std
    return: frames (T, H, W, 3) uint8
    """
    # > Ensure mean/std are indexable; move data to CPU, clone as float to avoid in-place side effects
    if isinstance(mean, torch.Tensor): mean = mean.tolist()
    if isinstance(std, torch.Tensor): std = std.tolist()
    c, t, h, w = tensor_cthw.shape
    x = tensor_cthw.detach().cpu().float().clone()
    # > Per-channel de-normalization: x = x * std + mean
    for i in range(3):
        x[i] = x[i] * std[i] + mean[i]
    # > Clamp to [0,1], scale to uint8, permute to (T,H,W,C), return as numpy
    x = x.clamp(0, 1)
    x = (x * 255.0).byte()              # (C,T,H,W) uint8
    x = x.permute(1, 2, 3, 0).contiguous()  # (T,H,W,C)
    return x.numpy()

def draw_mask_on_frames(frames, mask, num_frames, input_size=224, patch_size=16,
                        tubelet_size=2, color=(0, 0, 255)):
    # Ensure tubelet_size matches your training config; otherwise the time-to-frame mapping will be off
    """
    Decode the 1D boolean mask into per-time-step spatial grids and paint the masked patches (default red) onto representative frames
    frames: (T, H, W, 3) uint8  T frames in pretraining
    mask:   1D bool of length T_*H_*W_（T_ = num_frames//tubelet_size）
    return: the frames after visualization (length = T_, each has the size of (H, W, 3))
    """
    # > T_ is the number of time steps after grouping by tubelets; H_ x W_ is the spatial patch grid; hw is patches per time step
    T  = num_frames
    T_ = T // tubelet_size
    H_ = input_size // patch_size
    W_ = H_
    hw = H_ * W_

    vis_frames = []
    # > Use the middle frame of each tubelet as the representative frame for that time step; copy it for drawing
    for t_ in range(T_):
        # > the middle frame as the representative frame for the time step
        rep_idx = min(t_ * tubelet_size + tubelet_size // 2, T - 1)
        img = frames[rep_idx].copy()

        # > Slice the 2D mask for this time step and overlay colored blocks on masked patches via simple alpha blending (70% color)
        m_t = mask[t_ * hw : (t_ + 1) * hw].reshape(H_, W_)
        for i in range(H_):
            for j in range(W_):
                if m_t[i, j]:
                    y1 = i * patch_size; y2 = y1 + patch_size
                    x1 = j * patch_size; x2 = x1 + patch_size
                    # overlay red blocks on the masked patches
                    img[y1:y2, x1:x2, :] = (img[y1:y2, x1:x2, :] * 0.3 + np.array(color) * 0.7).astype(np.uint8)
        vis_frames.append(img)
    return vis_frames

def save_grid(frames_list, out_path, cols=8):
    # save_grid uses 8 columns by default; adjust cols if the grid is too cramped or too wide
    """
    Tile frames into a single grid image with cols columns for compact visualization
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if len(frames_list) == 0:
        return
    h, w = frames_list[0].shape[:2]
    rows = int(np.ceil(len(frames_list) / cols))
    canvas = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255
    # > Place frames by row/column; save with OpenCV (convert RGB to BGR via [:, :, ::-1])
    for k, im in enumerate(frames_list):
        r = k // cols; c = k % cols
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = im
    cv2.imwrite(str(out_path), canvas[:, :, ::-1])  # BGR

