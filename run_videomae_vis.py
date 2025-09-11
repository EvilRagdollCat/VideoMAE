# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from pathlib import Path
from timm.models import create_model
import utils
import modeling_pretrain
from datasets import DataAugmentationForVideoMAE
from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from torchvision import transforms
from transforms import *
from masking_generator import  TubeMaskingGenerator

from utils_dlc_roi import DLCManager, get_bbox_from_dlc
import cv2


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406] # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225] # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupCenterCrop(args.input_size)
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        # > added: for mice dataset we should use random masking in each frame
        elif args.mask_type == 'frame_random':
            def _gen():
                """
                Random mask generator
                """
                tubelet = getattr(args, 'tubelet_size', 2) # Reads tubelet_size from args (default 2). A tubelet spans how many frames in time
                ps = args.patch_size[0] if isinstance(args.patch_size, tuple) else args.patch_size # Gets the spatial patch size (e.g., 16). Handles both tuple (16,16) and int 16
                T_ = args.num_frames // tubelet # Number of temporal tokens (frames divided by tubelet size). E.g., 16 frames, tubelet=2 → T_=8
                H_ = args.input_size // ps # Patch grid height/width per frame. With 224 input and 16-pixel patches → H_=W_=14
                W_ = H_
                hw = H_ * W_ # Number of spatial patches per frame
                m_hw = max(1, int(round(args.mask_ratio * hw))) # How many patches to mask per frame, at least 1
                mask = np.zeros(T_ * hw, dtype=bool) # Allocates a 1D boolean mask for the whole clip (T_ frames × hw patches), initialized to False (unmasked)
                for t in range(T_): # > For each time step
                    idx = np.random.choice(hw, size=m_hw, replace=False) # picks m_hw unique spatial positions within the frame
                    mask[t * hw + idx] = True  # marks those positions as masked in the global 1D mask using an offset of t*hw
                return mask  # Returns a 1D numpy.bool_ mask of length T_ * H_ * W_ (True = masked)
            self.masked_position_generator = _gen

    def __call__(self, images):
        process_data , _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def get_args():
    parser = argparse.ArgumentParser('VideoMAE visualization reconstruction script', add_help=False)
    parser.add_argument('img_path', type=str, help='input video path')
    parser.add_argument('save_path', type=str, help='save video path')
    parser.add_argument('model_path', type=str, help='checkpoint path of model')
    parser.add_argument('--mask_type', default='random', choices=['random', 'tube', 'frame_random'], # > edited "frame_random"
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    # > dlc roi
    parser.add_argument('--use_dlc_roi', action='store_true',
                       help='Use DLC ROI cropping')
    parser.add_argument('--dlc_dir', type=str, default=None,
                       help='Directory containing DLC files')
    parser.add_argument('--dlc_likelihood_threshold', type=float, default=0.6)
    #parser.add_argument('--roi_padding', type=float, default=0.25)
    parser.add_argument('--roi_margin', type=int, default=30,  # 直接用margin，不用padding
                       help='Fixed margin around keypoints in pixels')
    parser.add_argument('--roi_min_size', type=int, default=96)
    
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth
    )

    return model


def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    dlc_manager = None
    dlc_df = None
    if args.use_dlc_roi and args.dlc_dir:
        dlc_manager = DLCManager(args.dlc_dir)
        dlc_df = dlc_manager.load_dlc_data(args.img_path)
        if dlc_df is None:
            print(f"Warning: No DLC data found for {args.img_path}, using full frame")
            args.use_dlc_roi = False

    with open(args.img_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    duration = len(vr)
    #new_length  = 1 
    #new_step = 1
    #skip_length = new_length * new_step
    # frame_id_list = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]

    
    tmp = np.arange(0,32, 2) + 60
    frame_id_list = tmp.tolist()
    # average_duration = (duration - skip_length + 1) // args.num_frames
    # if average_duration > 0:
    #     frame_id_list = np.multiply(list(range(args.num_frames)),
    #                             average_duration)
    #     frame_id_list = frame_id_list + np.random.randint(average_duration,
    #                                             size=args.num_frames)

    video_data = vr.get_batch(frame_id_list).asnumpy()
    print(f"Original video shape: {video_data.shape}") #print(video_data.shape)

    if args.use_dlc_roi and dlc_df is not None:
        cropped_frames = []
        for i, frame_idx in enumerate(frame_id_list):
            frame = video_data[i]
            H, W = frame.shape[:2]

            # > calculate bbox
            y1, y2, x1, x2 = get_bbox_from_dlc(
                dlc_df, frame_idx,
                likelihood_threshold=args.dlc_likelihood_threshold,
                margin=args.roi_margin,  # margin instead of padding
                min_size=args.roi_min_size
            )

            # > boundary
            y1 = max(0, min(y1, H-1))
            y2 = max(y1+1, min(y2, H))
            x1 = max(0, min(x1, W-1))
            x2 = max(x1+1, min(x2, W))

            # > crop and resize
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:  # 防止空裁剪
                print(f"Warning: Empty crop at frame {frame_idx}, using full frame")
                crop = frame

            crop_resized = cv2.resize(crop, (args.input_size, args.input_size),
                                     interpolation=cv2.INTER_LINEAR)
            cropped_frames.append(crop_resized)

        video_data = np.array(cropped_frames)
        print(f"After DLC ROI cropping: {video_data.shape}")



    img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]

    transforms = DataAugmentationForVideoMAE(args)
    img, bool_masked_pos = transforms((img, None)) # T*C,H,W
    # print(img.shape)
    img = img.view((args.num_frames , 3) + img.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
    # img = img.view(( -1 , args.num_frames) + img.size()[-2:]) 
    bool_masked_pos = torch.from_numpy(bool_masked_pos)

    with torch.no_grad():
        # img = img[None, :]
        # bool_masked_pos = bool_masked_pos[None, :]
        img = img.unsqueeze(0)
        print(img.shape)
        bool_masked_pos = bool_masked_pos.unsqueeze(0)
        
        img = img.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        outputs = model(img, bool_masked_pos)

        #save original video
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
        ori_img = img * std + mean  # in [0, 1]
        imgs = [ToPILImage()(ori_img[0,:,vid,:,:].cpu()) for vid, _ in enumerate(frame_id_list)  ]
        for id, im in enumerate(imgs):
            im.save(f"{args.save_path}/ori_img{id}.jpg")

        img_squeeze = rearrange(ori_img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size[0], p2=patch_size[0])
        img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        img_patch[bool_masked_pos] = outputs

        #make mask
        mask = torch.ones_like(img_patch)
        mask[bool_masked_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        mask = rearrange(mask, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)

        #save reconstruction video
        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
        rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
        rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)
        imgs = [ ToPILImage()(rec_img[0, :, vid, :, :].cpu().clamp(0,0.996)) for vid, _ in enumerate(frame_id_list)  ]

        for id, im in enumerate(imgs):
            im.save(f"{args.save_path}/rec_img{id}.jpg")

        #save masked video 
        img_mask = rec_img * mask
        imgs = [ToPILImage()(img_mask[0, :, vid, :, :].cpu()) for vid, _ in enumerate(frame_id_list)]
        for id, im in enumerate(imgs):
            im.save(f"{args.save_path}/mask_img{id}.jpg")

if __name__ == '__main__':
    opts = get_args()
    main(opts)
