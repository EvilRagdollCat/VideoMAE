import os
import cv2 # > Yiran added
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
from kinetics import VideoClsDataset, VideoMAE
from ssv2 import SSVideoClsDataset
from mice_dataset import MicePretrainDataset, MiceClassificationDataset # > Yiran added
from utils_motion_roi import roi_by_tiles, expand_and_snap # > Yiran added


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
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
        # > Yiran added: for mice dataset we should use random masking in each frame
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
        # images: (imgs, label) 或 list[PIL]
        if isinstance(images, tuple):
            imgs, _ = images
        else:
            imgs = images

        # > ROI before transform
        if getattr(self.args, 'use_motion_roi', False):
            # PIL -> numpy uint8 (T,H,W,3) RGB
            frames = np.stack([np.array(im.convert('RGB')) for im in imgs], axis=0)
            y1,y2,x1,x2 = roi_by_tiles(
                frames, G=self.args.roi_grid, topk=self.args.roi_topk,
                margin=self.args.roi_margin, min_wh=self.args.roi_min_wh
            )
            if getattr(self.args, 'roi_snap16', False):
                y1,x1,y2,x2 = snap16(y1,x1,y2,x2, H=frames.shape[1], W=frames.shape[2],
                                     min_wh=self.args.roi_min_wh, snap=16)
            # optional jitter
            j = getattr(self.args, 'roi_jitter', 0)
            if j > 0:
                dy = random.randint(-j, j); dx = random.randint(-j, j)
                y1 = max(0, min(frames.shape[1]-1, y1+dy))
                y2 = max(1, min(frames.shape[1],   y2+dy))
                x1 = max(0, min(frames.shape[2]-1, x1+dx))
                x2 = max(1, min(frames.shape[2],   x2+dx))
            # > crop and convert back to PIL list
            frames = frames[:, y1:y2, x1:x2, :]
            imgs = [Image.fromarray(frames[t]) for t in range(frames.shape[0])]


        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    # > Yiran added
    if getattr(args, 'dataset_type', 'videomae') == 'mice_pretrain':
        # use our own dataset
        dataset = MicePretrainDataset(
            list_txt=args.data_path,
            num_frames=args.num_frames,
            sampling_rate=args.sampling_rate,
            input_size=args.input_size,
            mask_ratio=args.mask_ratio,
            mask_type=args.mask_type, # 'tube' or 'random'
            tubelet_size=2,
            #use_motion_roi=args.use_motion_roi,
            #roi_grid=args.roi_grid,
            #roi_topk=args.roi_topk,
            #roi_margin=args.roi_margin,
            #roi_min_wh=args.roi_min_wh,
            #roi_prob=args.roi_prob,
            #roi_jitter=args.roi_jitter,
            #roi_snap16=args.roi_snap16,
            use_motion_roi=getattr(args, 'use_motion_roi', False),
            roi_grid=getattr(args, 'roi_grid', 4),
            roi_topk=getattr(args, 'roi_topk', 1),
            roi_margin=getattr(args, 'roi_margin', 0.10),
            roi_min_wh=getattr(args, 'roi_min_wh', 96),
            roi_prob=getattr(args, 'roi_prob', 1.0),
            roi_jitter=getattr(args, 'roi_jitter', 0.0),
            roi_snap16=getattr(args, 'roi_snap16', False),
        )
        print(f"[mice_pretrain] videos={len(dataset)}")
        return dataset


    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    # > Yiran added
    elif args.data_set == 'mice_classification':
        if is_train:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode:
            mode = 'test'
            base = args.eval_data_path or args.data_path
            anno_path = os.path.join(base, 'test.csv')
        else:
            mode = 'validation'
            base = args.eval_data_path or args.data_path
            anno_path = os.path.join(base, 'val.csv')

        # Read path, label
        dataset = MiceClassificationDataset(
            csv_path=anno_path,
            num_frames=args.num_frames,
            sampling_rate=args.sampling_rate,
            input_size=args.input_size,
            mode=mode,
            num_sample=(args.num_sample if mode == "train" else 1),
            use_motion_roi=getattr(args, 'use_motion_roi', False),
            roi_grid=args.roi_grid,
            roi_topk=args.roi_topk,
            roi_margin=args.roi_margin,
            roi_min_wh=args.roi_min_wh,
            roi_snap16=getattr(args, 'roi_snap16', False),
            roi_jitter=args.roi_jitter,
            roi_prob=getattr(args, 'roi_prob', 1.0),
        )
        nb_classes = 2  
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
