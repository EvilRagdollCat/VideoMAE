import csv, random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from decord import VideoReader, cpu

# > import dlc tools
from utils_dlc_roi import DLCManager, get_bbox_from_dlc

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _uniform_sample_indices(num_frames, total, sampling_rate):
    """
    Uniformaly sample the indices
    """
    span = (num_frames - 1) * sampling_rate + 1
    if total >= span:
        start = random.randint(0, total - span)
        idx = [start + i * sampling_rate for i in range(num_frames)]
    else:
        base = list(range(0, total, max(1, sampling_rate)))[:num_frames]
        if len(base) == 0:
            base = [0]
        while len(base) < num_frames:
            base.append(base[-1])
        idx = base[:num_frames]
    return idx

def _read_clip_with_dlc(path, num_frames, sampling_rate, input_size,
                        dlc_manager, dlc_likelihood_threshold=0.5,
                        roi_padding=0.2, roi_min_size=96, roi_snap16=False):
    """
    Read the clip with dlc
    """
    vr = VideoReader(str(path), ctx=cpu(0))
    total = len(vr)
    idx = _uniform_sample_indices(num_frames, total, sampling_rate)

    #print(f"Vid {Path(path).stem}: using frame {idx}")
    
    # > load dlc data
    dlc_df = dlc_manager.load_dlc_data(path)
    
    if dlc_df is None:
        # > use the full frame if no dlc roi
        frames = vr.get_batch(idx).asnumpy()
        frames = np.stack([cv2.resize(f, (input_size, input_size)) for f in frames])
    else:
        
        # > read the frame and crop
        frames = vr.get_batch(idx).asnumpy()
        H, W = frames.shape[1], frames.shape[2]
    
        # > crop and resize
        crops = []
        for i, frame_idx in enumerate(idx):  # define frame_idx
            y1, y2, x1, x2 = get_bbox_from_dlc(
                dlc_df, frame_idx,
                likelihood_threshold=dlc_likelihood_threshold,
                margin=15,
                min_size=roi_min_size
            )

            # make sure bbox is within the boundary
            y1 = max(0, min(y1, H-1))
            y2 = max(y1+1, min(y2, H))
            x1 = max(0, min(x1, W-1))
            x2 = max(x1+1, min(x2, W))

            crop = frames[i, y1:y2, x1:x2]
            crop = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_AREA)
            crops.append(crop)
        frames = np.stack(crops)
    
    # > convert to tensor and normalize
    frames = frames.astype(np.float32) / 255.0
    frames = (frames - IMAGENET_MEAN[None, None, None, :]) / IMAGENET_STD[None, None, None, :]
    frames = np.transpose(frames, (3, 0, 1, 2))  # (T,H,W,C) -> (C,T,H,W)
    
    return torch.from_numpy(frames.astype(np.float32))

def _read_clip_full_frame(path, num_frames, sampling_rate, input_size):
    """
    Read the full frame without dlc roi
    """
    vr = VideoReader(str(path), ctx=cpu(0))
    total = len(vr)
    idx = _uniform_sample_indices(num_frames, total, sampling_rate)
    frames = vr.get_batch(idx).asnumpy()
    
    # > resize
    frames = np.stack([cv2.resize(f, (input_size, input_size)) for f in frames])
    
    # > normalize
    frames = frames.astype(np.float32) / 255.0
    frames = (frames - IMAGENET_MEAN[None, None, None, :]) / IMAGENET_STD[None, None, None, :]
    frames = np.transpose(frames, (3, 0, 1, 2))
    
    return torch.from_numpy(frames.astype(np.float32))

def _make_bool_mask(num_frames, input_size, mask_ratio=0.9,
                    mask_type="tube", tubelet_size=2, patch_size=16):
    """
    Generate MAE mask
    """
    T_ = num_frames // tubelet_size
    H_ = input_size // patch_size
    W_ = H_
    
    if mask_type == "tube":
        hw = H_ * W_
        m_hw = max(1, int(round(mask_ratio * hw)))
        mask_hw = np.zeros(hw, dtype=bool)
        idx = np.random.choice(hw, size=m_hw, replace=False)
        mask_hw[idx] = True
        mask = np.tile(mask_hw, T_)
    elif mask_type == "frame_random":
        # > random mask for each frame
        mask = np.zeros(T_ * H_ * W_, dtype=bool)
        for t in range(T_):
            frame_start = t * H_ * W_
            frame_mask_count = max(1, int(round(mask_ratio * H_ * W_)))
            frame_indices = np.random.choice(H_ * W_, size=frame_mask_count, replace=False)
            mask[frame_start + frame_indices] = True
    else:
        total = T_ * H_ * W_
        m = max(1, int(round(mask_ratio * total)))
        mask = np.zeros(total, dtype=bool)
        idx = np.random.choice(total, size=m, replace=False)
        mask[idx] = True
    
    return torch.from_numpy(mask)


class MicePretrainDataset(Dataset):
    """
    Pretrainig dataset
    """
    def __init__(self, list_txt, num_frames=16, sampling_rate=4, input_size=224,
                 mask_ratio=0.9, mask_type="tube", tubelet_size=2,
                 use_dlc_roi=False, dlc_dir=None, 
                 dlc_likelihood_threshold=0.5, roi_padding=0.2,
                 roi_min_size=96, roi_snap16=False, roi_prob=1.0,
                 max_retries=3):
        
        self.paths = [Path(p.strip()) for p in Path(list_txt).read_text().splitlines() if p.strip()]
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.input_size = input_size
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.tubelet_size = tubelet_size
        self.max_retries = max_retries
        
        # > dlc parameters
        self.use_dlc_roi = use_dlc_roi
        self.dlc_manager = DLCManager(dlc_dir) if use_dlc_roi else None
        self.dlc_likelihood_threshold = dlc_likelihood_threshold
        self.roi_padding = roi_padding
        self.roi_min_size = roi_min_size
        self.roi_snap16 = roi_snap16
        self.roi_prob = roi_prob
    
    def __len__(self):
        return len(self.paths)
    
    #def __getitem__(self, idx):
    #    for _ in range(self.max_retries):
    #        try:
    #            p = self.paths[idx]

    #            vr = VideoReader(str(p), ctx=cpu(0))
    #            total = len(vr)
    #            idx = _uniform_sample_indices(self.num_frames, total, self.sampling_rate)

    #            if random.random() < 0.01:  
    #                print(f"Pretrain - {Path(p).stem}: frames {idx[0]}-{idx[-1]} (total: {total})")
                
    #            # > whether to use dlc roi
    #            use_roi = self.use_dlc_roi and (random.random() < self.roi_prob)
                
    #            if use_roi and self.dlc_manager:
    #                vid = _read_clip_with_dlc(
    #                    p, self.num_frames, self.sampling_rate, self.input_size,
    #                    self.dlc_manager, self.dlc_likelihood_threshold,
    #                    self.roi_padding, self.roi_min_size, self.roi_snap16
    #                )
    #            else:
    #                vid = _read_clip_full_frame(
    #                    p, self.num_frames, self.sampling_rate, self.input_size
    #                )
                
    #            mask = _make_bool_mask(
    #                self.num_frames, self.input_size, self.mask_ratio,
    #                self.mask_type, self.tubelet_size
    #            )
                
    #            return vid, mask
                
    #        except Exception as e:
    #            print(f"Error loading {self.paths[idx]}: {e}")
    #            idx = (idx + 1) % len(self.paths)
    #    
    #    raise RuntimeError(f"Failed to load any video after {self.max_retries} attempts")

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            try:
                p = self.paths[idx]

                # > only visualize the first 10 samples
                #visualize = (idx < 10)

                vr = VideoReader(str(p), ctx=cpu(0))
                total = len(vr)
                frame_indices = _uniform_sample_indices(self.num_frames, total, self.sampling_rate)

                # > get original frame to compare
                #if visualize:
                #    raw_frames = vr.get_batch(frame_indices).asnumpy()

                use_roi = self.use_dlc_roi and (random.random() < self.roi_prob)

                #if visualize and use_roi and self.dlc_manager:
                #    dlc_df = self.dlc_manager.load_dlc_data(p)
                #    if dlc_df is not None:
                #    
                #        row = dlc_df.iloc[frame_indices[0] if frame_indices[0] < len(dlc_df) else -1]

                    
                #        debug_frame = raw_frames[0].copy()

                    
                #        for i in range(0, len(row), 3):
                #            if i+2 < len(row):
                #                x, y, likelihood = row.iloc[i], row.iloc[i+1], row.iloc[i+2]
                #                if not np.isnan(x):
                #                    color = (0, 255, 0) if likelihood > self.dlc_likelihood_threshold else (0, 0, 255)
                #                    cv2.circle(debug_frame, (int(x), int(y)), 5, color, -1)
                #                    cv2.putText(debug_frame, f"{i//3}", (int(x)+8, int(y)-8),
                #                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    
                #        y1, y2, x1, x2 = get_bbox_from_dlc(
                #            dlc_df, frame_indices[0],
                #            self.dlc_likelihood_threshold,
                #            self.roi_padding,
                #            self.roi_min_size
                #        )
                #        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    
                #        dlc_debug_path = f"/data/videomae_outputs/pretrain_crop_check/dlc_debug_{idx:03d}.jpg"
                #        cv2.imwrite(dlc_debug_path, cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
                #        print(f"[DLC DEBUG] Saved keypoints visualization to {dlc_debug_path}")

                if use_roi and self.dlc_manager:
                    vid = _read_clip_with_dlc(
                        p, self.num_frames, self.sampling_rate, self.input_size,
                        self.dlc_manager, self.dlc_likelihood_threshold,
                        self.roi_padding, self.roi_min_size, self.roi_snap16
                    )
                    #crop_status = "DLC_ROI"
                else:
                    vid = _read_clip_full_frame(
                        p, self.num_frames, self.sampling_rate, self.input_size
                    )
                    #crop_status = "FULL_FRAME"

            
                #if visualize:
                #    import matplotlib
                #    matplotlib.use('Agg')
                #    import matplotlib.pyplot as plt
                #    from pathlib import Path

                #    save_dir = Path("/data/videomae_outputs/pretrain_crop_check")
                #    save_dir.mkdir(exist_ok=True)

                #    vid_np = vid.numpy()
                #    vid_display = vid_np.transpose(1,2,3,0)
                #    vid_display = vid_display * IMAGENET_STD + IMAGENET_MEAN
                #    vid_display = np.clip(vid_display * 255, 0, 255).astype(np.uint8)

                    # > save the first frame to compare
                #    original = cv2.resize(raw_frames[0], (224, 224))
                #    processed = vid_display[0]

                    # > concatenate
                #    comparison = np.hstack([original, processed])

                
                #    comparison = cv2.putText(comparison, "Original", (10, 30),
                #                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                #    comparison = cv2.putText(comparison, crop_status, (234, 30),
                #                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                
                #    save_path = save_dir / f"sample_{idx:03d}_{Path(p).stem}_{crop_status}.jpg"
                #    cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

                
                #    print(f"\n[CROP CHECK {idx}] {Path(p).stem}: {crop_status}")
                #    if use_roi and self.dlc_manager:
                #        dlc_df = self.dlc_manager.load_dlc_data(p)
                #        print(f"  DLC found: {dlc_df is not None}")
                #    print(f"  Saved: {save_path}")

                mask = _make_bool_mask(
                    self.num_frames, self.input_size, self.mask_ratio,
                    self.mask_type, self.tubelet_size
                )

                return vid, mask

            except Exception as e:
                print(f"Error loading {self.paths[idx]}: {e}")
                import traceback
                traceback.print_exc()
                idx = (idx + 1) % len(self.paths)

        raise RuntimeError(f"Failed to load any video after {self.max_retries} attempts")


class MiceClassificationDataset(Dataset):
    """
    Classification dataset
    """
    def __init__(self, csv_path, num_frames=16, sampling_rate=4, input_size=224,
                 mode="train", num_sample=1,
                 use_dlc_roi=False, dlc_dir=None,
                 dlc_likelihood_threshold=0.5, roi_padding=0.25,
                 roi_min_size=96, roi_snap16=True, roi_prob=1.0):
        
        # > load csv
        self.rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                p = Path(r["path"]).expanduser()
                y = int(r["label"])
                self.rows.append((p, y))
        
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.input_size = input_size
        self.mode = mode
        self.num_sample = num_sample
        
        # > dlc parameters
        self.use_dlc_roi = use_dlc_roi
        self.dlc_manager = DLCManager(dlc_dir) if use_dlc_roi else None
        self.dlc_likelihood_threshold = dlc_likelihood_threshold
        self.roi_padding = roi_padding
        self.roi_min_size = roi_min_size
        self.roi_snap16 = roi_snap16
        self.roi_prob = roi_prob if mode == "train" else 1.0  # always use dlc roi if testing
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        p, label = self.rows[idx]
        if self.mode == 'train' and idx % 100 == 0:  # documnt one in every 100 samples
            with open('frame_log.txt', 'a') as f:
                f.write(f"{p},{label}\n")
        
        try:
            # > whether to use dlc roi
            use_roi = self.use_dlc_roi and (random.random() < self.roi_prob)
            
            if use_roi and self.dlc_manager:
                vid = _read_clip_with_dlc(
                    p, self.num_frames, self.sampling_rate, self.input_size,
                    self.dlc_manager, self.dlc_likelihood_threshold,
                    self.roi_padding, self.roi_min_size, self.roi_snap16
                )
            else:
                vid = _read_clip_full_frame(
                    p, self.num_frames, self.sampling_rate, self.input_size
                )
            
            if self.mode == "test":
                return (vid, label, 
                       torch.tensor(idx, dtype=torch.int64),
                       torch.tensor(0, dtype=torch.int64),
                       torch.tensor(0, dtype=torch.int64))
            else:
                return (vid, label,
                       torch.tensor(idx, dtype=torch.int64),
                       torch.tensor(0, dtype=torch.int64))
                       
        except Exception as e:
            print(f"Error loading {p}: {e}")
            # > return an empty tensor
            empty_vid = torch.zeros((3, self.num_frames, self.input_size, self.input_size))
            if self.mode == "test":
                return (empty_vid, label,
                       torch.tensor(idx, dtype=torch.int64),
                       torch.tensor(0, dtype=torch.int64),
                       torch.tensor(0, dtype=torch.int64))
            else:
                return (empty_vid, label,
                       torch.tensor(idx, dtype=torch.int64),
                       torch.tensor(0, dtype=torch.int64))
