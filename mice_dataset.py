import csv, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from utils_motion_roi import roi_by_tiles
import cv2

try:
    import decord
    from decord import VideoReader, cpu
except ImportError:
    raise ImportError("Please `pip install decord`")

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1,1)
# [0.485, 0.456, 0.406] [0.229, 0.224, 0.225]: ImageNet default statistic values
# view(): Returns a new tensor with the same data as the self tensor but of a different shape. mean and std went from (3,) to (3, 1, 1, 1)
# reshape the original default mean and std so they can do broadcasting
IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_NP  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _to_uint8_cthw_for_roi(vid_normed, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    # vid_normed: (C,T,H,W), 归一化后
    C,T,H,W = vid_normed.shape
    x = vid_normed.detach().cpu().float().clone()
    for i in range(3):
        x[i] = x[i] * std[i] + mean[i]
    x = x.clamp(0,1)
    x = (x*255.0).byte()                  # C,T,H,W uint8
    return x

def _read_clip_motion_cropped(path, num_frames, sampling_rate, input_size,
                              G=4, topk=1, margin=0.10, min_wh=96, snap16=False, jitter=0):
    vr = VideoReader(str(path), ctx=cpu(0))
    dur = len(vr)
    
    idx = np.linspace(0, dur-1, num_frames*sampling_rate, dtype=int)[::sampling_rate]
    #idx = _uniform_sample_indices(num_frames, total, sampling_rate)  # T indices
    frames = vr.get_batch(idx).asnumpy()              # (T,H,W,3) uint8, RGB

    #H, W = frames.shape[1], frames.shape[2]
    T, H, W, _ = frames.shape

    # > use the motion energy of the whole clip to decide a stable ROI
    y1,y2,x1,x2 = roi_by_tiles(frames, G=G, topk=topk, margin=margin, min_wh=min_wh)
    if jitter > 0:
        dy = np.random.randint(-jitter, jitter + 1)
        dx = np.random.randint(-jitter, jitter + 1)
        y1 = max(0, min(H - 1, y1 + dy)); y2 = max(1, min(H, y2 + dy))
        x1 = max(0, min(W - 1, x1 + dx)); x2 = max(1, min(W, x2 + dx))

    if snap16:
        #y1, x1, y2, x2 = expand_and_snap(y1, x1, y2, x2, H, W, min_wh=min_wh, snap=16)
        side = max(min_wh, max(y2 - y1, x2 - x1))
        side = int(np.ceil(side / 16.0) * 16)
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        y1 = max(0, min(H - side, cy - side // 2)); y2 = y1 + side
        x1 = max(0, min(W - side, cx - side // 2)); x2 = x1 + side

    # > crop and resize to input_size
    #crops = [cv2.resize(f[y1:y2, x1:x2], (input_size, input_size), interpolation=cv2.INTER_AREA) for f in frames]
    #arr = (np.stack(crops).astype(np.float32) / 255.0)  # (T,H,W,3)
    ## > Normalization -> (C,T,H,W)
    ##arr = (arr - IMNET_MEAN) / IMNET_STD
    ##arr = (arr - IMAGENET_MEAN_NP[None, None, None, :]) / IMAGENET_STD_NP[None, None, None, :]
    #arr = (arr - IMAGENET_MEAN.numpy()) / IMAGENET_STD.numpy()
    #arr = np.transpose(arr, (3,0,1,2))
    #return torch.from_numpy(arr)  # float32, (C,T,H,W)
    # > crop and resize to input_size
    crops = [cv2.resize(f[y1:y2, x1:x2], (input_size, input_size), interpolation=cv2.INTER_AREA) for f in frames]
    arr = (np.stack(crops).astype(np.float32) / 255.0)  # (T,H,W,3) in [0,1]

    # > numpy version of ImageNet mean std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean[None, None, None, :]) / std[None, None, None, :]

    # -> (C,T,H,W)
    arr = np.transpose(arr, (3, 0, 1, 2)).astype(np.float32)
    return torch.from_numpy(arr)

def _uniform_sample_indices(num_frames, total, sampling_rate):
    """Uniformly sample num_frames frames from [0, total-1] with stride sampling_rate; if there aren't enough frames, repeat the last frame."""
    # num_frames: the number of frames after sampling. normally 8-16
    # total: the number of frames before sampling
    span = (num_frames - 1) * sampling_rate + 1
    # the span covered by the downsampling
    if total >= span:
        start = random.randint(0, total - span) # Random pick a valid start point
        idx = [start + i * sampling_rate for i in range(num_frames)] 
        # Generate the array of sampling indices start, start + sampling_rate, start + 2 * sampling_rate, ..., start + num_frames * sampling
    else:
        # Get as much frames as possible then pad
        base = list(range(0, total, max(1, sampling_rate)))[:num_frames] 
        # len(base): we can get len(base) indices from 0 to total with a sampling rate of sampling_rate
        # base: the list of frame indices you can obtain when the video is too short (i.e. in the else branch), starting from 0 with a fixed stride. 
        # It serves as the initial index (seed list): first enumerate all indices you can get; if there are fewer than num_frames, 
        # pad to the target length by repeating the last index; finally, assign the first num_frames of the padded list to idx.
        if len(base) == 0:
            base = [0]
        while len(base) < num_frames:
            base.append(base[-1]) # repetitively append the last index to the list
        idx = base[:num_frames]
    return idx

def _read_clip_as_tensor(path, num_frames, sampling_rate, input_size):
    vr = VideoReader(str(path), ctx=cpu(0)) # read the vid
    total = len(vr) # total number of frames in vr
    inds = _uniform_sample_indices(num_frames, total, sampling_rate) # get the indices of the downsampled frames
    frames = vr.get_batch(inds).asnumpy()  # Read all the corresponding frames of inds. (T,H,W,C), uint8

    # resize to squares input_size
    # convert numpy to torch then interpolate
    vid = torch.from_numpy(frames).permute(3,0,1,2).float() / 255.0  # Permute: convert from (T, H, W, C) to (C, T, H, W) 
    C,T,H,W = vid.shape 
    vid = vid.unsqueeze(0)  # add a batch dimension (1,C,T,H,W). The main purpose is to not edit T
    # Do scaling only in the spacial dimensions
    vid = torch.nn.functional.interpolate(
        vid.flatten(0,1), size=(input_size, input_size), mode="bilinear", align_corners=False
    ).view(1, C, T, input_size, input_size).squeeze(0)
    # vid.flatten(0, 1): merge 1 and C into one dimension
    # nn.functional.interpolate() wants the input as "N, C, H, W". We put our T on the "C" position to make it remain unchanged
    # view(): change the shape
    # squeeze(): convert (1, C, T, H', W') back to (C, T, H', W')

    # normalization
    vid = (vid - IMAGENET_MEAN) / IMAGENET_STD
    # for each color channel: substract a fixed mean and be divided by a fixed std
    return vid  # (C, T, H', W')


PATCH_SIZE = 16 # ViT-B/16
DEFAULT_TUBELET = 2 # VideoMAE default

def _make_bool_mask(num_frames, input_size, mask_ratio=0.9,
                    mask_type="tube", tubelet_size=DEFAULT_TUBELET, patch_size=PATCH_SIZE):
    # tubelet_size: The number of frames contained in each temporal “tubelet” (VideoMAE Base commonly uses 2). It down-samples the time axis
    # patch_size: The spatial patch size of the ViT (e.g., ViT-B/16 means patch size 16). It down-samples the spatial dimensions

    # Calculate the size of the token grid (all the parameters are int)
    T_ = num_frames // tubelet_size 
    H_ = input_size // patch_size
    W_ = H_
    if mask_type == "tube":
        # First select the masked positions on the H×W grid, then reuse (replicate) them across all T_ time steps.
        hw = H_ * W_ # total token number. patch number of each time step
        m_hw = max(1, int(round(mask_ratio * hw))) # how many spatial places should be masked
        mask_hw = np.zeros(hw, dtype=bool) 
        idx = np.random.choice(hw, size=m_hw, replace=False)  # np.random.choice(..., replace=False): sample without replacement, selecting m_hw unique positions (no duplicates).
        mask_hw[idx] = True
        mask = np.tile(mask_hw, T_) 
        # np.tile(mask_hw, T_): replicate the single-frame spatial mask T_ times to form a 1D mask of length T_ × hw, i.e. reuse the same spatial positions across all time steps (forming a "tube")
    else:  # randomly chose m positions to be true in T_*H_*W_ 
        total = T_ * H_ * W_
        m = max(1, int(round(mask_ratio * total)))
        mask = np.zeros(total, dtype=bool)
        idx = np.random.choice(total, size=m, replace=False)
        mask[idx] = True
    return torch.from_numpy(mask)  # torch.bool


class MicePretrainDataset(Dataset):
    """Read pretrain_list.txt and video_tensor (C,T,H,W)"""
    def __init__(self, list_txt, num_frames=16, sampling_rate=4, input_size=224, 
            mask_ratio=0.9, mask_type="tube", tubelet_size=DEFAULT_TUBELET,
            max_retries=3,
            use_motion_roi=False, roi_grid=4, roi_topk=1, roi_margin=0.10,
            roi_min_wh=96, roi_prob=1.0, roi_jitter=0, roi_snap16=False):
        # num_frames: how many frames to sample
        self.paths = [Path(p.strip()) for p in Path(list_txt).read_text().splitlines() if p.strip()]
        # if p.strip(): skip the empty lines
        # .splitlines(): split by rows
        # p.strip(): removes leading and trailing whitespace, then Path(...) converts it into a Path object.
        # self.paths: a list. each element is a string of the path
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.input_size = input_size
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.tubelet_size = tubelet_size
        self.max_retries = max_retries

        self.use_motion_roi = use_motion_roi
        self.roi_grid = roi_grid
        self.roi_topk = roi_topk
        self.roi_margin = roi_margin
        self.roi_min_wh = roi_min_wh
        self.roi_prob = roi_prob     # prob = 0.7 means 70% of the clips are ROI, 30% of the clips are full images
        self.roi_jitter = roi_jitter
        self.roi_snap16 = roi_snap16

    def __len__(self): return len(self.paths) # return the amount of the vids

    def __getitem__(self, idx): # get the idxth vid
        #p = self.paths[idx]
        #vid = _read_clip_as_tensor(p, self.num_frames, self.sampling_rate, self.input_size)

        # > This version is too avoid the MOOV atom issue
        tries = self.max_retries
        last_err = None
        for _ in range(tries):
            p = self.paths[idx]
            try:
                use_roi = self.use_motion_roi and (random.random() < self.roi_prob)
                if use_roi:
                    vid = _read_clip_motion_cropped(
                        p, self.num_frames, self.sampling_rate, self.input_size,
                        G=self.roi_grid, topk=self.roi_topk, margin=self.roi_margin, min_wh=self.roi_min_wh, snap16=self.roi_snap16, jitter=self.roi_jitter
                    )
                else:
                    vid = _read_clip_as_tensor(p, self.num_frames, self.sampling_rate, self.input_size)

                mask = _make_bool_mask(self.num_frames, self.input_size,
                                    mask_ratio=self.mask_ratio,
                                    mask_type=self.mask_type,
                                    tubelet_size=self.tubelet_size,
                                    patch_size=16).to(torch.bool)
                return vid, mask


                #vid = _read_clip_as_tensor(p, self.num_frames, self.sampling_rate, self.input_size) # this means return vid
                #vid = _read_clip_motion_cropped(p, self.num_frames, self.sampling_rate, self.input_size, G=4, topk=3)
                #mask = _make_bool_mask(self.num_frames, self.input_size,
                #                       mask_ratio=self.mask_ratio,
                #                       mask_type=self.mask_type,
                #                       tubelet_size=self.tubelet_size,
                #                       patch_size=PATCH_SIZE).to(torch.bool)
                #return vid, mask
            except Exception as e:
                print(f"Skip: {p} | {e}")
                idx = (idx + 1) % len(self.paths) # skip to the next
                last_err = e
        # Raise error after multiple failures
        raise RuntimeError(f"No readable video after {tries} tries. Last file: {p} | {last_err}")
        #return vid  # No labels in pretraining

class MiceClassificationDataset(Dataset):
    """Read the 'path,label' CSV and return (video_tensor, label, index, meta)."""
    def __init__(self, csv_path, num_frames=16, sampling_rate=4, input_size=224, mode="train", num_sample=1, 
            use_motion_roi=False, roi_grid=4, roi_topk=1, roi_margin=0.10, roi_min_wh=96, roi_snap16=True, roi_jitter=0, roi_prob=1.0):
        # Load rows from CSV (expects header: path,label; comma-separated)
        self.rows = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)  # columns: path,label
            for r in reader:
                p = Path(r["path"]).expanduser()
                y = int(r["label"])
                self.rows.append((p, y))

        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.input_size = input_size
        self.mode = mode # "train" "val" "test"
        self.num_sample = int(num_sample)

        # Cache of unreadable/broken videos to skip quickly next time
        self._bad = set()

        self.use_motion_roi = use_motion_roi
        self.roi_grid = roi_grid
        self.roi_topk = roi_topk
        self.roi_margin = roi_margin
        self.roi_min_wh = roi_min_wh
        self.roi_snap16 = roi_snap16
        self.roi_jitter = roi_jitter
        self.roi_prob = float(roi_prob)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        tries = min(5, len(self.rows))
        start_idx = idx
        for _ in range(tries):
            p, label = self.rows[idx]
            if p in self._bad:
                idx = (idx + 1) % len(self.rows)
                continue
            try:
                # read a clip. flag decides whether to use ROI or not
                def _read(flag):
                    if flag:
                        return _read_clip_motion_cropped(
                            p, self.num_frames, self.sampling_rate, self.input_size,
                            G=self.roi_grid, topk=self.roi_topk, margin=self.roi_margin,
                            min_wh=self.roi_min_wh, snap16=self.roi_snap16, jitter=self.roi_jitter
                        )
                    else:
                        return _read_clip_as_tensor(
                            p, self.num_frames, self.sampling_rate, self.input_size
                        )

                # > test: fixed roi, returns 5-tuple
                if self.mode == "test":
                    vid = _read(self.use_motion_roi)
                    return (
                        vid, label,
                        torch.tensor(idx, dtype=torch.int64),
                        torch.tensor(0,   dtype=torch.int64),  # chunk_nb
                        torch.tensor(0,   dtype=torch.int64),  # split_nb
                    )

                # > multiple samples in training
                if self.mode == "train" and self.num_sample > 1:
                    vids, labels, indices, chunks = [], [], [], []
                    for _k in range(self.num_sample):
                        use_roi_k = self.use_motion_roi and (random.random() < getattr(self, "roi_prob", 1.0))
                        v = _read(use_roi_k)
                        vids.append(v); labels.append(label); indices.append(idx); chunks.append(0)
                    return vids, labels, indices, chunks

                # > Single sample (train/val): training uses probabilistic selection; validation is fixed
                if self.mode == "train":
                    use_roi_once = self.use_motion_roi and (random.random() < getattr(self, "roi_prob", 1.0))
                else:  # "validation"
                    use_roi_once = self.use_motion_roi

                vid = _read(use_roi_once)
                return (
                    vid, label,
                    torch.tensor(idx, dtype=torch.int64),
                    torch.tensor(0,   dtype=torch.int64),  # chunk_nb
                )

            except Exception as e:
                print(f"Skip: {p} | {e}")
                self._bad.add(p)
                idx = (idx + 1) % len(self.rows)

        raise RuntimeError(f"No readable video after {tries} tries. Start from: {self.rows[start_idx][0]}")



#class MiceClassificationDataset(Dataset):
#    """Read the "path,label" CSV and return (video_tensor, label)"""
#    def __init__(self, csv_path, num_frames=16, sampling_rate=4, input_size=224):
#        self.rows = [] # to store all the rows in the csv file
#        with open(csv_path, "r", newline="") as f:
#            reader = csv.DictReader(f) # read each row as a dict {"path": "...", "label": "1"}
#            for r in reader:
#                self.rows.append((Path(r["path"]).expanduser(), int(r["label"])))
#                # Path(...).expanduser() expands a user-home shortcut like ~ into an absolute path prefix (e.g., ~/data -> /home/you/data). 
#        self.num_frames = num_frames
#        self.sampling_rate = sampling_rate
#        self.input_size = input_size
#
#    def __len__(self): return len(self.rows) # the length of dataset
#
#    def __getitem__(self, idx): # input the index, get the vid and the label
#        p, label = self.rows[idx]
#        vid = _read_clip_as_tensor(p, self.num_frames, self.sampling_rate, self.input_size)
#        return vid, label
