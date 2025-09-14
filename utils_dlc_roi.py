"""
Changed the way of reading dlc roi data
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import cv2


def load_dlc_tracking(dlc_path: str) -> pd.DataFrame:
    """
    Load DeepLabCut tracking data with multi-level column index
    """
    # > convert string path to "Path" type
    dlc_path = Path(dlc_path)
    
    if dlc_path.suffix == '.csv':
        # > read .csv file without the headers
        # DLC CSV has 3 header rows: scorer, bodyparts, coords
        # This creates MultiIndex columns: (bodypart, coordinate)
        df = pd.read_csv(dlc_path, header=[1, 2], index_col=0)
        # read header[1, 2] as labels: row1 bodyparts, row2 (x, y, likelihood)
        # creates multiindex column structure
        # index_col = 0: uses the 0th col as the index
        # dataframe structure: df['nosetip']['x'].iloc[0] is the xcoord of nosetip in the 0th frame
        return df
    elif dlc_path.suffix in ['.h5', '.hdf5']:
        # > if reads .h5
        df = pd.read_hdf(dlc_path) # read directly
        return df
    else:
        raise ValueError(f"Unsupported format: {dlc_path.suffix}")

def _read_clip_with_dlc(path, num_frames, sampling_rate, input_size,
                        dlc_manager, dlc_likelihood_threshold=0.5,
                        roi_padding=0.2, roi_min_size=96, roi_snap16=False):
    vr = VideoReader(str(path), ctx=cpu(0))
    total = len(vr)
    idx = _uniform_sample_indices(num_frames, total, sampling_rate)
    
    dlc_df = dlc_manager.load_dlc_data(path)
    
    if dlc_df is None:
        frames = vr.get_batch(idx).asnumpy()
        frames = np.stack([cv2.resize(f, (input_size, input_size)) for f in frames])
    else:
        frames = vr.get_batch(idx).asnumpy()
        H, W = frames.shape[1], frames.shape[2]
        
        crops = []
        for i, frame_idx in enumerate(idx):
            # > calculate bbox for each frame
            y1, y2, x1, x2 = get_bbox_from_dlc(
                dlc_df, frame_idx,
                likelihood_threshold=dlc_likelihood_threshold,
                margin=15,  # fixed margin
                min_size=roi_min_size
            )
            
            # > make sure bbox is in the boundary
            y1 = max(0, min(y1, H-1))
            y2 = max(y1+1, min(y2, H))
            x1 = max(0, min(x1, W-1))
            x2 = max(x1+1, min(x2, W))
            
            # > crop the current frame
            crop = frames[i, y1:y2, x1:x2]
            
            # > resize to the target size
            crop = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_AREA)
            crops.append(crop)
        
        frames = np.stack(crops)
    
    frames = frames.astype(np.float32) / 255.0
    frames = (frames - IMAGENET_MEAN[None, None, None, :]) / IMAGENET_STD[None, None, None, :]
    frames = np.transpose(frames, (3, 0, 1, 2))
    
    return torch.from_numpy(frames.astype(np.float32))


def get_bbox_from_dlc(dlc_df: pd.DataFrame, 
                      #frame_indices: List[int],
                      frame_idx: int,
                      likelihood_threshold: float = 0.6,
                      padding: float = 0.2,
                      min_size: int = 1,
                      margin: int = 15,
                      max_bodyparts: int = None,  # Use None for all, or limit
                      video_path: Path = None) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box from DLC tracking data with multi-level index
    """
    if frame_idx >= len(dlc_df):
        frame_idx = len(dlc_df) - 1
    all_x, all_y = [], []

    # > Get bodypart names from the first level of column index
    bodyparts = dlc_df.columns.get_level_values(0).unique()


    # > Optionally limit the number of bodyparts used
    if max_bodyparts is not None:
        bodyparts = bodyparts[:min(max_bodyparts, len(bodyparts))]

    # > Debug: print structure on first call
    if not hasattr(get_bbox_from_dlc, '_structure_printed'):
        # hasattr(object, attribute_name): checks if an object has a certain attribute
        # first time called: no such attribute, returns false (in this case will then print bodyparts), add this attribute
        # next time called: attribute added, reurns true (in this case will skip)

        print(f"\n[DLC Structure] Bodyparts found: {list(bodyparts)}")
        get_bbox_from_dlc._structure_printed = True

    for bodypart in bodyparts:
        try: # for external data
            # > access multi-level indexed data
            x = dlc_df[bodypart]['x'].iloc[frame_idx]
            y = dlc_df[bodypart]['y'].iloc[frame_idx]
            likelihood = dlc_df[bodypart]['likelihood'].iloc[frame_idx]

            # > check validity
            if not np.isnan(x) and not np.isnan(y) and likelihood > likelihood_threshold:
                all_x.append(x)
                all_y.append(y)
        except (KeyError, IndexError) as e:
            # > skip if bodypart doesn't exist or other access errors
            continue
    
    # > if no valid points found, return default bbox
    if not all_x:
        return (0, min_size, 0, min_size)
    
    # > calculate dlc roi to include all the keypoints in this frame
    x_min = int(np.floor(min(all_x))) - margin  # smaller margin
    x_max = int(np.ceil(max(all_x))) + margin
    y_min = int(np.floor(min(all_y))) - margin
    y_max = int(np.ceil(max(all_y))) + margin
    
    # > maske sure it's in the bbox
    x_min = max(0, x_min)
    y_min = max(0, y_min)

    # > ensure minimum size
    # find the center and expand x/y by min_size
    if (x_max - x_min) < min_size:
        center = (x_max + x_min) // 2
        x_min = max(0, center - min_size // 2)
        x_max = x_min + min_size
    if (y_max - y_min) < min_size:
        center = (y_min + y_max) // 2
        y_min = max(0, center - min_size // 2)
        y_max = y_min + min_size
    
    return (int(y_min), int(y_max), int(x_min), int(x_max))

get_bbox_from_dlc.first_call = True
def visualize_debug_info(debug_info, video_path):
    
    import cv2
    from pathlib import Path

    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, img = cap.read()
    cap.release()

    if not ret:
        print("cant read frame")
        return

    
    for kpt in debug_info['keypoints']:
        if not np.isnan(kpt['x']):
            x, y = int(kpt['x']), int(kpt['y'])
            color = (0, 255, 0) if kpt['used'] else (0, 0, 255)  
            thickness = 2 if kpt['used'] else 1

        
            cv2.circle(img, (x, y), 8, color, -1)
            cv2.circle(img, (x, y), 10, (255, 255, 255), 2)

            
            text = f"K{kpt['idx']}:c={kpt['conf']:.2f}"
            cv2.putText(img, text, (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img, text, (x+12, y-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


    x_min_orig, x_max_orig = debug_info['x_range']
    y_min_orig, y_max_orig = debug_info['y_range']
    cv2.rectangle(img,
                  (int(x_min_orig), int(y_min_orig)),
                  (int(x_max_orig), int(y_max_orig)),
                  (255, 0, 0), 2)  

    
    x_min, y_min, x_max, y_max = debug_info['bbox']
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  (0, 0, 0), 3)  
    info_text = f"BBox: ({x_min},{y_min})-({x_max},{y_max})"
    cv2.rectangle(img, (0, 0), (500, 30), (255, 255, 255), -1)
    cv2.putText(img, info_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    
    cv2.putText(img, "Green=Used, Red=Skipped", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Blue=Original Range", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, "Black=Final BBox", (50, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    
    info_text = [
        f"X range: {x_min_orig:.1f} - {x_max_orig:.1f}",
        f"Y range: {y_min_orig:.1f} - {y_max_orig:.1f}",
        f"Final bbox: ({x_min}, {y_min}) to ({x_max}, {y_max})"
    ]

    for i, text in enumerate(info_text):
        cv2.putText(img, text, (50, 150 + i*30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    
    cv2.imwrite("/data/videomae_outputs/pretrain_crop_check/dlc_debug_visualization.jpg", img)
    print("saved at: /data/videomae_outputs/pretrain_crop_check/dlc_debug_visualization.jpg")

class DLCManager:
    """
    File management
    """
    def __init__(self, dlc_dir: Optional[Path] = None):
        self.dlc_dir = Path(dlc_dir) if dlc_dir else None
        self._cache = {}
    
    def find_dlc_file(self, video_path: Path) -> Optional[Path]:
        """
        Search dlc file
        """
        video_path = Path(video_path)
        video_stem = video_path.stem
        base_pattern = video_stem.replace('_M', '_MDLC').replace('_F', '_FDLC')

        if self.dlc_dir:
            # > find the filtered version
            filtered_pattern = f"{base_pattern}*_filtered.csv"
            filtered_files = list(self.dlc_dir.glob(filtered_pattern))
            if filtered_files:
                return filtered_files[0]

            # > if not filtered versions
            normal_pattern = f"{base_pattern}*.csv"
            normal_files = list(self.dlc_dir.glob(normal_pattern))
            
            normal_files = [f for f in normal_files if '_filtered' not in f.name]
            if normal_files:
                return normal_files[0]

        return None

        ## > try the files under the dir
        #for suffix in ['_DLC.csv', '_DLC.h5', 'DLC_resnet50_*.csv']:
        #    pattern = f"{video_path.stem}*{suffix}"
        #    candidates = list(video_path.parent.glob(pattern))
        #    if candidates:
        #        return candidates[0]
        
        ## > try dlc dir
        #if self.dlc_dir:
        #    for suffix in ['_DLC.csv', '_DLC.h5', 'DLC_resnet50_*.csv']:
        #        pattern = f"{video_path.stem}*{suffix}"
        #        candidates = list(self.dlc_dir.glob(pattern))
        #        if candidates:
        #            return candidates[0]
        
        #return None
    
    def load_dlc_data(self, video_path: Path) -> Optional[pd.DataFrame]:
        """
        Load dlc data
        """
        video_path = Path(video_path)
        
        if str(video_path) in self._cache:
            return self._cache[str(video_path)]
        
        dlc_path = self.find_dlc_file(video_path)
        if dlc_path is None:
            return None
        
        try:
            df = load_dlc_tracking(str(dlc_path))
            self._cache[str(video_path)] = df
            return df
        except Exception as e:
            print(f"Error loading {dlc_path}: {e}")
            return None
