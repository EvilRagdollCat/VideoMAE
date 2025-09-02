import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict

def load_dlc_tracking(dlc_path: str) -> pd.DataFrame:
    """
    Load DeepLabCut tracking data
    """
    dlc_path = Path(dlc_path)
    
    if dlc_path.suffix == '.csv':
        # > read .csv file without the headers
        df = pd.read_csv(dlc_path, header=[1, 2], index_col=0)
        return df
    elif dlc_path.suffix in ['.h5', '.hdf5']:
        df = pd.read_hdf(dlc_path)
        return df
    else:
        raise ValueError(f"Unsupported format: {dlc_path.suffix}")

def get_bbox_from_dlc(dlc_df: pd.DataFrame, 
                      frame_indices: List[int],
                      likelihood_threshold: float = 0.5,
                      padding: float = 0.2,
                      min_size: int = 96) -> Tuple[int, int, int, int]:
    """
    Get bboxs from dlc
    """
    all_x, all_y = [], []
    
    for idx in frame_indices:
        if idx >= len(dlc_df):
            idx = len(dlc_df) - 1
        
        row = dlc_df.iloc[idx]
        
        # > Get the coordinates: (x, y, likelihood) 
        for i in range(0, len(row), 3):
            if i+2 < len(row):
                #x, y, likelihood = row[i], row[i+1], row[i+2]
                x, y, likelihood = row.iloc[i], row.iloc[i+1], row.iloc[i+2]
                if likelihood > likelihood_threshold and not np.isnan(x):
                    all_x.append(x)
                    all_y.append(y)
    
    if not all_x:
        return (0, min_size, 0, min_size)
    
    # > Calculate bbox
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # > padding
    width = x_max - x_min
    height = y_max - y_min
    x_min -= width * padding
    x_max += width * padding
    y_min -= height * padding
    y_max += height * padding
    
    # > min size
    if x_max - x_min < min_size:
        center = (x_min + x_max) / 2
        x_min = center - min_size / 2
        x_max = center + min_size / 2
    
    if y_max - y_min < min_size:
        center = (y_min + y_max) / 2
        y_min = center - min_size / 2
        y_max = center + min_size / 2
    
    return (int(y_min), int(y_max), int(x_min), int(x_max))

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
