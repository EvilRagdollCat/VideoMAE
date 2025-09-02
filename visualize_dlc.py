import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from decord import VideoReader, cpu
import pandas as pd

def visualize_dlc_roi(video_path, dlc_csv_path, output_dir, 
                      sample_frames=10, num_frames=16, sampling_rate=4):
    """
    visualize dlc roi
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # > load vid
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    
    # > load dlc
    if dlc_csv_path and Path(dlc_csv_path).exists():
        # > read dlc csv
        dlc_df = pd.read_csv(dlc_csv_path, header=[1, 2], index_col=0)
        
        # > simulate videomae frame sampling
        span = (num_frames - 1) * sampling_rate + 1
        if total_frames >= span:
            
            for sample_idx in range(sample_frames):
                # > random start frame
                start = np.random.randint(0, total_frames - span)
                frame_indices = [start + i * sampling_rate for i in range(num_frames)]
                
                # > get dlc roi
                y1, y2, x1, x2 = get_roi_from_dlc(dlc_df, frame_indices)
                frames_to_show = frame_indices[::4]
                combined_frames = []
                
                # > show sampled frames
                #for i, frame_idx in enumerate(frame_indices[::4]):  # every 4 frames
                for frame_idx in frames_to_show:
                    frame = vr[frame_idx].asnumpy()
                    frame_vis = frame.copy()
                    
                    
                    # > draw roi bbox
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # > draw keypoints
                    for bodypart in dlc_df.columns.levels[0]:
                        if frame_idx < len(dlc_df):
                            x = dlc_df[bodypart]['x'].iloc[frame_idx]
                            y = dlc_df[bodypart]['y'].iloc[frame_idx]
                            likelihood = dlc_df[bodypart]['likelihood'].iloc[frame_idx]
                            if likelihood > 0.5 and not np.isnan(x):
                                cv2.circle(frame_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
                    
                    cv2.putText(frame_vis, f'Frame {frame_idx}',
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                               1, (255, 255, 255), 2)
                    combined_frames.append(frame_vis)
                combined_image = np.hstack(combined_frames)
                
                # > save sampling info
                output_path = output_dir / f'{Path(video_path).stem}_sample_{sample_idx}.jpg'
                cv2.imwrite(str(output_path), cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
                with open(output_dir / f'sample_{sample_idx}_frames.txt', 'w') as f:
                    f.write(f"Video: {video_path}\n")
                    f.write(f"Start frame: {start}\n")
                    f.write(f"Sampled frames: {frame_indices}\n")
                    f.write(f"Displayed frames: {frames_to_show}\n")
                    f.write(f"ROI: y1={y1}, y2={y2}, x1={x1}, x2={x2}\n")
                    f.write(f"ROI size: {y2-y1}x{x2-x1}\n")
            print(f"Saved {sample_frames} samples at {output_dir}")
    else:
        print(f"No dlc file found: {dlc_csv_path}")
            


def get_roi_from_dlc(dlc_df, frame_indices, padding=0.2, min_size=96):
    """
    Calculate roi from dlc
    """
    all_x, all_y = [], []
    
    for idx in frame_indices:
        if idx >= len(dlc_df):
            idx = len(dlc_df) - 1
        
        for bodypart in dlc_df.columns.levels[0]:
            x = dlc_df[bodypart]['x'].iloc[idx]
            y = dlc_df[bodypart]['y'].iloc[idx]
            likelihood = dlc_df[bodypart]['likelihood'].iloc[idx]
            
            if likelihood > 0.5 and not np.isnan(x):
                all_x.append(x)
                all_y.append(y)
    
    if not all_x:
        return 0, min_size, 0, min_size
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # > padding
    width = x_max - x_min
    height = y_max - y_min
    x_min -= width * padding
    x_max += width * padding
    y_min -= height * padding
    y_max += height * padding
    
    return int(y_min), int(y_max), int(x_min), int(x_max)

def log_training_frames(output_dir='logs/frame_usage'):
    """
    Document the frames used in training
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    class FrameLogger:
        def __init__(self):
            self.log_file = open(f'{output_dir}/frame_log.csv', 'w')
            self.log_file.write('epoch,batch,video_path,start_frame,sampled_frames,roi\n')
        
        def log(self, epoch, batch, video_path, start_frame, sampled_frames, roi):
            self.log_file.write(f'{epoch},{batch},{video_path},{start_frame},"{sampled_frames}","{roi}"\n')
            self.log_file.flush()
        
        def close(self):
            self.log_file.close()
    
    return FrameLogger()


def batch_visualize_rois(data_csv, dlc_dir, output_dir='visualization/roi_validation'):
    """
    validate all rois in all vids in bulk
    """
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    
    for idx, row in df.iterrows():
        if idx >= 5:  # only process the first 5 vids
            break
        video_path = row['path']
        video_name = Path(video_path).stem
        
        # > find dlc
        dlc_pattern = f"{video_name}*.csv"
        dlc_files = list(Path(dlc_dir).glob(dlc_pattern))
        
        if dlc_files:
            print(f"Processing {video_name}...")
            visualize_dlc_roi(
                video_path=video_path,
                dlc_csv_path=dlc_files[0],
                output_dir=output_dir / video_name,
                sample_frames=num_samples,  # 3 sampes for each vid
                num_frames=16,
                sampling_rate=4
            )

if __name__ == '__main__':
    # > validate one vid
    visualize_dlc_roi(
        video_path='/data/datasets/ForKeypointMouseSeq/Female-cKO/28_cKO_F.mp4',
        dlc_csv_path='/data/dlc/test-yd-2025-08-27/videos/28_cKO_FDLC_HrnetW32_testAug27shuffle1_detector_best-200_snapshot_best-30_filtered.csv',
        output_dir='/data/videomae_outputs/visualization/',
        sample_frames=20
    )
    
    # > validate vids in bulk
    # validate_all_rois(
    #     data_csv='data/mice_classification/train_sex.csv',
    #     dlc_dir='/data/dlc/test-yd...-08-27/videos',
    #     output_dir='visualization/all_rois'
    # )
