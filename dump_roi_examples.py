# tools/dump_roi_examples.py
import os, csv, random, argparse
from pathlib import Path
import numpy as np
import cv2

try:
    from decord import VideoReader, cpu
except ImportError:
    raise ImportError("Please `pip install decord`")

from utils_motion_roi import roi_by_tiles

def uniform_indices(num_frames, total, sampling_rate):
    span = (num_frames - 1) * sampling_rate + 1
    if total >= span:
        start = random.randint(0, total - span)
        return [start + i * sampling_rate for i in range(num_frames)]
    base = list(range(0, total, max(1, sampling_rate)))[:num_frames]
    if len(base) == 0: base = [0]
    while len(base) < num_frames:
        base.append(base[-1])
    return base[:num_frames]

def draw_and_save(frames, y1, y2, x1, x2, input_size, out_dir, tag):
    """
      1) *_frame.jpg : img with bbox
      2) *_crop.jpg  : crop ROI and resize to input_size
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames.shape
    mid = min(T//2, T-1)

    # > red bbox
    vis = frames[mid].copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0,0,255), 2)
    cv2.imwrite(str(Path(out_dir) / f"{tag}_frame.jpg"), vis[:, :, ::-1])

    # > crop and resize
    crop = frames[mid, y1:y2, x1:x2]
    crop = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(Path(out_dir) / f"{tag}_crop.jpg"), crop[:, :, ::-1])

def process_one(path, num_frames, sampling_rate, input_size,
                G, topk, margin, min_wh, snap16, out_dir, tag):
    vr = VideoReader(str(path), ctx=cpu(0))
    total = len(vr)
    idx = uniform_indices(num_frames, total, sampling_rate)
    frames = vr.get_batch(idx).asnumpy()  # (T,H,W,3) RGB uint8

    T, H, W, _ = frames.shape
    y1, y2, x1, x2 = roi_by_tiles(frames, G=G, topk=topk, margin=margin, min_wh=min_wh)

    if snap16:
        side = max(min_wh, max(y2 - y1, x2 - x1))
        side = int(np.ceil(side / 16.0) * 16)
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        y1 = max(0, min(H - side, cy - side // 2)); y2 = y1 + side
        x1 = max(0, min(W - side, cx - side // 2)); x2 = x1 + side

    draw_and_save(frames, y1, y2, x1, x2, input_size, out_dir, tag)

def read_csv_list(csv_path):
    items = []
    with open(csv_path, "r", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            items.append(Path(r["path"]))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_root", required=True,
                    help="the root dir of the csv files")
    ap.add_argument("--out_dir", required=True,
                    help="out put dir")
    ap.add_argument("--num_train", type=int, default=2)
    ap.add_argument("--num_test", type=int, default=2)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--sampling_rate", type=int, default=4)
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--roi_grid", type=int, default=4)
    ap.add_argument("--roi_topk", type=int, default=1)
    ap.add_argument("--roi_margin", type=float, default=0.10)
    ap.add_argument("--roi_min_wh", type=int, default=96)
    ap.add_argument("--roi_snap16", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)

    train_csv = Path(args.csv_root) / "train.csv"
    test_csv  = Path(args.csv_root) / "test.csv"
    train_list = read_csv_list(train_csv)
    test_list  = read_csv_list(test_csv)

    # > Extract 2 in train
    for k, p in enumerate(random.sample(train_list, min(args.num_train, len(train_list)))):
        try:
            process_one(p, args.num_frames, args.sampling_rate, args.input_size,
                        args.roi_grid, args.roi_topk, args.roi_margin, args.roi_min_wh,
                        args.roi_snap16, Path(args.out_dir)/"train",
                        tag=f"train_{k}_{p.stem}")
        except Exception as e:
            print(f"[train] skip {p}: {e}")

    # > Extract 2 in test
    for k, p in enumerate(random.sample(test_list, min(args.num_test, len(test_list)))):
        try:
            process_one(p, args.num_frames, args.sampling_rate, args.input_size,
                        args.roi_grid, args.roi_topk, args.roi_margin, args.roi_min_wh,
                        args.roi_snap16, Path(args.out_dir)/"test",
                        tag=f"test_{k}_{p.stem}")
        except Exception as e:
            print(f"[test] skip {p}: {e}")

if __name__ == "__main__":
    main()

