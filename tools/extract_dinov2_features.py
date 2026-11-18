"""
Sample script: extract frame-wise DINOv2 (ViT-L/14) features for a video, in the format expected by the dataset classes.

For every frame of the (optionally trimmed) video the script stores one feature vector, i.e. an array of shape [T, 1024]
(float32). The output path should be

    <DATA.PATH_TO_DATA_DIR>/features/dinov2_vitl14_spatial/<action_type>_<video_id>/<view>.npy     (Assembly101)
    <DATA.PATH_TO_DATA_DIR>/features/dinov2_vitl14_spatial/<video_name>.npy                         (IKEA ASM)

Preprocessing: resize the short side to 256 (bilinear), center crop 224x224, normalize with mean 0.45 / std 0.225.
The DINOv2 weights are downloaded via torch.hub (facebookresearch/dinov2, pinned commit) and are NOT fine-tuned.

Example (Assembly101, one view of the 'assembly' part of a recording, frames 2833-6959 at 30 fps):
    python tools/extract_dinov2_features.py \
        --video /path/to/videos/<video_id>/C10395_rgb.mp4 \
        --out /path/to/assembly101/features/dinov2_vitl14_spatial/assembly_<video_id>/C10395_rgb.npy \
        --start-frame 2833 --end-frame 6959
"""
import argparse
import os
import subprocess

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

MEAN, STD = 0.45, 0.225
# pinned commit of facebookresearch/dinov2 (Dec 2023): reproducible and compatible with Python 3.9
DINOV2_URL = "https://github.com/facebookresearch/dinov2.git"
DINOV2_COMMIT = "2302b6bf46953431b969155307b9bed152754069"


def get_dinov2_repo():
    """Clones the pinned commit of the DINOv2 repository (once) and returns its local path."""
    path = os.path.join(torch.hub.get_dir(), f"dinov2_{DINOV2_COMMIT[:8]}")
    if not os.path.isdir(path):
        os.makedirs(path)
        subprocess.run(["git", "init", "-q", path], check=True)
        subprocess.run(["git", "-C", path, "fetch", "-q", "--depth", "1", DINOV2_URL, DINOV2_COMMIT], check=True)
        subprocess.run(["git", "-C", path, "checkout", "-q", "FETCH_HEAD"], check=True)
    return path


def read_frames(video_path, start_frame, end_frame):
    """Yields RGB uint8 frames [H, W, 3] in [start_frame, end_frame] (inclusive); end_frame=-1 means until the end."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    while end_frame < 0 or idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()


def preprocess(batch, short_side=256, crop=224):
    """batch: uint8 tensor [B, H, W, 3] -> float tensor [B, 3, crop, crop]"""
    x = batch.permute(0, 3, 1, 2).float() / 255.0
    _, _, h, w = x.shape
    scale = short_side / min(h, w)
    x = F.interpolate(x, size=(round(h * scale), round(w * scale)), mode="bilinear",
                      antialias=True, align_corners=False)
    h, w = x.shape[-2:]
    top, left = (h - crop) // 2, (w - crop) // 2
    x = x[:, :, top: top + crop, left: left + crop]
    return (x - MEAN) / STD


@torch.no_grad()
def extract(video_path, start_frame=0, end_frame=-1, batch_size=64, model_name="dinov2_vitl14", device="cuda"):
    model = torch.hub.load(get_dinov2_repo(), model_name, source="local").to(device).eval()
    feats, buffer = [], []

    def flush():
        x = preprocess(torch.from_numpy(np.stack(buffer))).to(device)
        feats.append(model(x).float().cpu().numpy())  # [B, D]
        buffer.clear()

    for frame in tqdm(read_frames(video_path, start_frame, end_frame), desc=os.path.basename(video_path)):
        buffer.append(frame)
        if len(buffer) == batch_size:
            flush()
    if buffer:
        flush()
    return np.concatenate(feats, axis=0)  # [T, D]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="path to the video file")
    parser.add_argument("--out", required=True, help="path of the output .npy file")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1, help="inclusive; -1: until the end of the video")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="dinov2_vitl14", help="dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    features = extract(args.video, args.start_frame, args.end_frame, args.batch_size, args.model, args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, features, allow_pickle=False)
    print(f"Saved {features.shape} to {args.out}")
