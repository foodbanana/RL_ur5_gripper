"""PNG frames of a folder -> H.264 mp4 (yuv420p), e.g. the frames of ``scripts/render_scene_video.py``.

Usage (from the repository root):

    uv run --project tools/lerobot_export python tools/lerobot_export/frames_to_mp4.py outputs/scene_frames/stewart \\
        outputs/report_videos/scene_stewart.mp4 --fps 30 [--label "Stewart, mid"]
"""

import argparse
import glob
import os

import make_report_videos
from make_report_videos import label, write_frames
from PIL import Image

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("frames")
ap.add_argument("out")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--label", type=str, default=None)
args = ap.parse_args()

make_report_videos.FPS = args.fps
paths = sorted(glob.glob(os.path.join(args.frames, "*.png")))
frames = []
for k, p in enumerate(paths):
    img = Image.open(p).convert("RGB")
    if args.label:
        img = label(img, [args.label, f"t {k / args.fps:4.1f} s"], size=22)
    frames.append(img)
write_frames(args.out, frames)
print(f"[frames_to_mp4] {len(frames)} frames -> {args.out}")
