"""Report videos from recorded demonstrations: wrist camera clips, RGB | depth, and a grid of episodes (H.264).

Reads the intermediate episode folders written by ``scripts/record_demos.py`` (``wrist/*.png``, ``wrist_depth/*.png``,
``states.npy``, ``meta.json``) and writes H.264 mp4 files (yuv420p, plays in PowerPoint and browsers; the data sets
themselves hold AV1). CPU only.

Usage (from the repository root):

    uv run --project tools/lerobot_export python tools/lerobot_export/make_report_videos.py \\
        --set fixed=datasets/intermediate/teacher20b_vdr_depth_300 \\
        --set husky=datasets/intermediate/husky26a_moving_vdr_depth_300 \\
        --set stewart=datasets/intermediate/stewart25h_moving_vdr_depth_300 --out outputs/report_videos
"""

import argparse
import glob
import json
import os

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 25


def read_meta(ep_dir):
    with open(os.path.join(ep_dir, "meta.json")) as f:
        return json.load(f)


def load_episode(ep_dir):
    meta = read_meta(ep_dir)
    states = np.load(os.path.join(ep_dir, "states.npy"))
    rgb = sorted(glob.glob(os.path.join(ep_dir, "wrist", "*.png")))
    depth = sorted(glob.glob(os.path.join(ep_dir, "wrist_depth", "*.png")))
    return meta, states, rgb, depth


def label(img, lines, size=18):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    y = 6
    for text in lines:
        w = draw.textlength(text, font=font)
        draw.rectangle([4, y - 2, 10 + w, y + size + 2], fill=(0, 0, 0))
        draw.text((7, y), text, fill=(255, 255, 255), font=font)
        y += size + 6
    return img


def depth_view(path, size):
    """16-bit mm -> gray (0.1 m bright, 3 m dark), 0 (no reading) black, at the RGB size."""
    mm = np.asarray(Image.open(path)).astype(np.float32)
    vis = np.where(mm > 0, 255 - np.clip((mm - 100) / 2900 * 230, 0, 230), 0).astype(np.uint8)
    return Image.fromarray(np.stack([vis] * 3, -1)).resize(size)


def writer(path, width, height):
    container = av.open(path, "w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "20", "preset": "medium"}
    return container, stream


def write_frames(path, frames):
    first = frames[0]
    container, stream = writer(path, first.width - first.width % 2, first.height - first.height % 2)
    for img in frames:
        frame = av.VideoFrame.from_image(img.crop((0, 0, stream.width, stream.height)))
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def episode_lines(name, meta, k, states):
    ep = meta["episode"]
    base = ep.get("base_motion", "fixed base")
    preset = ep.get("base_motion_preset")
    lines = [f"{name}: {base}" + (f" ({preset})" if preset else "") + f", drone hover {ep.get('hover_preset')}"]
    kill = meta.get("drone_kill") or []
    phase = "motors off" if kill and k >= kill[0]["frame"] else ""
    lines.append(f"t {k / FPS:4.1f} s  gripper {states[k, 6]:.2f}  {phase}")
    return lines


def pick(eps, preset=None, n=1):
    out = []
    for d in eps:
        meta = read_meta(d)
        if preset is None or meta["episode"].get("base_motion_preset") == preset:
            out.append(d)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", action="append", required=True, help="name=intermediate folder")
    ap.add_argument("--out", default="outputs/report_videos")
    ap.add_argument("--per_set", type=int, default=2, help="Clips per set (per preset for presets).")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    chosen = []  # (name, episode dir)
    for item in args.set:
        name, folder = item.split("=", 1)
        eps = sorted(d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d))
        presets = sorted({read_meta(d)["episode"].get("base_motion_preset") for d in eps[:200]} - {None})
        if presets:
            for pr in ("small", "mid", "big"):
                if pr in presets:
                    chosen += [(f"{name}-{pr}", d) for d in pick(eps, pr, args.per_set)]
        else:
            chosen += [(name, d) for d in eps[: args.per_set]]

    for i, (name, d) in enumerate(chosen):
        meta, states, rgb, depth = load_episode(d)
        frames, pair = [], []
        for k, path in enumerate(rgb):
            img = label(Image.open(path).convert("RGB"), episode_lines(name, meta, k, states))
            frames.append(img)
            if depth:
                both = Image.new("RGB", (img.width * 2, img.height))
                both.paste(img, (0, 0))
                both.paste(
                    label(depth_view(depth[k], img.size), ["depth (D435i-like, black = no reading)"]), (img.width, 0)
                )
                pair.append(both)
        stem = f"{i:02d}_{name}_{os.path.basename(d)[-6:]}"
        write_frames(os.path.join(args.out, f"{stem}_wrist.mp4"), frames)
        if pair:
            write_frames(os.path.join(args.out, f"{stem}_wrist_rgb_depth.mp4"), pair)
        print(f"[videos] {stem}: {len(frames)} frames ({len(frames) / FPS:.1f} s)")

    # grid: the first episode of each kind (fixed, husky, stewart-small/mid/big), half size; shorter ones hold
    # their last frame
    cols = 3
    first = {}
    for name, d in chosen:
        first.setdefault(name, d)
    tiles = [(name, load_episode(d)) for name, d in list(first.items())[: cols * 2]]
    length = max(len(t[1][2]) for t in tiles)
    w, h = 320, 240
    rows = (len(tiles) + cols - 1) // cols
    grid = []
    for k in range(length):
        canvas = Image.new("RGB", (w * cols, h * rows))
        for j, (name, (_, _, rgb, _)) in enumerate(tiles):
            kk = min(k, len(rgb) - 1)
            img = Image.open(rgb[kk]).convert("RGB").resize((w, h))
            canvas.paste(label(img, [name, f"t {kk / FPS:4.1f} s"], size=14), ((j % cols) * w, (j // cols) * h))
        grid.append(canvas)
    write_frames(os.path.join(args.out, "grid_wrist.mp4"), grid)
    print(f"[videos] grid_wrist.mp4: {len(tiles)} episodes, {length / FPS:.1f} s -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
