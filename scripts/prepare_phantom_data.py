#!/usr/bin/env python3
"""
Build Phantom-ready demos from folder-based RGB / mask / depth / meta frames.

Input expected layout (like your existing scripts):
  rgb_base_dir/demo_*/frame_*.png
  mask_base_dir/demo_*/frame_*.png          (grayscale)
  depth_base_dir/demo_*/frame_*.png         (uint16 mm or float png)
  meta_base_dir/demo_*/frame_*.json         (robot0_eef_pos, robot0_eef_quat_xyzw, robot0_gripper_qpos)

Output Phantom layout:
  out_root/<demo_idx>/
    camera_meta.json
    depth.npy
    inpaint_processor/video_human_inpaint.mkv
    segmentation_processor/masks_arm.npy
    action_processor/actions_right_single_arm.npz
    smoothing_processor/smoothed_actions_right_single_arm.npz
    meta.json   (debug bookkeeping)
"""

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np

try:
    from scipy.spatial.transform import Rotation
except Exception as e:
    raise RuntimeError("Please install scipy: pip install scipy") from e


def _sorted_demo_dirs(base: Path):
    return sorted([p for p in base.glob("demo_*") if p.is_dir()],
                  key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 10**18)


def _sorted_frames(demo_dir: Path, pattern="frame_*.png"):
    return sorted(glob.glob(str(demo_dir / pattern)))


def _sorted_meta(demo_dir: Path, pattern="frame_*.json"):
    return sorted(glob.glob(str(demo_dir / pattern)))


def _read_rgb(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
    if img is None:
        raise RuntimeError(f"Failed to read rgb: {path}")
    return img


def _read_mask(path: str):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # uint8
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    return m


def _read_depth(path: str):
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f"Failed to read depth: {path}")
    return d


def _resize_if_needed(img: np.ndarray, target_hw):
    th, tw = target_hw
    h, w = img.shape[:2]
    if (h, w) == (th, tw):
        return img
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_AREA
    return cv2.resize(img, (tw, th), interpolation=interp)


def _write_mkv(frames_bgr, out_path: Path, fps: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open video writer for {out_path}. "
            "Try changing extension, codec, or install ffmpeg codecs."
        )
    for f in frames_bgr:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(f)
    writer.release()


def _meta_to_npz(meta_files):
    ee_pts = []
    ee_oris = []
    ee_widths = []
    union_indices = []

    for i, mf in enumerate(meta_files):
        with open(mf, "r") as f:
            m = json.load(f)

        ee_pts.append(m["robot0_eef_pos"])
        quat_xyzw = m["robot0_eef_quat_xyzw"]
        R_mat = Rotation.from_quat(quat_xyzw).as_matrix()
        ee_oris.append(R_mat)

        qpos = m["robot0_gripper_qpos"]
        width = abs(qpos[0] - qpos[1])
        ee_widths.append(width)

        union_indices.append(i)

    ee_pts = np.asarray(ee_pts, dtype=np.float32)              # (T,3)
    ee_oris = np.stack(ee_oris, axis=0).astype(np.float32)     # (T,3,3)
    ee_widths = np.asarray(ee_widths, dtype=np.float32)        # (T,)
    union_indices = np.asarray(union_indices, dtype=np.int64)  # (T,)
    return ee_pts, ee_oris, ee_widths, union_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_base_dir", type=Path, required=True)
    ap.add_argument("--mask_base_dir", type=Path, required=True)
    ap.add_argument("--depth_base_dir", type=Path, required=True)
    ap.add_argument("--meta_base_dir", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--camera_meta", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--bimanual_setup", type=str, default="single_arm")
    ap.add_argument("--hand", type=str, default="right",
                    choices=["left", "right"])
    ap.add_argument("--max_demos", type=int, default=None)
    ap.add_argument("--depth_mm_to_m", action="store_true",
                    help="If set, converts uint16 mm depth -> float meters by /1000.")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if not args.camera_meta.exists():
        raise FileNotFoundError(args.camera_meta)

    rgb_demos = _sorted_demo_dirs(args.rgb_base_dir)
    if args.max_demos is not None:
        rgb_demos = rgb_demos[:args.max_demos]

    if not rgb_demos:
        raise FileNotFoundError(f"No demo_* dirs found in {args.rgb_base_dir}")

    for demo_idx, rgb_demo_dir in enumerate(rgb_demos):
        demo_name = rgb_demo_dir.name  # demo_0, demo_1, ...

        mask_demo_dir = args.mask_base_dir / demo_name
        depth_demo_dir = args.depth_base_dir / demo_name
        meta_demo_dir = args.meta_base_dir / demo_name

        if not mask_demo_dir.exists():
            raise FileNotFoundError(f"Missing mask dir: {mask_demo_dir}")
        if not depth_demo_dir.exists():
            raise FileNotFoundError(f"Missing depth dir: {depth_demo_dir}")
        if not meta_demo_dir.exists():
            raise FileNotFoundError(f"Missing meta dir: {meta_demo_dir}")

        rgb_files = _sorted_frames(rgb_demo_dir)
        mask_files = _sorted_frames(mask_demo_dir)
        depth_files = _sorted_frames(depth_demo_dir)
        meta_files = _sorted_meta(meta_demo_dir)

        if not (rgb_files and mask_files and depth_files and meta_files):
            raise RuntimeError(
                f"{demo_name}: missing one of rgb/mask/depth/meta frames")

        T = min(len(rgb_files), len(mask_files),
                len(depth_files), len(meta_files))
        rgb_files = rgb_files[:T]
        mask_files = mask_files[:T]
        depth_files = depth_files[:T]
        meta_files = meta_files[:T]

        # Read first RGB to define target resolution
        first_rgb = _read_rgb(rgb_files[0])
        H, W = first_rgb.shape[:2]

        # Load RGB frames (BGR)
        rgb_frames = [first_rgb]
        for p in rgb_files[1:]:
            rgb_frames.append(_resize_if_needed(_read_rgb(p), (H, W)))

        # Load masks and enforce same resolution
        masks = []
        for p in mask_files:
            m = _resize_if_needed(_read_mask(p), (H, W))
            masks.append(m)
        masks = np.stack(masks, axis=0).astype(np.uint8)  # (T,H,W)

        # Load depth and enforce same resolution
        depths = []
        for p in depth_files:
            d = _read_depth(p)
            # squeeze if (H,W,1)
            if d.ndim == 3 and d.shape[-1] == 1:
                d = d[..., 0]
            d = _resize_if_needed(d, (H, W))

            if args.depth_mm_to_m:
                # common case: uint16 mm
                d = d.astype(np.float32) / 1000.0
            else:
                d = d.astype(np.float32)
            depths.append(d)
        depth = np.stack(depths, axis=0)  # (T,H,W) float32

        # NPZ from meta json
        ee_pts, ee_oris, ee_widths, union_indices = _meta_to_npz(meta_files)

        # ---- Write Phantom structure ----
        out_demo = args.out_root / str(demo_idx)
        (out_demo / "inpaint_processor").mkdir(parents=True, exist_ok=True)
        (out_demo / "segmentation_processor").mkdir(parents=True, exist_ok=True)
        (out_demo / "action_processor").mkdir(parents=True, exist_ok=True)
        (out_demo / "smoothing_processor").mkdir(parents=True, exist_ok=True)

        # camera_meta.json
        (out_demo / "camera_meta.json").write_text(args.camera_meta.read_text())

        # video
        _write_mkv(rgb_frames, out_demo / "inpaint_processor" /
                   "video_human_inpaint.mkv", fps=args.fps)

        # masks
        np.save(out_demo / "segmentation_processor" / "masks_arm.npy", masks)

        # depth at DEMO ROOT (this fixes the path mismatch)
        np.save(out_demo / "depth.npy", depth)

        # action + smoothing npz (Phantom naming)
        hand = args.hand
        suffix = args.bimanual_setup
        np.savez_compressed(out_demo / "action_processor" / f"actions_{hand}_{suffix}.npz",
                            union_indices=union_indices)
        np.savez_compressed(out_demo / "smoothing_processor" / f"smoothed_actions_{hand}_{suffix}.npz",
                            ee_pts=ee_pts, ee_oris=ee_oris, ee_widths=ee_widths)

        # debug meta
        dbg = {
            "demo_name": demo_name,
            "demo_idx": demo_idx,
            "T": int(T),
            "H": int(H),
            "W": int(W),
            "outputs": {
                "video": "inpaint_processor/video_human_inpaint.mkv",
                "masks": "segmentation_processor/masks_arm.npy",
                "depth": "depth.npy",
                "actions": f"action_processor/actions_{hand}_{suffix}.npz",
                "smoothed": f"smoothing_processor/smoothed_actions_{hand}_{suffix}.npz",
                "camera_meta": "camera_meta.json",
            }
        }
        (out_demo / "meta.json").write_text(json.dumps(dbg, indent=2))

        print(f"[ok] {demo_name} -> {out_demo} (T={T}, HxW={H}x{W})")

    print("\nDone.")
    print(f"Wrote Phantom-ready demos to: {args.out_root}")


if __name__ == "__main__":
    main()
