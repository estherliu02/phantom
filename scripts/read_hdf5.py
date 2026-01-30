#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import csv
import h5py
import numpy as np
import cv2
from tqdm import tqdm

# -----------------------------
# Helpers
# -----------------------------


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    # HDF5 images are usually RGB; OpenCV expects BGR for correct colors
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_rgb


def depth_to_uint16_mm(depth: np.ndarray, max_mm: int = 65535) -> np.ndarray:
    """
    Convert float depth in meters to uint16 millimeters for PNG.
    NaNs/inf -> 0. Values > max range -> clipped.
    """
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    d_mm = (d * 1000.0).astype(np.float64)
    d_mm = np.clip(d_mm, 0, max_mm).astype(np.uint16)
    return d_mm


def write_json(path: Path, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def try_simple_mask(rgb: np.ndarray, depth: np.ndarray | None) -> np.ndarray:
    """
    VERY rough, optional mask guesser; defaults to blank mask if not requested.
    - Color heuristic: looks for neutral/gray robot pixels (tune thresholds)
    - Depth heuristic: close-range emphasis if depth exists
    Returns 8-bit 0/255 mask.
    """
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    # Color range heuristic (gray-ish / low saturation)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    v = hsv[..., 2]
    color_mask = ((s < 40) & (v > 60)).astype(np.uint8) * 255

    if depth is not None:
        # emphasize nearer pixels (e.g., robot close to camera)
        near = (depth > 0) & (depth < np.percentile(
            depth[depth > 0], 25))  # closest quartile
        depth_mask = near.astype(np.uint8) * 255
        combined = cv2.bitwise_and(color_mask, depth_mask)
    else:
        combined = color_mask

    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return combined

# -----------------------------
# Main export
# -----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export HDF5 to OpenCV-ready files for inpainting.")
    parser.add_argument("--hdf5", type=str, required=True,
                        help="Path to stack_d1_abs.hdf5 (e.g., /Volumes/TOSHIBA/baseline/data/raw_space_agent.hdf5)")
    parser.add_argument("--camera_meta", type=str, required=True,
                        help="Path to multi_camera_meta.json")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--cameras", type=str, default="frontview,agentview,birdview,sideview,sideview2,backview,spaceview",
                        help="Comma-separated list of cameras to export")
    parser.add_argument("--stride", type=int, default=1,
                        help="Export every Nth frame")
    parser.add_argument("--start", type=int, default=0,
                        help="Start frame index (inclusive)")
    parser.add_argument("--end", type=int, default=-1,
                        help="End frame index (exclusive); -1 = all")
    parser.add_argument("--auto_mask", action="store_true",
                        help="If set, write a rough guessed mask; otherwise write an empty mask")
    parser.add_argument("--copy_cam_meta", action="store_true",
                        help="Copy camera JSON to out_dir/meta/cameras.json")
    args = parser.parse_args()

    h5_path = Path(args.hdf5)
    cam_meta_path = Path(args.camera_meta)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "meta")

    # Load camera meta
    with open(cam_meta_path, "r") as f:
        camera_meta = json.load(f)

    if args.copy_cam_meta:
        write_json(out_dir / "meta" / "cameras.json", camera_meta)

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    # Prepare per-camera directories
    for cam in cameras:
        ensure_dir(out_dir / cam / "rgb")
        ensure_dir(out_dir / cam / "depth")
        ensure_dir(out_dir / cam / "mask")
        ensure_dir(out_dir / cam / "meta")

    manifest_path = out_dir / "meta" / "manifest.csv"
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow([
            "demo_id", "frame_idx", "camera",
            "rgb_path", "depth_path", "mask_path", "meta_path"
        ])

        with h5py.File(h5_path, "r") as f:
            if "data" not in f:
                raise RuntimeError("HDF5 file missing 'data' group.")
            data_grp = f["data"]

            demo_keys = list(data_grp.keys())
            # Sort demo keys in natural order if they look like 'demo_0', 'demo_1', ...
            try:
                demo_keys = sorted(
                    demo_keys, key=lambda k: int(k.split("_")[-1]))
            except Exception:
                demo_keys = sorted(demo_keys)

            for demo_key in demo_keys:
                demo = data_grp[demo_key]
                obs = demo["obs"]

                # Probe available frames via one modality we expect to exist
                # Use robot0_joint_pos length as authoritative
                n_frames = len(obs["robot0_joint_pos"])
                start = max(0, args.start)
                end = n_frames if args.end < 0 else min(args.end, n_frames)

                # Preload robot kinematics for metadata
                robot_joints = obs["robot0_joint_pos"][start:end:args.stride]
                gripper_q = obs["robot0_gripper_qpos"][start:end:args.stride]
                eef_pos = obs["robot0_eef_pos"][start:end:args.stride]
                eef_quat = obs["robot0_eef_quat"][start:end:args.stride]

                # Determine which cameras are actually present in this demo
                available_cams = []


                cam_has_depth = {}
                for cam in cameras:
                    rgb_key = f"{cam}_image"
                    depth_key = f"{cam}_depth"

                    if rgb_key in obs:
                        available_cams.append(cam)
                        cam_has_depth[cam] = (depth_key in obs)
                    # else: skip cameras that don't exist in this file

                    # Silently skip cameras that don't exist in this file

                # Create per-demo subdirectories for each available camera so
                # frames from the same demo are grouped together.
                for cam in available_cams:
                    ensure_dir(out_dir / cam / "rgb" / demo_key)
                    ensure_dir(out_dir / cam / "depth" / demo_key)
                    ensure_dir(out_dir / cam / "mask" / demo_key)
                    ensure_dir(out_dir / cam / "meta" / demo_key)

                frame_indices = range(start, end, args.stride)
                pbar = tqdm(frame_indices,
                            desc=f"Exporting {demo_key}", unit="frame")

                for local_idx, t in enumerate(pbar):
                    # Pull kinematics for metadata (aligned by stride)
                    joints = robot_joints[local_idx].tolist()
                    grip = gripper_q[local_idx].tolist()
                    epos = eef_pos[local_idx].tolist()
                    equat = eef_quat[local_idx].tolist()

                    # for cam in available_cams:
                    #     rgb_key = f"{cam}_image"
                    #     depth_key = f"{cam}_depth"

                    #     # Read single frame on demand (no huge preload)
                    #     # expected HxWx3 uint8 RGB
                    #     rgb = obs[rgb_key][t]
                    #     # expected HxW float (meters) or uint16 (check)
                    #     depth = obs[depth_key][t]
                    #     # Harmonize depth to uint16(mm) for OpenCV/PNG
                    #     if depth.dtype != np.uint16:
                    #         depth_u16 = depth_to_uint16_mm(depth)
                    #     else:
                    #         depth_u16 = depth

                    #     bgr = to_bgr(rgb)

                    #     # File paths: save inside a demo subfolder and avoid
                    #     # repeating the demo id in the filename itself.
                    #     stem = f"frame_{t:06d}"
                    #     rgb_path = out_dir / cam / "rgb" / demo_key / f"{stem}.png"
                    #     depth_path = out_dir / cam / "depth" / demo_key / f"{stem}.png"
                    #     mask_path = out_dir / cam / "mask" / demo_key / f"{stem}.png"
                    #     meta_path = out_dir / cam / "meta" / demo_key / f"{stem}.json"

                    #     # Mask (blank or rough guess)
                    #     if args.auto_mask:
                    #         # Use depth in mm for the depth cue
                    #         mask = try_simple_mask(bgr, depth_u16)
                    #     else:
                    #         mask = np.zeros(bgr.shape[:2], dtype=np.uint8)

                    #     # Write images
                    #     cv2.imwrite(str(rgb_path), bgr)          # 8-bit color
                    #     cv2.imwrite(str(depth_path), depth_u16)  # 16-bit depth
                    #     cv2.imwrite(str(mask_path), mask)        # 8-bit binary

                    #     # Compose per-frame metadata
                    #     cam_info = camera_meta.get(cam, {})
                    #     meta_obj = {
                    #         "demo_id": demo_key,
                    #         "frame_idx": int(t),
                    #         "camera": cam,
                    #         # [W, H]
                    #         "image_size": [int(bgr.shape[1]), int(bgr.shape[0])],
                    #         "intrinsic": cam_info.get("intrinsic", None),
                    #         "extrinsic": cam_info.get("extrinsic", None),

                    #         # Robot state
                    #         "robot0_joint_pos": joints,
                    #         "robot0_gripper_qpos": grip,
                    #         "robot0_eef_pos": epos,
                    #         "robot0_eef_quat_xyzw": equat,  # verify ordering in your downstream code

                    #         # File paths (relative to out_dir)
                    #         "files": {
                    #             "rgb": str(rgb_path.relative_to(out_dir)),
                    #             "depth_mm_u16": str(depth_path.relative_to(out_dir)),
                    #             "mask_u8": str(mask_path.relative_to(out_dir)),
                    #         }
                    #     }
                    #     write_json(meta_path, meta_obj)

                    #     # Manifest row
                    #     writer.writerow([
                    #         demo_key, int(t), cam,
                    #         str(rgb_path.relative_to(out_dir)),
                    #         str(depth_path.relative_to(out_dir)),
                    #         str(mask_path.relative_to(out_dir)),
                    #         str(meta_path.relative_to(out_dir)),
                    #     ])
                    for cam in available_cams:
                        rgb_key = f"{cam}_image"
                        depth_key = f"{cam}_depth"

                        # Read RGB
                        rgb = obs[rgb_key][t]
                        bgr = to_bgr(rgb)

                        # Handle depth if present, otherwise create dummy
                        if cam_has_depth.get(cam, False):
                            depth = obs[depth_key][t]
                            if depth.dtype != np.uint16:
                                depth_u16 = depth_to_uint16_mm(depth)
                            else:
                                depth_u16 = depth
                            depth_for_mask = depth_u16
                        else:
                            # No depth in this HDF5: use an all-zero depth image
                            depth_u16 = np.zeros(bgr.shape[:2], dtype=np.uint16)
                            depth_for_mask = None  # mask will ignore depth

                        stem = f"frame_{t:06d}"
                        rgb_path = out_dir / cam / "rgb" / demo_key / f"{stem}.png"
                        depth_path = out_dir / cam / "depth" / demo_key / f"{stem}.png"
                        mask_path = out_dir / cam / "mask" / demo_key / f"{stem}.png"
                        meta_path = out_dir / cam / "meta" / demo_key / f"{stem}.json"

                        # Mask (blank or rough guess)
                        if args.auto_mask:
                            mask = try_simple_mask(bgr, depth_for_mask)
                        else:
                            mask = np.zeros(bgr.shape[:2], dtype=np.uint8)

                        # Write images
                        cv2.imwrite(str(rgb_path), bgr)
                        cv2.imwrite(str(depth_path), depth_u16)
                        cv2.imwrite(str(mask_path), mask)

                        # Compose per-frame metadata
                        cam_info = camera_meta.get(cam, {})
                        meta_obj = {
                            "demo_id": demo_key,
                            "frame_idx": int(t),
                            "camera": cam,
                            "image_size": [int(bgr.shape[1]), int(bgr.shape[0])],
                            "intrinsic": cam_info.get("intrinsic", None),
                            "extrinsic": cam_info.get("extrinsic", None),

                            "robot0_joint_pos": joints,
                            "robot0_gripper_qpos": grip,
                            "robot0_eef_pos": epos,
                            "robot0_eef_quat_xyzw": equat,

                            "files": {
                                "rgb": str(rgb_path.relative_to(out_dir)),
                                "depth_mm_u16": str(depth_path.relative_to(out_dir)),
                                "mask_u8": str(mask_path.relative_to(out_dir)),
                            }
                        }
                        write_json(meta_path, meta_obj)

                        writer.writerow([
                            demo_key, int(t), cam,
                            str(rgb_path.relative_to(out_dir)),
                            str(depth_path.relative_to(out_dir)),
                            str(mask_path.relative_to(out_dir)),
                            str(meta_path.relative_to(out_dir)),
                        ])

    print(f"\nDone. Export written to: {out_dir}")
    print(f"- Manifest: {manifest_path}")
    if args.copy_cam_meta:
        print(f"- Camera meta: {out_dir/'meta'/'cameras.json'}")
    print("These files are ready for cv2.inpaint (use the RGB + mask).")


if __name__ == "__main__":
    main()
