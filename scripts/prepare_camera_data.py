#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def mat3_from_4x4(T):
    R = [[float(x) for x in row[:3]] for row in T[:3]]
    t = [float(T[0][3]), float(T[1][3]), float(T[2][3])]
    return R, t


def v_fov_from_fy(fy, img_h):
    # radians
    return 2.0 * math.atan(float(img_h) / (2.0 * float(fy)))


def infer_image_size_from_cxy(cx, cy):
    # assumes principal point is (W/2, H/2)
    w = int(round(2.0 * float(cx)))
    h = int(round(2.0 * float(cy)))
    if w <= 0 or h <= 0:
        raise ValueError(f"Cannot infer image size from cx={cx}, cy={cy}")
    return w, h


def main():
    parser = argparse.ArgumentParser(
        description="Convert camera_meta.json into per-view camera_intrinsics_<view>.json and camera_extrinsics_<view>.json"
    )
    parser.add_argument("--input", type=Path, default=Path("camera_meta.json"),
                        help="Path to camera_meta.json")
    parser.add_argument("--view", required=True,
                        help="Camera view name to export (e.g., spaceview, frontview, agentview...)")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory (default: same directory as input)")
    args = parser.parse_args()

    in_path: Path = args.input
    out_dir: Path = args.out_dir if args.out_dir is not None else in_path.parent

    data = json.loads(in_path.read_text())
    if args.view not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(
            f"View '{args.view}' not found in {in_path}. Available: {available}")

    cam = data[args.view]

    # ---------- intrinsics ----------
    K = cam["intrinsic"]
    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])

    img_w, img_h = infer_image_size_from_cxy(cx, cy)
    v_fov = v_fov_from_fy(fy, img_h)

    intrinsics_out = {
        "left": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "v_fov": v_fov},
        "right": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "v_fov": v_fov},
    }

    # ---------- extrinsics ----------
    T = cam["extrinsic"]
    R, t = mat3_from_4x4(T)

    extrinsics_out = [
        {
            "camera_base_ori": R,
            "camera_base_pos": t,
        }
    ]

    # ---------- write ----------
    view = args.view
    out_intr = out_dir / f"camera_intrinsics_{view}.json"
    out_ext = out_dir / f"camera_extrinsics_{view}.json"

    out_intr.write_text(json.dumps(intrinsics_out, indent=4))
    out_ext.write_text(json.dumps(extrinsics_out, indent=4))

    print(f"Input: {in_path}")
    print(f"Exported view: {view}")
    print(f"Inferred image size: {img_w}x{img_h}")
    print(f"Wrote: {out_intr}")
    print(f"Wrote: {out_ext}")


if __name__ == "__main__":
    main()
