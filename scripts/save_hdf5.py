import os
import shutil
import h5py
import numpy as np
import mediapy as media
import cv2
import argparse

# ========= CONFIG =========
ROBOT = "Panda"
BIMANUAL = "single_arm"

# Define which observation keys to replace and their corresponding overlay roots
OVERLAY_CONFIGS = [
    {
        "obs_key": "spaceview_image",
        "phantom_root": "npz_new_data_space",
    },
    {
        "obs_key": "sideview_image",
        "phantom_root": "npz_new_data_side",
    }
]
# ==========================


def get_overlay_video_path(demo_idx: int, phantom_root: str) -> str:
    demo_dir = os.path.join(phantom_root, f"demo_{demo_idx}", "0")
    video_name = f"video_overlay_{ROBOT}_{BIMANUAL}.mkv"
    return os.path.join(demo_dir, video_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replace observation images in HDF5 with overlay videos"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input HDF5 file (must be clean original)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output HDF5 file with overlays"
    )
    return parser.parse_args()


def verify_dataset_readable(dset, demo_name: str, obs_key: str):
    """
    Read the entire dataset once to make sure gzip chunks are valid.
    Raises if anything goes wrong.
    """
    print(f"  [VERIFY] Reading back {demo_name}/{obs_key} for integrity...")
    try:
        _ = dset[...]   # loads all data; ensures all chunks decompress
    except Exception as e:
        raise RuntimeError(
            f"Verification failed for {demo_name}/{obs_key}: {type(e).__name__}: {e}"
        ) from e
    print(f"  [VERIFY] OK for {demo_name}/{obs_key}")


def main():
    args = parse_args()
    h5_in = args.input
    h5_out = args.output

    # ---- 0. Sanity: make sure input is really the original ----
    if not os.path.isfile(h5_in):
        raise FileNotFoundError(f"Original HDF5 not found: {h5_in}")

    # Remove old overlay & temp
    if os.path.exists(h5_out):
        print(f"[INFO] Removing existing overlay file: {h5_out}")
        os.remove(h5_out)

    tmp_path = h5_out + ".tmp"
    if os.path.exists(tmp_path):
        print(f"[WARN] Removing leftover temp file: {tmp_path}")
        os.remove(tmp_path)

    print(f"[INFO] Copying base file {h5_in} -> {tmp_path}")
    shutil.copy2(h5_in, tmp_path)

    # ---- 1. Open temp file and modify ----
    with h5py.File(tmp_path, "r+") as hf:
        data_group = hf["data"]
        demo_names = sorted(k for k in data_group.keys()
                            if k.startswith("demo_"))
        print(f"[INFO] Found {len(demo_names)} demos: "
              f"{demo_names[:5]}{' ...' if len(demo_names) > 5 else ''}")

        # ---- 2. Process each overlay configuration ----
        for config in OVERLAY_CONFIGS:
            obs_key = config["obs_key"]
            phantom_root = config["phantom_root"]

            print(f"\n{'='*60}")
            print(f"[CONFIG] Processing overlay for: {obs_key}")
            print(f"[CONFIG] Using phantom root: {phantom_root}")
            print(f"{'='*60}")

            # Get reference resolution for this view
            ref_H = ref_W = None
            for demo_name in demo_names:
                demo_group = data_group[demo_name]
                obs_group = demo_group["obs"]
                if obs_key in obs_group:
                    old_shape = obs_group[obs_key].shape  # [T, H, W, C]
                    _, ref_H, ref_W, _ = old_shape
                    print(
                        f"[INFO] Reference {obs_key} shape: {old_shape} (from {demo_name})")
                    break

            if ref_H is None:
                print(
                    f"[WARN] Could not find {obs_key} in any demo, skipping this view.")
                continue

            # ---- 3. Process demos one by one for this view ----
            for demo_name in demo_names:
                demo_idx = int(demo_name.split("_")[-1])
                print(
                    f"\n[DEMO] Processing {demo_name} (idx={demo_idx}) for {obs_key}")

                demo_group = data_group[demo_name]
                obs_group = demo_group["obs"]

                if obs_key not in obs_group:
                    print(f"  [WARN] {obs_key} not in obs, skipping.")
                    continue

                actions = demo_group["actions"]
                T_h5 = actions.shape[0]

                video_path = get_overlay_video_path(demo_idx, phantom_root)
                if not os.path.isfile(video_path):
                    print(
                        f"  [WARN] Overlay video not found, skipping: {video_path}")
                    continue

                # ---- 3.1 Read overlay video ----
                print(f"  [INFO] Reading video: {video_path}")
                vid = media.read_video(video_path)
                frames_overlay = np.array(vid)  # [T_vid, H_new, W_new, 3]

                if frames_overlay.ndim != 4 or frames_overlay.shape[-1] != 3:
                    raise RuntimeError(
                        f"Unexpected video shape {frames_overlay.shape} for {video_path}"
                    )

                # Ensure uint8
                if frames_overlay.dtype != np.uint8:
                    if frames_overlay.max() <= 1.5:
                        frames_overlay = (
                            frames_overlay * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        frames_overlay = frames_overlay.clip(
                            0, 255).astype(np.uint8)

                T_vid, H_new, W_new, _ = frames_overlay.shape
                print(f"  [INFO] Overlay frames shape: {frames_overlay.shape}")
                print(f"  [INFO] HDF5 actions length: {T_h5}")

                if T_vid != T_h5:
                    raise RuntimeError(
                        f"Length mismatch for {demo_name}: video {T_vid}, hdf5 {T_h5}"
                    )

                # ---- 3.2 Resize if needed ----
                if (H_new, W_new) != (ref_H, ref_W):
                    print(
                        f"  [INFO] Res mismatch: overlay ({H_new},{W_new}), "
                        f"expected ({ref_H},{ref_W}) – resizing."
                    )
                    resized = [
                        cv2.resize(f, (ref_W, ref_H),
                                   interpolation=cv2.INTER_AREA)
                        for f in frames_overlay
                    ]
                    frames_overlay = np.stack(resized, axis=0)
                    print(
                        f"  [INFO] Resized overlay frames to: {frames_overlay.shape}")

                assert frames_overlay.shape == (T_h5, ref_H, ref_W, 3), \
                    f"Final overlay shape mismatch for {demo_name}: {frames_overlay.shape}"

                # ---- 3.3 Replace dataset for this demo ----
                if obs_key in obs_group:
                    print(f"  [INFO] Deleting old {obs_key}")
                    del obs_group[obs_key]

                print(f"  [INFO] Creating new {obs_key} with compression=gzip")
                dset = obs_group.create_dataset(
                    obs_key,
                    data=frames_overlay,
                    compression="gzip",
                    dtype="uint8",
                )

                # ---- 3.4 Verify we can read back what we just wrote ----
                print(f"  [VERIFY] Reading back {demo_name}/{obs_key}...")
                try:
                    _ = dset[...]   # force all gzip chunks to decompress
                except Exception as e:
                    raise RuntimeError(
                        f"Verification failed for {demo_name}/{obs_key}: {type(e).__name__}: {e}"
                    )
                print(f"  [VERIFY] OK for {demo_name}/{obs_key}")

                # ---- 3.5 Verify we can read back the data we just wrote ----
                verify_dataset_readable(dset, demo_name, obs_key)

                # optional flush to be extra safe
                hf.flush()

        print(
            "\n[INFO] Finished writing and verifying all overlay datasets in temp file.")

    # ---- 3. Final integrity pass: check all demos quickly ----
    print("[INFO] Final integrity check over all demos in temp file...")
    with h5py.File(tmp_path, "r") as hf:
        data_group = hf["data"]
        demo_names = sorted(k for k in data_group.keys()
                            if k.startswith("demo_"))

        for config in OVERLAY_CONFIGS:
            obs_key = config["obs_key"]
            print(f"[INFO] Checking integrity for {obs_key}...")

            for demo_name in demo_names:
                demo = data_group[demo_name]
                obs = demo["obs"]
                if obs_key not in obs:
                    continue
                arr = obs[obs_key]
                # Just touch all data once
                _ = arr[...]

    print("[INFO] Integrity check passed for all demos and views.")

    # ---- 4. Atomic rename ----
    print(f"[INFO] Moving {tmp_path} -> {h5_out}")
    os.replace(tmp_path, h5_out)
    print("[INFO] Overlay HDF5 successfully rebuilt.")


if __name__ == "__main__":
    main()
