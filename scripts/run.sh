
## scripts/run_phantom_pipeline.sh

#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_phantom_pipeline.sh <DATA_NAME> [VIEW]
#
# Example:
#   bash scripts/run_phantom_pipeline.sh stack_d1_abs spaceview

DATA_NAME="${1:-}"
VIEW="${2:-spaceview}"

if [[ -z "${DATA_NAME}" ]]; then
  echo "ERROR: DATA_NAME is required."
  echo "Usage: bash scripts/run_phantom_pipeline.sh <DATA_NAME> [VIEW]"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPLOAD_DIR="${REPO_ROOT}/uploads/${DATA_NAME}"
HDF5_IN="${UPLOAD_DIR}/${DATA_NAME}.hdf5"
CAM_META="${UPLOAD_DIR}/camera_meta.json"

RAW_OUT="${REPO_ROOT}/raw_data/${DATA_NAME}"
PROCESSED_OUT="${REPO_ROOT}/data/processed_data"

INTRINSICS_OUT="${UPLOAD_DIR}/camera_intrinsics_${VIEW}.json"
EXTRINSICS_OUT="${UPLOAD_DIR}/camera_extrinsics_${VIEW}.json"
HDF5_OUT="${UPLOAD_DIR}/${DATA_NAME}_overlay.hdf5"

# ----- sanity checks -----
if [[ ! -f "${HDF5_IN}" ]]; then
  echo "ERROR: Missing input HDF5: ${HDF5_IN}"
  exit 1
fi
if [[ ! -f "${CAM_META}" ]]; then
  echo "ERROR: Missing camera_meta.json: ${CAM_META}"
  exit 1
fi

echo "== Phantom pipeline =="
echo "Repo root : ${REPO_ROOT}"
echo "Data name : ${DATA_NAME}"
echo "View      : ${VIEW}"
echo "Uploads   : ${UPLOAD_DIR}"
echo

# ----- step 1: unpack hdf5 -----
echo "== [1/5] Unpack HDF5 -> raw_data/"
python "${REPO_ROOT}/phantom/scripts/read_hdf5.py" \
  --hdf5 "${HDF5_IN}" \
  --camera_meta "${CAM_META}" \
  --out_dir "${RAW_OUT}" \
  --cameras "frontview,agentview,birdview,sideview,sideview2,backview,spaceview" \
  --stride 1 \
  --copy_cam_meta

# ----- step 2: prepare phantom data (uses chosen view) -----
echo "== [2/5] Prepare Phantom folder structure -> data/processed_data/"
RGB_DIR="${RAW_OUT}/${VIEW}/rgb"
MASK_DIR="${RAW_OUT}/${VIEW}/mask"
DEPTH_DIR="${RAW_OUT}/${VIEW}/depth"
META_DIR="${RAW_OUT}/${VIEW}/meta"

for p in "${RGB_DIR}" "${MASK_DIR}" "${DEPTH_DIR}" "${META_DIR}"; do
  if [[ ! -d "${p}" ]]; then
    echo "ERROR: Missing expected directory for view '${VIEW}': ${p}"
    echo "Check that --cameras in read_hdf5.py included '${VIEW}'."
    exit 1
  fi
done

mkdir -p "${PROCESSED_OUT}"

python "${REPO_ROOT}/phantom/scripts/prepare_phantom_data.py" \
  --rgb_base_dir "${RGB_DIR}" \
  --mask_base_dir "${MASK_DIR}" \
  --depth_base_dir "${DEPTH_DIR}" \
  --meta_base_dir "${META_DIR}" \
  --out_root "${PROCESSED_OUT}" \
  --camera_meta "${CAM_META}" \
  --fps 15 \
  --hand right \
  --bimanual_setup single_arm \
  --depth_mm_to_m

# ----- step 3: prepare camera intrinsics/extrinsics for chosen view -----
echo "== [3/5] Prepare camera intrinsics/extrinsics -> uploads/<DATA_NAME>/"
python "${REPO_ROOT}/phantom/scripts/prepare_camera_data.py" \
  --input "${CAM_META}" \
  --view "${VIEW}"

if [[ ! -f "${INTRINSICS_OUT}" ]]; then
  echo "ERROR: Expected intrinsics not found: ${INTRINSICS_OUT}"
  exit 1
fi
if [[ ! -f "${EXTRINSICS_OUT}" ]]; then
  echo "ERROR: Expected extrinsics not found: ${EXTRINSICS_OUT}"
  exit 1
fi

# ----- step 4: phantom inpaint -----
echo "== [4/5] Run robot_inpaint"
python "${REPO_ROOT}/phantom/phantom/process_data.py" \
  data_root_dir="${PROCESSED_OUT}" \
  processed_data_root_dir="${PROCESSED_OUT}" \
  mode=robot_inpaint \
  robot=Panda \
  gripper=Panda \
  bimanual_setup=single_arm \
  target_hand=right \
  camera_intrinsics="${INTRINSICS_OUT}" \
  camera_extrinsics="${EXTRINSICS_OUT}" \
  input_resolution=128 \
  output_resolution=128 \
  depth_for_overlay=False \
  render=True \
  n_processes=1 \
  square=True

# ----- step 5: convert back to hdf5 -----
echo "== [5/5] Save overlay HDF5"
python "${REPO_ROOT}/phantom/scripts/save_hdf5.py" \
  --input "${HDF5_IN}" \
  --output "${HDF5_OUT}"

echo
echo "✅ Done."
echo "Overlay HDF5: ${HDF5_OUT}"
echo "Intrinsics  : ${INTRINSICS_OUT}"
echo "Extrinsics  : ${EXTRINSICS_OUT}"
echo "Processed   : ${PROCESSED_OUT}"