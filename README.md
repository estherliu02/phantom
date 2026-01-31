# Phantom data pipeline (Mac/Linux)

## 1) Install repo deps

From repo root:

```bash
cd phantom
./install_mac.sh   # macOS
# ./install_linux.sh   # Linux
cd ..
conda activate phantom
```

Then install runtime deps (skip if linux):

```bash
pip install torch torchvision torchaudio
```

### macOS note (egl_probe import issue)

If `egl_probe` causes issues on macOS, comment out lines **91–97** in:
`phantom/submodules/phantom-robomimic/robomimic/envs/env_robosuite.py`

## 2) Put inputs in uploads/<DATA_NAME>/
### Structure

```
uploads/                          # Same level as phantom repo
    <DATA_NAME>/
        <DATA_NAME>.hdf5
        camera_meta.json

phantom
```
Required files:

* `uploads/<DATA_NAME>/<DATA_NAME>.hdf5`
* `uploads/<DATA_NAME>/camera_meta.json`

Example:

```
uploads/stack_d1_abs/stack_d1_abs.hdf5
uploads/stack_d1_abs/camera_meta.json
```

## 3) Run the full pipeline (automated)

From repo root:

```bash
bash phantom/scripts/run.sh stack_d1_abs spaceview
```

Arguments:

* `DATA_NAME` (required): folder + hdf5 base name
* `VIEW` (optional, default: `spaceview`): camera view to process

Outputs:

* overlay HDF5: `uploads/<DATA_NAME>/<DATA_NAME>_overlay.hdf5`
* processed folder: `data/processed_data/`
* camera files: `uploads/<DATA_NAME>/camera_intrinsics.json`, `uploads/<DATA_NAME>/camera_extrinsics.json`
