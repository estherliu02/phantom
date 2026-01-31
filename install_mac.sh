eval "$(conda shell.bash hook)"
# ######################## Phantom Env ###############################
conda create -n phantom python=3.10 -y
conda activate phantom
# conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0 -y

# Install phantom-robosuite
cd submodules/phantom-robosuite
pip install -e .
cd ../..

# Install phantom-robomimic
cd submodules/phantom-robomimic
pip install -e . --no-deps
cd ../..

# robomimic deps manually
pip install h5py psutil tqdm tensorboard tensorboardX imageio imageio-ffmpeg matplotlib diffusers

# Install additional packages
pip install joblib mediapy open3d pandas
pip install transformers==4.42.4
pip install PyOpenGL==3.1.4
pip install Rtree
pip install git+https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes.git
pip install protobuf==3.20.0
pip install hydra-core==1.3.2
pip install omegaconf==2.3.0
pip install h5py

# Install phantom 
pip install -e .