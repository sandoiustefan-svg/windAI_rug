#!/bin/bash
#SBATCH --time=03:00:00         
#SBATCH --partition=gpushort     
#SBATCH --gres=gpu:1            
#SBATCH --mem=96000

module spider CUDA
module spider cuDNN
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.8.0

export CUDA_HOME=$EBROOTCUDA
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME


module load cuDNN/8.7.0.84-CUDA-11.8.0


# Activate your virtual environment
source /home2/s5549329/windAI_rug/.venv/bin/activate

echo "Running on GPU node..."
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU')); print('CUDA version:', tf.sysconfig.get_build_info().get('cuda_version')); print('cuDNN version:', tf.sysconfig.get_build_info().get('cudnn_version'))"

# Run your Python script
python /home2/s5549329/windAI_rug/WindAi/deep_learning/preprocessing/preprocessing_forecast.py

# Deactivate environment
deactivate