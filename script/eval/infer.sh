#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"/home/yinjun/project/Marigold-main/stable-diffusion-2"}
subfolder=${2:-"eval"}
base_data_dir=${3:-"/path/to/base/data/dir"}  # 这里指定 base_data_dir 的默认值

python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $base_data_dir \
    --denoise_steps 50 \
    --ensemble_size 20 \
    --processing_res 0 \
    --dataset_config config/dataset/data_nyu_test.yaml \
    --output_dir output/${subfolder}/nyu_test/prediction \