# Echo2DepthDiff: Depth Estimation from Echoes via Pretrained Geometry-Aligned Waveform Diffusion

This repository provides the official implementation of **Echo2DepthDiff**, a conditional generative framework for echo-based depth estimation. The model combines spectrograms and pretrained geometry-aware waveform embeddings to produce high-fidelity depth maps in acoustically accessible but visually challenging scenes.

## 📦 Features

- Conditional diffusion model tailored for acoustic signals
- Waveform-aligned contrastive embedding (WACE) for structural supervision
- Hybrid loss combining latent-space denoising and pixel-level reconstruction
- Supports ensemble inference and adjustable denoising steps
- Evaluated on Replica and Matterport3D datasets

---

## 📁 Datasets

### Replica  
Download from the [VisualEchoes](https://github.com/facebookresearch/VisualEchoes) repository.


### Matterport3D  
Requires Matterport3D RGB-D + acoustic simulation data (preprocessed separately). Follow instructions at [Matterport3D](https://niessner.github.io/Matterport/).

## Training
python train.py --config config/train.yaml --no_wandb

## Infer
bash script/eval/infer.sh
