# VoxHammer: Training-Free Precise and Coherent 3D Editing in Native 3D Space

## 🏠 [Project Page](https://huanngzh.github.io/VoxHammer-Page/) | [Paper](https://arxiv.org/abs/2508.19247) | [Edit3D-Bench](https://huggingface.co/datasets/huanngzh/Edit3D-Bench) | [Online Demo (Comming soon)](https://huggingface.co/spaces/VAST-AI/MIDI-3D)

![teaser](assets/doc/teaser.png)

**TL;DR:** A training-free 3D editing approach that performs precise and coherent editing in native 3D latent space instead of multi-view space.

## 🔥 Updates

* [2025-08-27] Release inference code and Edit3D-Bench.

## 📦 Installation

Please set up the Python environment following [TRELLIS](https://github.com/Microsoft/TRELLIS), or you can create by:

### Prerequisites

- **System**: Linux (Ubuntu 20.04/22.04 recommended)  
- **GPU**: NVIDIA GPU with at least **40 GB** memory (an A100 80GB is strongly recommended)  
- **Python**: 3.10  
- **Conda** (Miniconda/Anaconda) recommended  

### Installation Steps

1. Create and activate the environment
```bash
conda create -n hammer python=3.10 -y
conda activate hammer
```

2. Install requirements from `requirements.txt`
```bash
pip install -r requirements.txt
``` 

3. Install requirements from GitHub projects
```bash
pip install git+https://github.com/huanngzh/bpy-renderer.git
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
```

## 💡 Usage

### Step 1 - Add encoder to pipeline
After downloading "TRELLIS-image-large" model, we can find "sparse_structure_encoder" in ckpts/ss_enc_conv3d_16l8_fp16.safetensors and "slat_encoder" in ckpts/slat_enc_swin8_B_64l8_fp16.safetensors.

Before using these two encoders, we should add
{"sparse_structure_encoder": "ckpts/ss_enc_conv3d_16l8_fp16", "slat_encoder": "ckpts/slat_enc_swin8_B_64l8_fp16"}
in pipeline.json, then the pipeline can correctly call them.

### Step 2 — Create a 3D mask
Prepare a mesh that marks the editable region (e.g. `assets/example/mask.glb`).  

https://github.com/user-attachments/assets/4f1265f5-911b-4a44-9ca2-d36496e83cea

Inputs required (example):  
- `assets/example/model.glb` — original 3D asset  
- `assets/example/mask.glb` — mask mesh  

### Step 3 — Render RGB views & masks
```bash
python utils/render_rgb_and_mask.py \
--source_model assets/example/model.glb \
--mask_model assets/example/mask.glb \
--output_dir outputs
```

### Step 4 — Inpaint the rendered views
```bash
python utils/inpaint.py \
--image_path outputs/images/render_0002.png \
--mask_path outputs/images/mask_0002.png \
--output_dir outputs/images \
--prompt "A dog."
```

### Step 5 — Run inference
```bash
python inference.py \
--input_model assets/example/model.glb \
--mask_model assets/example/mask.glb \
--image_dir outputs/images \
--output_dir outputs
```

## Evaluation

Explore [Edit3D-Bench](./Edit3D-Bench/) for more details.

## License

This project is licensed under the [MIT License](https://github.com/Nelipot-Lee/VoxHammer/blob/main/LICENSE).  
However, please note that the code in **`trellis`** originates from the [TRELLIS](https://github.com/Microsoft/TRELLIS) project and remains subject to its original license terms.  
Users must comply with the licensing requirements of [TRELLIS](https://github.com/Microsoft/TRELLIS) when using or redistributing that portion of the code.

## Citation

```
@article{li2025voxhammer,
    title = {VoxHammer: Training-Free Precise and Coherent 3D Editing in Native 3D Space},
    author = {Li, Lin and Huang, Zehuan and Feng, Haoran and Zhuang, Gengxiong and Chen, Rui and Guo, Chunchao and Sheng, Lu},
    journal = {arXiv preprint arXiv:2508.19247},
    year = {2025}
}
```
