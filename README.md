# Adv-Diffusion

## Install

- Build environment

```bash
conda env create -f env.yaml
```

- Download checkpoints

  Pretrained LDM can be found [here](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/512-inpainting-ema.ckpt).

  We use IR152, IRSE50, FaceNet and MobileFace model checkpoints that provided by [AMT-GAN]([CGCL-codes/AMT-GAN: The official implementation of our CVPR 2022 paper "Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer". (github.com)](https://github.com/CGCL-codes/AMT-GAN)). The google drive link they provide is [here]([assets.zip - Google 云端硬盘](https://drive.google.com/file/d/1Vuek5-YTZlYGoeoqyM5DlvnaXMeii4O8/view)).

  The face parsing model we use can be found [here](https://github.com/TracelessLe/FaceParsing.PyTorch). And the model checkpoints they provide is here: [[google drive]](https://drive.google.com/file/d/1neFVTZCWZcCeIoYA7V3i1Kk3DqaK4iei/view).

  You need to create a directory named as pretrained_model and put the checkpoints into it.

- Download datasets

  In our experiment we use FFHQ and CelebA-HQ datasets for evaluation. Because we do not own the datasets,  you need to download them yourself. And you can refer to [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) for CelebA-HQ download.

  After download you need to select several target images and source images and put them in the correct path as following.

  ```
  ├─celeba-hq_sample
  │  ├─src
  │  └─target
  ├─configs
  │  └─stable-diffusion
  │      └─intel
  ├─FaceParsing
  │  └─networks
  ├─ffhq_sample
  │  ├─src
  │  └─target
  ├─fr_model
  ├─ldm
  │  ├─data
  │  ├─models
  │  │  └─diffusion
  │  │      └─dpm_solver
  │  └─modules
  │      ├─diffusionmodules
  │      ├─distributions
  │      ├─encoders
  │      ├─image_degradation
  │      └─midas
  │          └─midas
  └─pretrained_model
  ```

  

## Usage

```bash
bash eval.sh
```