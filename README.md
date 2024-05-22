# Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model
Official code for [Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model](https://arxiv.org/abs/2312.11285v2) (AAAI 2024). 

Decheng Liu<sup>\*1</sup>, Xijun Wang<sup>\*1</sup>, Chunlei Peng<sup>†1</sup>, Nannan Wang<sup>1</sup>, Ruimin Hu<sup>1</sup>, Xinbo Gao<sup>2</sup>

<sup>1</sup>Xidian University, <sup>2</sup>Chongqing University of Posts and Telecommunications


## Abstract
Adversarial attacks involve adding perturbations to the source image to cause misclassification by the target model, which
demonstrates the potential of attacking face recognition models. Existing adversarial face image generation methods still
can’t achieve satisfactory performance because of low transferability and high detectability. In this paper, we propose
a unified framework Adv-Diffusion that can generate imperceptible adversarial identity perturbations in the latent
space but not the raw pixel space, which utilizes strong inpainting capabilities of the latent diffusion model to generate realistic adversarial images. Specifically, we propose the
identity-sensitive conditioned diffusion generative model to
generate semantic perturbations in the surroundings. The designed adaptive strength-based adversarial perturbation algorithm can ensure both attack transferability and stealthiness. Extensive qualitative and quantitative experiments
on the public FFHQ and CelebA-HQ datasets prove the
proposed method achieves superior performance compared
with the state-of-the-art methods without an extra generative model training process.

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

## Citation

```
@inproceedings{liu2024adv,
  title={Adv-diffusion: imperceptible adversarial face identity attack via latent diffusion model},
  author={Liu, Decheng and Wang, Xijun and Peng, Chunlei and Wang, Nannan and Hu, Ruimin and Gao, Xinbo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={4},
  pages={3585--3593},
  year={2024}
}
```

If you have any questions, please contact xijunwang00 [AT] gmail [DOT] com.
