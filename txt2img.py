import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from utils import get_fr_model
from torchvision.utils import save_image
from torchvision import transforms
from dataset import base_dataset
from torch.utils.data import Subset
from torch import nn
from utils import asr_calculation


# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=47,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="pretrained_model/512-base-ema.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument('--model', type=str, default='IR152')
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--num', type=int, default='1000')
    parser.add_argument('--t', type=int, default=999)
    parser.add_argument('--save', type=str, default='res')
    parser.add_argument('--s', type=int, default=300)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    batch_size = 2

    prompt = opt.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    transform = transforms.Compose([transforms.Resize((512, 512)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    if opt.dataset == 'celeba':
        dataset = base_dataset(dir='./celeba-hq_sample', transform=transform)
    elif opt.dataset == 'ffhq':
        dataset = base_dataset(dir='./ffhq_sample', transform=transform)
    dataset = base_dataset(dir='./select', transform=transform)
    # dataset = Subset(dataset, [x for x in range(opt.num)])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    attack_model_names = [opt.model]
    attack_model_dict = {'IR152': get_fr_model('IR152'), 'IRSE50': get_fr_model('IRSE50'), 
                         'FaceNet': get_fr_model('FaceNet'), 'MobileFace': get_fr_model('MobileFace')}
    cos_sim_scores_dict = {opt.model: []}

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    for attack_model_name in attack_model_names:
        attack_model = attack_model_dict[attack_model_name]
        classifier = {k: v for k, v in attack_model_dict.items() if k != attack_model_name}
        resize = nn.AdaptiveAvgPool2d((112, 112)) if attack_model_name != 'FaceNet' else nn.AdaptiveAvgPool2d((160, 160))
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for i, (image, tgt_image) in enumerate(dataloader):
                        tgt_image = tgt_image.to(device)
                        B = image.shape[0]
                        prompts = data[0]
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                                prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                        _t = opt.t  # 0-999
                        z = model.get_first_stage_encoding(model.encode_first_stage(image.to(device)))
                        t = torch.tensor([_t] * batch_size, device=device)
                        z_t = model.q_sample(x_start=z, t=t)
                        samples, _ = sampler.sample(S=45,
                                                    conditioning=c,
                                                    batch_size=batch_size,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta,
                                                    x_T=z_t,
                                                    x_target=tgt_image,
                                                    _t=_t + 1,
                                                    classifier=classifier,
                                                    classifier_scale=300)

                        x_samples = model.decode_first_stage(samples)
                        result = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        os.makedirs(os.path.join(opt.save, 'img'), exist_ok=True)
                        os.makedirs(os.path.join(opt.save, 'msk'), exist_ok=True)
                        for x in range(result.shape[0]):
                            save_image(result[x], os.path.join(opt.save, 'img', f'{i * batch_size + x}.png'))

                        feature1 = attack_model(resize(result)).reshape(B, -1)
                        feature2 = attack_model(resize(tgt_image)).reshape(B, -1)
                        
                        from torch.nn import functional as F
                        score = F.cosine_similarity(feature1, feature2)
                        print(score)
                        cos_sim_scores_dict[attack_model_name] += score.tolist()
    asr_calculation(cos_sim_scores_dict)

if __name__ == "__main__":
    main()