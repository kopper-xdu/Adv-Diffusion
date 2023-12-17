import torch
from torch import nn
from utils import setup_seed, get_fr_model, initialize_model, asr_calculation
import os
from FaceParsing.interface import FaceParsing
from dataset import base_dataset
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image
import argparse
from torch.utils.data import Subset

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 0
setup_seed(0)

@torch.no_grad()
def main(args):
    h = 512
    w = 512
    txt = ''
    ddim_steps = 45
    scale = 0
    classifier_scale = args.s
    batch_size = 2
    num_workers = 0
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    transform = transforms.Compose([transforms.Resize((512, 512)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    if args.dataset == 'celeba':
        dataset = base_dataset(dir='./celeba-hq_sample', transform=transform)
    elif args.dataset == 'ffhq':
        dataset = base_dataset(dir='./ffhq_sample', transform=transform)
    # dataset = base_dataset(dir='./s', transform=transform)
    dataset = Subset(dataset, [x for x in range(args.num)])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    sampler = initialize_model('configs/stable-diffusion/v2-inpainting-inference.yaml', 
                               'pretrained_model/512-inpainting-ema.ckpt')
    model = sampler.model


    # prng = np.random.RandomState(seed)
    # start_code = prng.randn(batch_size, 4, h // 8, w // 8)
    # start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    # attack_model_names = ['IR152', 'IRSE50', 'FaceNet', 'MobileFace']
    # attack_model_names = ['IRSE50']
    attack_model_names = [args.model]
    attack_model_dict = {'IR152': get_fr_model('IR152'), 'IRSE50': get_fr_model('IRSE50'), 
                         'FaceNet': get_fr_model('FaceNet'), 'MobileFace': get_fr_model('MobileFace')}
    # attack_model_resize_dict = {'IR152': 112, 'IRSE50': 112, 'FaceNet': 160, 'MobileFace': 112}
    # cos_sim_scores_dict = {'IR152': [], 'IRSE50': [], 'FaceNet': [], 'MobileFace': []}
    # cos_sim_scores_dict = {'IRSE50': []}
    cos_sim_scores_dict = {args.model: []}
    
    for attack_model_name in attack_model_names:
        attack_model = attack_model_dict[attack_model_name]
        classifier = {k: v for k, v in attack_model_dict.items() if k != attack_model_name}
        resize = nn.AdaptiveAvgPool2d((112, 112)) if attack_model_name != 'FaceNet' else nn.AdaptiveAvgPool2d((160, 160))
        with torch.no_grad():
            for i, (image, tgt_image) in enumerate(dataloader):
                tgt_image = tgt_image.to(device)
                B = image.shape[0]
                
                face_parsing = FaceParsing()
                pred = face_parsing(image)

# label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 
# 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
                def get_mask(number):
                    return pred == number
                
                masks = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
                # skin = get_mask(1)
                # nose = get_mask(2)
                # eye_gla = get_mask(3)
                # l_eye = get_mask(4)
                # r_eye = get_mask(5)
                # l_brow = get_mask(6)
                # r_brow = get_mask(7)
                # mouth = get_mask(10)
                # u_lip = get_mask(11)
                # l_lip = get_mask(12)
                mask = None
                for x in masks:
                    if mask is not None:
                        mask |= get_mask(x)
                    else:
                        mask = get_mask(x)
                mask = (mask == 0).float().reshape(B, 1, h, w)

                masked_image = image * (mask < 0.5)

                batch = {
                    "image": image.to(device),
                    "txt": batch_size * [txt],
                    "mask": mask.to(device),
                    "masked_image": masked_image.to(device),
                }

                c = model.cond_stage_model.encode(batch["txt"])
                c_cat = list()
                for ck in model.concat_keys:
                    cc = batch[ck].float()
                    if ck != model.masked_image_key:
                        bchw = [batch_size, 4, h // 8, w // 8]
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                    else:
                        cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                    c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)

                # cond
                cond = {"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                uc_cross = model.get_unconditional_conditioning(batch_size, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                shape = [model.channels, h // 8, w // 8]
                
                # start code
                _t = args.t  # 0-999
                z = model.get_first_stage_encoding(model.encode_first_stage(image.to(device)))
                t = torch.tensor([_t] * batch_size, device=device)
                z_t = model.q_sample(x_start=z, t=t)

                samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    batch_size,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=z_t,
                    _t=_t + 1,
                    log_every_t=1,
                    classifier=classifier,
                    classifier_scale=classifier_scale,
                    x_target=tgt_image
                )

                x_samples_ddim = model.decode_first_stage(samples_cfg)
                result = torch.clamp(x_samples_ddim, min=-1, max=1)
                # print(len(intermediates['x_inter']))
                #for i, x_inter in enumerate(intermediates['x_inter']):
                  #  x_inter = torch.clamp(model.decode_first_stage(x_inter), min=-1, max=1)
                   # for y, x in enumerate(range(x_inter.shape[0])):
                    #    save_image((x_inter[x] + 1) / 2, f'inter/{i}_{y}.png')


                os.makedirs(os.path.join(args.save, 'img'), exist_ok=True)
                os.makedirs(os.path.join(args.save, 'msk'), exist_ok=True)
                print(i, batch_size)
                for x in range(result.shape[0]):
                    save_image((result[x] + 1) / 2, os.path.join(args.save, 'img', f'{i * batch_size + x}.png'))
                    save_image((masked_image[x] + 1) / 2, os.path.join(args.save, 'msk', f'{i * batch_size + x}_m.png'))

                # save_image((x_inter + 1) / 2, f'res/{i}_inter.png')
                
                # attack_model = attack_model_dict[attack_model_name]
                feature1 = attack_model(resize(result)).reshape(B, -1)
                feature2 = attack_model(resize(tgt_image)).reshape(B, -1)
                
                score = F.cosine_similarity(feature1, feature2)
                print(score)
                cos_sim_scores_dict[attack_model_name] += score.tolist()

                
                # feature3 = attack_model(resize(x_inter)).reshape(B, -1)
                # score = F.cosine_similarity(feature3, feature2)
                # print(score)

    asr_calculation(cos_sim_scores_dict)
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='IR152')
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--num', type=int, default='1000')
    parser.add_argument('--t', type=int, default=999)
    parser.add_argument('--save', type=str, default='res')
    parser.add_argument('--s', type=int, default=300)
    args = parser.parse_args()
    
    main(args)
