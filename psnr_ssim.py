from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import numpy as np
import os
import argparse
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dir0', type=str)
parser.add_argument('--dir1', type=str)
args = parser.parse_args()

true_image_dir = args.dir0
test_image_dir = args.dir1

true_img_names = sorted(os.listdir(true_image_dir))
test_img_names = sorted(os.listdir(test_image_dir), key=lambda x: int(x.split('.')[0]))
# print(test_img_names)

psnr = []
ssim = []
for i in tqdm(range(len(test_img_names))):
    true_img_name = true_img_names[i]
    test_img_name = test_img_names[i]
    true_img = os.path.join(true_image_dir, true_img_name)
    test_img = os.path.join(test_image_dir, test_img_name)
    
    true_img = Image.open(true_img)
    test_img = Image.open(test_img)
    size = test_img.size
    true_img = true_img.resize(size)

    true_img = np.array(true_img)
    test_img = np.array(test_img)

    psnr.append(PSNR(true_img, test_img))
    ssim.append(SSIM(true_img, test_img, multichannel=True, channel_axis=2))
    # print(psnr, ssim)

print(np.mean(psnr), np.mean(ssim))
