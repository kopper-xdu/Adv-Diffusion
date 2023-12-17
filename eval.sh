# the --num option value need to be set less than or equal to your src images numbers.
python eval.py --model IR152 --dataset celeba --t 999 --save res --num 5
python -m pytorch_fid res/img celeba-hq_sample/src
python psnr_ssim.py --dir1 res/img --dir2 celeba-hq_sample/src