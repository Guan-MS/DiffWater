import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default=r'experiments/UIEB_230429_090146/results')
    args = parser.parse_args()

    gt_names = list(glob.glob('{}/*_target.png'.format(args.path)))
    input_names = list(glob.glob('{}/*_input.png'.format(args.path)))

    gt_names.sort()
    input_names.sort()

    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    for gname, iname in zip(gt_names, input_names):
        idx += 1
        gidx = gname.rsplit("_target")[0]
        iidx = iname.rsplit("_input")[0]
        assert gidx == iidx, 'Image gidx:{gidx}!=iidx:{iidx}'.format(gidx=gidx, iidx=iidx)

        gt_img = np.array(Image.open(gname))
        input_img = np.array(Image.open(iname))
        psnr = Metrics.calculate_psnr(input_img, gt_img)
        ssim = Metrics.calculate_ssim(input_img, gt_img)

        avg_psnr += psnr
        avg_ssim += ssim

        if idx % 1 == 0:
            print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}'.format(idx, psnr, ssim))

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx


    # log

    print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    print('# Validation # SSIM: {:.4e}'.format(avg_ssim))
