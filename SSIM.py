import torch  
import torch.nn.functional as F 
import numpy as np
import math
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow
import time
import gc

def gaussian(window_size, sigma):
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)     
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

load_images = lambda x: np.asarray(np.resize(x, (len(x), 480, 640, 3)))
tensorify = lambda x: torch.Tensor(x.transpose((2, 0, 1))).unsqueeze(0).float().div(255.0)

class SSIM:
    def __init__(self, img1_, img2_):
        temp = load_images(img1_)
        self.img1 = []
        self.img2 = []
        for i in range(len(temp)):
           self.img1.append(tensorify(temp[i]))

        temp2 = load_images(img2_)
        for j in range(len(temp2)):
           self.img2.append(tensorify(temp2[j]))
                
    def calc_score(self, window_size=11, window=None):
        pad = window_size // 2
        _, channels, height, width = self.img1[0].size()
        if window is None: 
            real_size = min(window_size, height, width) # window should be atleast 11x11 
            window = create_window(real_size, channel=channels).to(self.img1[0].device)
        res = []
        C1 = (0.01 ) ** 2 
        C2 = (0.03 ) ** 2

        mu1s = []
        sigma1s = []
        
        start = time.time()

        for i in range(len(self.img1)):
            mu1s.append(F.conv2d(self.img1[i], window, padding=pad, groups=channels))
            sigma1s.append(F.conv2d(self.img1[i] * self.img1[i], window, padding=pad, groups=channels))
        
        mu2s = []        
        sigmas2s = [] 
        for j in range(len(self.img2)):
            mu2s.append(F.conv2d(self.img2[j], window, padding=pad, groups=channels))
            sigmas2s.append(F.conv2d(self.img2[j] * self.img2[j], window, padding=pad, groups=channels))

        end = time.time()
        print(end - start , "seconds (first)")

        for i in range(len(self.img1)):
            mu1 = mu1s[i]
            mu1_sq = mu1 ** 2
            sigma1_sq = sigma1s[i] - mu1_sq
            start2 = time.time()
            for j in range(len(self.img2)):
                mu2 = mu2s[j]
                mu2_sq = mu2 ** 2 
                mu12 = mu1 * mu2
                sigma2_sq = sigmas2s[j] - mu2_sq
                sigma12 =  F.conv2d(self.img1[i] * self.img2[j], window, padding=pad, groups=channels) - mu12
                contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
                contrast_metric = torch.mean(contrast_metric)
                numerator1 = 2 * mu12 + C1
                numerator2 = 2 * sigma12 + C2
                denominator1 = mu1_sq + mu2_sq + C1 
                denominator2 = sigma1_sq + sigma2_sq + C2
                ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)
                ret = ssim_score.mean()
                res.append([ret.numpy(), i, j])
                # print('**********************************************')
                # for k in range(len(res)):
                #     for l in range(len(res[0])):
                #         print(res[k][l], end = '\t')
                #     print('\n--------------------------------\n')
                # print('**********************************************')
            end2 = time.time()
            # print(end2 - start2, "seconds (second)")
        return res


# ssim = SSIM(_img1, _img2)
# score = ssim.calc_score()
# print(score)