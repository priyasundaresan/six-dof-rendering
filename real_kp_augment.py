import os
import xml.etree.cElementTree as ET
from xml.dom import minidom
import random
import colorsys
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

sometimes = lambda aug: iaa.Sometimes(0.3, aug)

AUGS = [ 
    iaa.LinearContrast((0.95, 1.05), per_channel=0.25), 
    iaa.Add((-10, 20), per_channel=False),
    iaa.GammaContrast((0.85, 1.15)),
    iaa.GaussianBlur(sigma=(1,1.75)),
    #iaa.Cutout(fill_mode="constant", cval=(0, 50), nb_iterations=(1, 3), size=0.25, squared=False, fill_per_channel=0.5),
    iaa.MultiplySaturation((0.85, 1.15)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
    sometimes(
        iaa.CropAndPad(
        percent=(0.1, 0.2),
        pad_mode=["constant"],
        pad_cval=(0, 20)
    ))
    ]

seq = iaa.Sequential(AUGS, random_order=True) 

def augment(img, annots, output_dir_img, output_annot_dir, new_idx, show=False):
    img_aug = seq(image=img)
    if show:
        cv2.imshow("img", img_aug)
        cv2.waitKey(0)

    cv2.imwrite(os.path.join(output_dir_img, "%05d.jpg"%new_idx), img_aug)
    np.save(os.path.join(output_annot_dir, '%05d.npy'%new_idx), annots)
    

if __name__ == '__main__':
    img_dir = 'images'
    annots_dir = 'annots'
    output_dir_img = img_dir
    output_annot_dir = annots_dir
    idx = len(os.listdir(img_dir))
    orig_len = len(os.listdir(img_dir))
    num_augs_per = 2
    for i in range(orig_len):
        print(i, orig_len)
        img = cv2.imread(os.path.join(img_dir, '%05d.jpg'%i))
        annots = np.load(os.path.join(annots_dir, '%05d.npy'%i), allow_pickle=True)
        for _ in range(num_augs_per):
            augment(img, annots, output_dir_img, output_annot_dir, idx+i, show=False)
            idx += 1
        idx -= 1
