# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:01:10 2022

@author: Administrator
"""
import cv2
import cmaps
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor


thread_pool = ThreadPoolExecutor(max_workers=16)
cmap_color = cmaps.BkBlAqGrYeOrReViWh200
base_pth = '/mnt/d1/paper_experiment/MM-RNN/save/MeteoNet'
modes = ['pred_img', 'truth_img', 'truth_pred_img']
models = os.listdir(base_pth)


def gray2rgb(model, demo, count):
    eles = os.listdir(os.path.join(base_pth, model, 'demo', demo))
    for ele in eles:
        for mode in modes:
            imgs = os.listdir(os.path.join(base_pth, model, 'demo', demo, ele, mode))
            data_save_pth = os.path.join(base_pth, model, 'demo', demo, ele, mode + '_rgb')
            if not os.path.exists(data_save_pth):
                os.makedirs(data_save_pth)
            for img in imgs:
                data = cv2.imread(os.path.join(base_pth, model, 'demo', demo, ele, mode, img), flags=0)
                plt.imsave(os.path.join(data_save_pth, img), data, cmap=cmap_color, vmin=0, vmax=255)
    print(model, count)


count = 0
for model in models:
    demos = os.listdir(os.path.join(base_pth, model, 'demo'))
    for demo in demos:
        count += 1
        future = thread_pool.submit(gray2rgb, model, demo, count)