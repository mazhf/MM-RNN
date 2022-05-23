import cv2
import cmaps
import matplotlib.pyplot as plt
import os
import numpy as np

times = ['0', '6', '12', '18']

read_pth = r'/mnt/d2/mm-rnn-demo'
save_pth = r'/mnt/d2/mm-rnn-demo-concat'
import shutil
if os.path.exists(save_pth):
    shutil.rmtree(save_pth)

if not os.path.exists(save_pth):
    os.makedirs(save_pth)

file_lis = os.listdir(read_pth)
count = 0
for f in range(len(file_lis)):
    file_name = file_lis[f]
    file_pth = os.path.join(read_pth, file_name)
    seq_lis = os.listdir(file_pth)
    seq_lis.sort(key=lambda x: int(x))
    l0 = []
    for s in range(len(seq_lis)):
        seq_name = seq_lis[s]
        if seq_name in times:
            seq_pth = os.path.join(file_pth, seq_name)
            ele_lis = ['radar', 'dd', 'ff', 'hu', 'td', 't', 'psl', 'elevation']
            l1 = []  # 1 frame elements
            for e in range(len(ele_lis)):
                ele_name = ele_lis[e] + '.png'
                ele_pth = os.path.join(seq_pth, ele_name)
                ele = cv2.imread(ele_pth)
                l1.append(ele)  # 8 h w c
            l1 = np.concatenate(l1, axis=1)
            l0.append(l1)  # 1 demo  # 4 h 8w c
    l0 = np.concatenate(l0, axis=0)  # 4h 8w c
    cv2.imwrite(os.path.join(save_pth, file_name + '.png'), l0)
    count += 1
    print(count)