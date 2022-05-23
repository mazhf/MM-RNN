import cv2
import cmaps
import matplotlib.pyplot as plt
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor


# single
# # colored other 8 ele
# cmap_color = cmaps.BkBlAqGrYeOrReViWh200
# dd = {'max': 486.85142463618564, 'min': -142.45055216479705}
# ff = {'max': 37.54846851792292, 'min': -0.8565767029418478}
# hu = {'max': 120.6067321927205, 'min': 1.4872401515288831}
# td = {'max': 313.9070046005785, 'min': 221.48327260054808}
# psl = {'max': 104626.48964329725, 'min': 95867.01941430081}
# t = {'max': 311.937838308806, 'min': 241.35155875346243}

# elevation_read_pth = r"C:\Users\Administrator\Desktop\MM-RNN\colored_elevation\NW_elevation.png"
# elevation = cv2.imread(elevation_read_pth, flags=0)
# elevation = cv2.resize(elevation, (128, 128))


# read_path = r"C:\Users\Administrator\Desktop\MM-RNN\colored_elevation\demo_new\0.npy"
# save_path = r"C:\Users\Administrator\Desktop\MM-RNN\colored_elevation\elements_new"
# data = np.load(read_path)
# data[0, ...] = data[0, ...] / 255.0
# data[1, ...] = (data[1, ...] - dd['min']) / (dd['max'] - dd['min'])
# data[2, ...] = (data[2, ...] - ff['min']) / (ff['max'] - ff['min'])
# data[3, ...] = (data[3, ...] - hu['min']) / (hu['max'] - hu['min'])
# data[4, ...] = (data[4, ...] - td['min']) / (td['max'] - td['min'])
# data[5, ...] = (data[5, ...] - t['min']) / (t['max'] - t['min'])
# data[6, ...] = (data[6, ...] - psl['min']) / (psl['max'] - psl['min'])
# data = data * 255
# data = data.astype(np.uint8)


# ele = ['radar', 'dd', 'ff', 'hu', 'td', 't', 'psl', 'elevation']

# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         img = data[i, j, 0, ...]
#         img_save_pth = os.path.join(save_path, str(j))
#         if not os.path.exists(img_save_pth):
#             os.makedirs(img_save_pth)
#         plt.imsave(os.path.join(img_save_pth, ele[i] + '.png'), img, cmap=cmap_color, vmin=0, vmax=255)
#         plt.imsave(os.path.join(img_save_pth, 'elevation.png'), elevation, cmap=cmap_color, vmin=0, vmax=255)


# total
cmap_color = cmaps.BkBlAqGrYeOrReViWh200
# dd = {'max': 486.85142463618564, 'min': -142.45055216479705}
dd = {'max': 360, 'min': 0}
ff = {'max': 30, 'min': 0}
hu = {'max': 100, 'min': 0}
# td = {'max': 313.9070046005785, 'min': 221.48327260054808}
# psl = {'max': 104626.48964329725, 'min': 95867.01941430081}
# t = {'max': 311.937838308806, 'min': 241.35155875346243}
td = {'max': 300, 'min': 230}
psl = {'max': 100000, 'min': 97000}
t = {'max': 300, 'min': 250}

elevation_read_pth = r'/home/mazhf/Precipitation-Nowcasting/dataset/MeteoNet/NW_elevation.png'
elevation = cv2.imread(elevation_read_pth, flags=0)
elevation = cv2.resize(elevation, (128, 128))

total_pth = r'/home/mazhf/Precipitation-Nowcasting/dataset/MeteoNet/resize_128/train'
total_save_pth = r'/mnt/d2/mm-rnn-demo'
import shutil
if os.path.exists(total_save_pth):
    shutil.rmtree(total_save_pth)

file_lis = os.listdir(total_pth)
count = 0
thread_pool = ThreadPoolExecutor(max_workers=16)


def save(file_lis_f, count):
    read_path = os.path.join(total_pth, file_lis_f)
    save_path = os.path.join(total_save_pth, file_lis_f.split('.')[0])
    # data = np.load(read_path, allow_pickle=True)
    data = np.load(read_path)
    data[0, ...] = data[0, ...] / 255.0
    data[1, ...] = (data[1, ...] - dd['min']) / (dd['max'] - dd['min'])
    data[2, ...] = (data[2, ...] - ff['min']) / (ff['max'] - ff['min'])
    data[3, ...] = (data[3, ...] - hu['min']) / (hu['max'] - hu['min'])
    data[4, ...] = (data[4, ...] - td['min']) / (td['max'] - td['min'])
    data[5, ...] = (data[5, ...] - t['min']) / (t['max'] - t['min'])
    data[6, ...] = (data[6, ...] - psl['min']) / (psl['max'] - psl['min'])
    data = data * 255
    data = data.astype(np.uint8)
    ele = ['radar', 'dd', 'ff', 'hu', 'td', 't', 'psl', 'elevation']
    for e in range(data.shape[0]):
        for s in range(data.shape[1]):
            img = data[e, s, 0, ...]
            img_save_pth = os.path.join(save_path, str(s))
            if not os.path.exists(img_save_pth):
                os.makedirs(img_save_pth)
            plt.imsave(os.path.join(img_save_pth, ele[e] + '.png'), img, cmap=cmap_color, vmin=0, vmax=255)
            plt.imsave(os.path.join(img_save_pth, 'elevation.png'), elevation, cmap=cmap_color, vmin=0, vmax=255)
    print(count)


for f in range(len(file_lis)):
    count += 1
    future = thread_pool.submit(save, file_lis[f], count)




