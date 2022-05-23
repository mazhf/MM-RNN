import cv2
import numpy
import sys
sys.path.append("..")
from config import cfg
from concurrent.futures import ThreadPoolExecutor, wait


thread_pool = ThreadPoolExecutor(max_workers=16)
H = cfg.height
W = cfg.width


def cv2_read_img(path, read_storage, gray):
    if gray:
        read_storage[:] = cv2.imread(path, 0)
    else:
        read_storage[:] = cv2.imread(path)  # 默认flag=1, 彩色, BGR


def quick_read_frames(path_list, gray=True):
    # Multi-thread Frame Loader
    img_num = len(path_list)
    if gray:
        read_storage = numpy.empty((img_num, H, W), dtype=numpy.uint8)  # S H W
    else:
        read_storage = numpy.empty((img_num, H, W, 3), dtype=numpy.uint8)  # S H W 3
    future_objs = []
    for i in range(img_num):
        obj = thread_pool.submit(cv2_read_img, path_list[i], read_storage[i], gray)
        future_objs.append(obj)
    wait(future_objs)
    if gray:
        read_storage = read_storage.reshape((img_num, 1, H, W))  # S 1 H W
    else:
        read_storage = read_storage.transpose((0, 3, 1, 2))  # S 3 H W

    return read_storage[:, ::-1, ...].copy()  # 彩色为RGB
