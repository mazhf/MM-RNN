import shutil
from torch.utils.data import Dataset
import os
import cv2
import sys
sys.path.append("..")
from config import cfg
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
import torch


IN_LEN = cfg.in_len
OUT_LEN = cfg.out_len
LEN = IN_LEN + OUT_LEN
elements = ['dd', 'ff', 'hu', 'td', 't', 'psl']


def exist(radar_batch_names_list):
    for i in range(LEN):
        ele_name = radar_batch_names_list[i].split('.')[0]
        year = ele_name.split('-')[0]
        month = ele_name.split('-')[1]
        ele_pth = os.path.join('/mnt/d0/stations_after_time_interp/hu', year, month, ele_name + '.npy')
        if not os.path.exists(ele_pth):
            return False
    return True


def matching_and_save_batch_names():
    # radar data
    radar_img_names_list = os.listdir(os.path.join(cfg.DATASET_PATH))
    radar_img_names_list.sort(
        key=lambda x: int(datetime.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')))
    img_num = len(radar_img_names_list)
    valid_radar_batch_names_lis = []
    for i in range(img_num // LEN):
        radar_batch_names_list = radar_img_names_list[i * LEN: (i + 1) * LEN]
        radar_batch_names_list_date = [datetime.strptime(batch.split('.')[0], '%Y-%m-%d %H:%M:%S') for batch in
                                       radar_batch_names_list]
        # 如果雷达数据不连续，剔除
        if radar_batch_names_list_date[0] + timedelta(minutes=5 * (LEN - 1)) != radar_batch_names_list_date[-1]:
            continue
        # 如果雷达数据对应的气象元素数据不存在，剔除
        if not exist(radar_batch_names_list):
            continue
        valid_radar_batch_names_lis.append(radar_batch_names_list)
        print('matching: ', i)
    return valid_radar_batch_names_lis


class Data(Dataset):
    def __init__(self, mode=''):
        super().__init__()
        valid_radar_batch_names_lis = matching_and_save_batch_names()
        batch_nums = len(valid_radar_batch_names_lis)  # all 34 month
        a = int(batch_nums // 7)
        if mode == 'train':  # 6 : 1
            self.batch_names_lis = valid_radar_batch_names_lis[a:]  # 29 months
        else:
            self.batch_names_lis = valid_radar_batch_names_lis[:a]  # 5 months: 201601, 201602, 201603, 201604, 201605

    def __getitem__(self, index):
        batch_names = self.batch_names_lis[index]
        mix_lis = []
        for i in range(len(batch_names)):  # = LEN
            mix = []
            name = batch_names[i]
            # radar data at some timestep
            radar_img = cv2.imread(os.path.join(cfg.DATASET_PATH, name), flags=0)  # (565, 784)
            radar_img = radar_img[:-1, :-1]  # (564, 783)
            mix.append(radar_img)
            # elements data at some timestep
            year = name.split('-')[0]
            month = name.split('-')[1]
            for j in range(len(elements)):
                var = elements[j]
                if var == 'hu':
                    element_base_pth = '/mnt/d0/stations_after_time_interp'
                elif var == 't':
                    element_base_pth = '/mnt/d0/stations_after_time_interp'
                elif var == 'psl':
                    element_base_pth = '/mnt/d1/stations_after_time_interp'
                elif var == 'td':
                    element_base_pth = '/home/mazhf/stations_after_time_interp'
                elif var == 'dd':
                    element_base_pth = '/mnt/d2/stations_after_time_interp'
                elif var == 'ff':
                    element_base_pth = '/mnt/d0/stations_after_time_interp'
                else:
                    element_base_pth = ''
                    print('no this element_base_pth!')
                ele_pth = os.path.join(element_base_pth, var, year, month, name.split('.')[0] + '.npy')
                ele = np.load(ele_pth)  # (564, 783)
                mix.append(ele)  # 7 * H * W
            mix = np.transpose(np.array(mix), (1, 2, 0))  # H * W * 7
            mix = cv2.resize(mix, (cfg.width, cfg.height))  # h * w * 7
            mix_lis.append(mix)  # s * h * w * 7
        mix_lis = np.array(mix_lis)
        mix_lis = np.expand_dims(mix_lis, 1)  # s * 1 * h * w * 7
        mix_lis = np.transpose(mix_lis, (4, 0, 1, 2, 3))  # 7 * s * 1 * h * w
        return mix_lis

    def __len__(self):
        return len(self.batch_names_lis)


def make_datasets():
    train_data = Data(mode='train')
    test_data = Data(mode='test')
    return train_data, test_data


def save():
    base_pth = os.path.dirname(cfg.DATASET_PATH)
    modes = ['train', 'test']
    datasets = make_datasets()
    for i in range(len(modes)):
        mode = modes[i]
        save_pth = os.path.join(base_pth, 'resize_' + str(cfg.height), mode)
        # save
        if os.path.exists(save_pth):
            shutil.rmtree(save_pth)
        os.makedirs(save_pth)
        dataset = datasets[i]
        loader = DataLoader(dataset, num_workers=12, batch_size=1, shuffle=False,
                            pin_memory=False)  # 1 * 7 * s * 1 * h * w
        count = 0
        for batch in loader:
            batch = torch.squeeze(batch, dim=0)  # 7 * s * 1 * h * w
            np.save(os.path.join(save_pth, str(count) + '.npy'), batch)
            print(mode, count)
            count += 1


class Data_fast(Dataset):
    def __init__(self, pth):
        super().__init__()
        self.pth = pth
        self.batch_names_lis = os.listdir(pth)

    def __getitem__(self, i):
        batch_data = np.load(os.path.join(self.pth, self.batch_names_lis[i]))
        return batch_data  # 7 * s * 1 * h * w

    def __len__(self):
        return len(self.batch_names_lis)


def load_data():
    base_pth = os.path.dirname(cfg.DATASET_PATH)
    modes = ['train', 'test']
    ret_dataset = []
    for i in range(len(modes)):
        mode = modes[i]
        save_pth = os.path.join(base_pth, 'resize_' + str(cfg.height), mode)
        # load
        ret_dataset.append(Data_fast(save_pth))
    return ret_dataset + [ret_dataset[1]]


if __name__ == '__main__':
    # Save data first for quick read data
    save()
