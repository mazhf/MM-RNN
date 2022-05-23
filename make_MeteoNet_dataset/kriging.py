import numpy as np
import os
import pandas as pd
import shutil
from pykrige.ok import OrdinaryKriging
from concurrent.futures import ThreadPoolExecutor, wait


thread_pool = ThreadPoolExecutor(max_workers=12)

_base_pth = '/media/mazhf/气象组数据移动硬盘/气象数据/数据集/MeteoNet/NW_Ground_Stations/NW_Ground_Stations'


# radar region coord
radar_region_coord_pth = os.path.join('/media/mazhf/气象组数据移动硬盘/气象数据/数据集/MeteoNet/Radar_coords/Radar_coords/radar_coords_NW.npz')
radar_region_coord = np.load(radar_region_coord_pth, allow_pickle=True)
print(radar_region_coord.files)
lats = radar_region_coord['lats']
lons = radar_region_coord['lons']
lats_min = lats.min()
lats_max = lats.max()
lons_min = lons.min()
lons_max = lons.max()

# Kriging interpolation  ——   http://pykrige.readthedocs.io/
gridx = np.arange(lons_min, lons_max, 0.01)  # 没有包含最大值，导致空间插值像素长和宽都少了1，影响不大
gridy = np.arange(lats_min, lats_max, 0.01)


def kriging(_df, _uni_dt, _var, _save_pth):
    datas = _df[_df.date == _uni_dt]  # all stations data in this date
    OK = OrdinaryKriging(datas.lon, datas.lat, datas[_var], variogram_model="linear")
    z, ss = OK.execute("grid", gridx, gridy)
    z_pth = os.path.join(_save_pth, _uni_dt + '.npy')
    np.save(z_pth, np.array(z))
    print(_var, ' ', _uni_dt)


space_pth = os.path.join('/media/mazhf/气象组数据移动硬盘', 'stations_after_space_interp')
# if os.path.exists(space_pth):
#     shutil.rmtree(space_pth)
# variables = ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']
# years = ['2016', '2017', '2018']
# months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
variables = ['t']
years = ['2017']
# months = ['01', '02', '03', '04', '05', '06', '07', '08']
months = ['06', '07', '08']
for var in variables:
    for year in years:
        for month in months:
            save_pth = os.path.join(space_pth, var, year, month)
            if not os.path.exists(save_pth):
                os.makedirs(save_pth)
            pth = os.path.join(_base_pth, 'vars_split_by_month', var, year + '-' + month + '.csv')
            df = pd.read_csv(pth)
            uni_dts = df.date.unique()
            for uni_dt in uni_dts:
                future = thread_pool.submit(kriging, df, uni_dt, var, save_pth)
                # OK = OrdinaryKriging(datas.lon, datas.lat, datas[var], variogram_model="linear")
                # z, ss = OK.execute("grid", gridx, gridy)
                # z_pth = os.path.join(save_pth, uni_dt + '.npy')
                # np.save(z_pth, np.array(z))
                # print(var, ' ', uni_dt)

