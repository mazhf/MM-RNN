import numpy as np
import os
import pandas as pd


def format_date(x):  # 2016-01-01 00:00:00.npy
    x = x.split('.')[0]  # 2016-01-01 00:00:00
    x0 = x.split(' ')[0]  # 2016-01-01
    x1 = x.split(' ')[1]  # 00:00:00
    ymd = x0.split('-')
    hms = x1.split(':')
    format_s = ymd[0] + ymd[1] + ymd[2] + hms[0] + hms[1] + hms[2]  # 20160101000000
    format_s = int(format_s)
    return format_s


time_read_pth = os.path.join('/media/mazhf/气象组数据移动硬盘/stations_after_space_interp')
time_save_pth = os.path.join('/mnt/d0', 'stations_after_time_interp')
# variables = ['dd', 'ff', 'hu', 'td', 't', 'psl']
# years = ['2016', '2017', '2018']
# months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
variables = ['t']
years = ['2017']
months = ['06', '07', '08', '09', '10', '11', '12']
for var in variables:
    for year in years:
        for month in months:
            # read and save pth
            read_pth = os.path.join(time_read_pth, var, year, month)
            save_pth = os.path.join(time_save_pth, var, year, month)
            if not os.path.exists(save_pth):
                os.makedirs(save_pth)
            # read one month file and sort it
            file_names = os.listdir(read_pth)
            file_names_sorted = sorted(file_names, key=format_date)
            print(file_names_sorted[0])
            print(file_names_sorted[-1])
            all_data = []
            # load file and concat along time axis
            for i in range(len(file_names_sorted)):
                file_name = file_names_sorted[i]
                file_pth = os.path.join(read_pth, file_name)
                file = np.load(file_pth)
                file = file.flatten()
                file = list(file)  # some time step data
                all_data.append(file)  # all time step data
            # interpolate to every 1 min and get every 5min data
            all_data = np.array(all_data)
            dates = [i.split('.')[0] for i in file_names_sorted]
            dates = pd.to_datetime(dates)
            five_min_data = []
            for i in range(all_data.shape[1]):
                data = pd.Series(all_data[:, i], index=dates)
                data = data.resample(rule='1min').interpolate()  # 不按6分钟的间隔的不连续数据也能插值为1分钟的
                data = data.iloc[::5]  # 每隔5个数据读取
                data = data.to_list()
                five_min_data.append(data)
            # save time interpolate data - every 5min
            five_min_data = np.array(five_min_data).T
            five_min_date = pd.date_range(dates[0], dates[-1], freq='5min')
            assert five_min_data.shape[0] == five_min_date.shape[0]
            for i in range(five_min_data.shape[0]):
                some_time_data = five_min_data[i, :]
                some_time_data = some_time_data.reshape([565 - 1, 784 - 1])  # reason see kriging.py
                some_time_save_pth = os.path.join(save_pth, str(five_min_date[i]) + '.npy')
                np.save(some_time_save_pth, some_time_data)
            # save last month time data  每月最后一天的23:55:00的数据使用23:54:00的代替
            month_last_data = all_data[-1, :]
            month_last_data = month_last_data.reshape([565 - 1, 784 - 1])
            last_time_save_pth = os.path.join(save_pth, str(dates[-1] + pd.offsets.Minute(1)) + '.npy')
            np.save(last_time_save_pth, month_last_data)


'''
# test one month
import numpy as np
import os
import pandas as pd


def format_date(x):  # 2016-01-01 00:00:00.npy
    x = x.split('.')[0]  # 2016-01-01 00:00:00
    x0 = x.split(' ')[0]  # 2016-01-01
    x1 = x.split(' ')[1]  # 00:00:00
    ymd = x0.split('-')
    hms = x1.split(':')
    format_s = ymd[0] + ymd[1] + ymd[2] + hms[0] + hms[1] + hms[2]  # 20160101000000
    format_s = int(format_s)
    return format_s


read_pth = '/home/mazhf/stations_after_space_interp/psl/2016/01'
save_pth = '/home/mazhf/stations_after_space_interp/psl/2016/test'

file_names = os.listdir(read_pth)
file_names_sorted = sorted(file_names, key=format_date)
print(file_names_sorted[0])
print(file_names_sorted[-1])

all_data = []
for i in range(len(file_names_sorted)):
    file_name = file_names_sorted[i]
    file_pth = os.path.join(read_pth, file_name)
    file = np.load(file_pth)
    file = file.flatten()
    file = list(file)  # some time step data
    all_data.append(file)  # all time step data

all_data = np.array(all_data)
dates = [i.split('.')[0] for i in file_names_sorted]
dates = pd.to_datetime(dates)
five_min_data = []
for i in range(all_data.shape[1]):
    data = pd.Series(all_data[:, i], index=dates)
    data = data.resample(rule='1min').interpolate()
    data = data.iloc[::5]  # 每隔5个数据读取
    data = data.to_list()
    five_min_data.append(data)

five_min_data = np.array(five_min_data).T
five_min_date = pd.date_range(dates[0], dates[-1], freq='5min')
assert five_min_data.shape[0] == five_min_date.shape[0]

for i in range(five_min_data.shape[0]):
    some_time_data = five_min_data[i, :]
    some_time_data = some_time_data.reshape([565-1, 784-1])
    some_time_save_pth = os.path.join(save_pth, str(five_min_date[i]) + '.npy')
    np.save(some_time_save_pth, some_time_data)

# save last month time data
# 每月最后一天的23:55:00的数据使用23:54:00的代替
month_last_data = all_data[-1, :]

month_last_data = month_last_data.reshape([565 - 1, 784 - 1])
last_time_save_pth = os.path.join(save_pth, str(dates[-1] + pd.offsets.Minute(1)) + '.npy')
np.save(last_time_save_pth, month_last_data)


# 归一化，如何确定最大值和最小值？通过遍历时间插值完毕后的所有该气象要素的文件来确定


# test interpolate
import numpy as np
import os
import pandas as pd

index = pd.to_datetime(['20160101000000', '20160101000600', '20160101001200', '20160101002400'])
data1 = pd.Series(np.arange(len(index)), index=index)
data2 = data1.resample(rule='1min').interpolate()
data3 = data2.iloc[::5]  # 每隔5行读取
data3 = data3.to_list()
'''
