import os
import pandas as pd
import shutil
from concurrent.futures import ThreadPoolExecutor, wait

_base_pth = '/media/mazhf/气象组数据移动硬盘/气象数据/数据集/MeteoNet/NW_Ground_Stations/NW_Ground_Stations'


thread_pool = ThreadPoolExecutor(max_workers=1)


# merge
raw_data_all_save_pth = os.path.join(_base_pth, 'NW_Ground_Stations_all.csv')
if os.path.exists(raw_data_all_save_pth):
    os.remove(raw_data_all_save_pth)
years = [2016, 2017, 2018]
for i in range(len(years)):
    year = years[i]
    data_pth = os.path.join(_base_pth, 'NW_Ground_Stations_' + str(year) + '.csv')
    data = pd.read_csv(data_pth)
    if i == 0:
        header = True
    else:
        header = False
    data.to_csv(raw_data_all_save_pth, mode='a', index=False, header=header)


# split by station
stations_data_save_pth = os.path.join(_base_pth, 'stations')
if not os.path.exists(stations_data_save_pth):
    os.makedirs(stations_data_save_pth)
all_data = pd.read_csv(raw_data_all_save_pth)
all_data = all_data[~all_data['date'].str.contains('date')]  # 删除数据为列名的错误行
all_data['date'] = pd.to_datetime(all_data['date'])  # str转换为日期，方便后续单站点数据按日期排序,但保存为csv后还是str格式, excel或许可以
print(all_data.head())
print(all_data.tail())
stations = all_data['number_sta'].unique()
for station in stations:
    df_sta = all_data[all_data['number_sta'] == station]
    df_sta.sort_values(by='date', axis=0, ascending=True, inplace=True, na_position='last')
    df_sta.to_csv(os.path.join(stations_data_save_pth, str(station) + '.csv'), index=False, header=True)


# count station samples
count = 0
count_stations = pd.DataFrame([], columns=['stations, samples'])
count_stations_save_pth = os.path.join(_base_pth, 'stations_count.csv')
for station in stations:
    df = pd.read_csv(os.path.join(stations_data_save_pth, str(station) + '.csv'))
    count_stations[str(station)] = [df.shape[0]]
    count_stations.T.to_csv(count_stations_save_pth, header=True, index=True)
    print(df.shape)
    count += df.shape[0]


# clean station
threshold_sample_num = 180000
clean_stations = []
stations_data_save_pth = os.path.join(_base_pth, 'stations')
stations = [i.split('.')[0] for i in os.listdir(stations_data_save_pth)]
for station in stations:
    df = pd.read_csv(os.path.join(stations_data_save_pth, station + '.csv'))
    if df.shape[0] < threshold_sample_num:
        continue
    else:
        clean_stations.append(station)

all_dates_lis = []
for clean_station in clean_stations:
    df = pd.read_csv(os.path.join(stations_data_save_pth, str(clean_station) + '.csv'))
    all_dates_lis.append(set(df['date']))  # date row becomes str type
all_dates = set.union(*all_dates_lis)  # union set

stations_clean_data_save_pth = os.path.join(_base_pth, 'stations_clean')
if os.path.exists(stations_clean_data_save_pth):
    shutil.rmtree(stations_clean_data_save_pth)
if not os.path.exists(stations_clean_data_save_pth):
    os.makedirs(stations_clean_data_save_pth)


def fill_miss(clean_station_):
    df_ = pd.read_csv(os.path.join(stations_data_save_pth, str(clean_station_) + '.csv'))
    df_ = df_.fillna(method='bfill').fillna(method='ffill')  # fill nan, already sorted in split step
    diff = all_dates.difference(set(df_['date'].tolist()))
    for mss_dt in diff:
        df_ = df_.append([{'date': mss_dt}], ignore_index=True)  # add miss date, other data will be NaN
    df_ = df_.sort_values(by='date', axis=0, ascending=True, na_position='last')  # sort again
    df_ = df_.fillna(method='bfill').fillna(method='ffill')  # fill NaN date row data
    df_.to_csv(os.path.join(stations_clean_data_save_pth, str(clean_station_) + '.csv'), header=True, index=False)
    print(clean_station_)


for clean_station in clean_stations:
    future = thread_pool.submit(fill_miss, clean_station)


# check clean stations samples
stations_clean_data_save_pth = os.path.join(_base_pth, 'stations_clean')
name_lis = os.listdir(stations_clean_data_save_pth)
clean_num = []
problem_station = []
for name in name_lis:
    df = pd.read_csv(os.path.join(stations_clean_data_save_pth, name))
    clean_num.append(df.shape[0])
    if df.shape[0] != 263403:
        problem_station.append(name)
        print(name)
print(pd.DataFrame(clean_num).value_counts())


# remove duplicate dates
stations_clean_data_save_pth = os.path.join(_base_pth, 'stations_clean')
for name in problem_station:
    df = pd.read_csv(os.path.join(stations_clean_data_save_pth, name))
    df = df.drop_duplicates(subset='date')
    print(df.shape[0])
    df.to_csv(os.path.join(stations_clean_data_save_pth, name), index=False, header=True)


# check clean stations samples again
stations_clean_data_save_pth = os.path.join(_base_pth, 'stations_clean')
name_lis = os.listdir(stations_clean_data_save_pth)
clean_num = []
problem_station = []
for name in name_lis:
    df = pd.read_csv(os.path.join(stations_clean_data_save_pth, name))
    clean_num.append(df.shape[0])
    if df.shape[0] != 263403:
        problem_station.append(name)
        print(name)
print(pd.DataFrame(clean_num).value_counts())


# 数据不一致，有两种，前者比后者少0.01
stations_clean_data_save_pth = os.path.join(_base_pth, 'stations_clean')
name_lis = os.listdir(stations_clean_data_save_pth)
for name in name_lis:
    pth = os.path.join(stations_clean_data_save_pth, name)
    df = pd.read_csv(pth)
    df.lat = df.lat.max()
    df.lon = df.lon.max()
    df.to_csv(pth, index=False, header=True)
    print(name)


# split by var
miss_var = pd.read_excel(os.path.join(_base_pth, 'station_miss_variable_statistic.xlsx'))
variables = ['t']
var_pth = os.path.join(_base_pth, 'vars')
if os.path.exists(var_pth):
    shutil.rmtree(var_pth)
if not os.path.exists(var_pth):
    os.makedirs(var_pth)
stations_clean_data_save_pth = os.path.join(_base_pth, 'stations_clean')
clean_stations = [i.split('.')[0] for i in os.listdir(stations_clean_data_save_pth)]
var_count_lis = [[], [], [], [], [], [], []]
for clean_station in clean_stations:
    df = pd.read_csv(os.path.join(stations_clean_data_save_pth, clean_station + '.csv'))
    for v in range(len(variables)):
        var = variables[v]
        if (miss_var[miss_var.number_sta == int(clean_station)][var] != 'miss').bool():  # if station's var no miss
            if len(var_count_lis[v]) == 0:
                header = True
            else:
                header = False
            df[['number_sta', 'lat', 'lon', 'date', var]].to_csv(os.path.join(var_pth, var + '.csv'), mode='a', index=False, header=header)  # add to var.csv
            var_count_lis[v].append(1)


# count vars date num
var_pth = os.path.join(_base_pth, 'vars')
variables = ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']
for v in variables:
    pth = os.path.join(var_pth, v + '.csv')
    df = pd.read_csv(pth)
    print(v)
    print(df.shape)
    print(len(df.date.unique()))


# split by month
var_pth = os.path.join(_base_pth, 'vars')
variables = ['t']
years = ['2016', '2017', '2018']
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
for var in variables:
    df = pd.read_csv(os.path.join(var_pth, var + '.csv'))
    var_split_pth = os.path.join(_base_pth, 'vars_split_by_month', var)
    if not os.path.exists(var_split_pth):
        os.makedirs(var_split_pth)
    for year in years:
        for month in months:
            df_month = df[df.date.map(lambda x: x[0:7]).isin([year + '-' + month])]
            df_month.to_csv(os.path.join(var_split_pth, year + '-' + month + '.csv'), header=True, index=False)





# import os
# import pandas as pd
# import shutil
# from concurrent.futures import ThreadPoolExecutor, wait
#
# _base_pth = '/mnt/d0/MeteoNet/NW_Ground_Stations/NW_Ground_Stations'
#
#
# thread_pool = ThreadPoolExecutor(max_workers=1)
#
#
# # merge
# raw_data_all_save_pth = os.path.join(_base_pth, 'NW_Ground_Stations_all.csv')
# if os.path.exists(raw_data_all_save_pth):
#     os.remove(raw_data_all_save_pth)
# years = [2016, 2017, 2018]
# for i in range(len(years)):
#     year = years[i]
#     data_pth = os.path.join(_base_pth, 'NW_Ground_Stations_' + str(year) + '.csv')
#     data = pd.read_csv(data_pth)
#     if i == 0:
#         header = True
#     else:
#         header = False
#     data.to_csv(raw_data_all_save_pth, mode='a', index=False, header=header)
#
#
# # split by station
# stations_data_save_pth = os.path.join(_base_pth, 'stations')
# if not os.path.exists(stations_data_save_pth):
#     os.makedirs(stations_data_save_pth)
# all_data = pd.read_csv(raw_data_all_save_pth)
# all_data = all_data[~all_data['date'].str.contains('date')]  # 删除数据为列名的错误行
# all_data['date'] = pd.to_datetime(all_data['date'])  # str转换为日期，方便后续单站点数据按日期排序,但保存为csv后还是str格式, excel或许可以
# print(all_data.head())
# print(all_data.tail())
# stations = all_data['number_sta'].unique()
# for station in stations:
#     df_sta = all_data[all_data['number_sta'] == station]
#     df_sta.sort_values(by='date', axis=0, ascending=True, inplace=True, na_position='last')
#     df_sta.to_csv(os.path.join(stations_data_save_pth, str(station) + '.csv'), index=False, header=True)
#
#
# # count station samples
# count = 0
# count_stations = pd.DataFrame([], columns=['stations, samples'])
# count_stations_save_pth = os.path.join(_base_pth, 'stations_count.csv')
# for station in stations:
#     df = pd.read_csv(os.path.join(stations_data_save_pth, str(station) + '.csv'))
#     count_stations[str(station)] = [df.shape[0]]
#     count_stations.T.to_csv(count_stations_save_pth, header=True, index=True)
#     print(df.shape)
#     count += df.shape[0]
#
#
# # clean station
# threshold_sample_num = 180000
# clean_stations = []
# stations_data_save_pth = os.path.join(_base_pth, 'stations')
# stations = [i.split('.')[0] for i in os.listdir(stations_data_save_pth)]
# for station in stations:
#     df = pd.read_csv(os.path.join(stations_data_save_pth, station + '.csv'))
#     if df.shape[0] < threshold_sample_num:
#         continue
#     else:
#         clean_stations.append(station)
#
# all_dates_lis = []
# for clean_station in clean_stations:
#     df = pd.read_csv(os.path.join(stations_data_save_pth, str(clean_station) + '.csv'))
#     all_dates_lis.append(set(df['date']))  # date row becomes str type
# all_dates = set.union(*all_dates_lis)  # union set
#
# stations_clean_data_save_pth = os.path.join(_base_pth, 'stations_clean')
# if os.path.exists(stations_clean_data_save_pth):
#     shutil.rmtree(stations_clean_data_save_pth)
# if not os.path.exists(stations_clean_data_save_pth):
#     os.makedirs(stations_clean_data_save_pth)
#
#
# def fill_miss(clean_station_):
#     df_ = pd.read_csv(os.path.join(stations_data_save_pth, str(clean_station_) + '.csv'))
#     df_ = df_.fillna(method='bfill').fillna(method='ffill')  # fill nan, already sorted in split step
#     diff = all_dates.difference(set(df_['date'].tolist()))
#     for mss_dt in diff:
#         df_ = df_.append([{'date': mss_dt}], ignore_index=True)  # add miss date, other data will be NaN
#     df_ = df_.sort_values(by='date', axis=0, ascending=True, na_position='last')  # sort again
#     df_ = df_.fillna(method='bfill').fillna(method='ffill')  # fill NaN date row data
#     df_.to_csv(os.path.join(stations_clean_data_save_pth, str(clean_station_) + '.csv'), header=True, index=False)
#     print(clean_station_)
#
#
# for clean_station in clean_stations:
#     future = thread_pool.submit(fill_miss, clean_station)
#
#
# # check clean stations samples
# stations_clean_data_save_pth = os.path.join(_base_pth, 'stations_clean')
# name_lis = os.listdir(stations_clean_data_save_pth)
# clean_num = []
# problem_station = []
# for name in name_lis:
#     df = pd.read_csv(os.path.join(stations_clean_data_save_pth, name))
#     clean_num.append(df.shape[0])
#     if df.shape[0] != 263403:
#         problem_station.append(name)
#         print(name)
# print(pd.DataFrame(clean_num).value_counts())
#
#
# # remove duplicate dates
# stations_clean_data_save_pth = os.path.join(_base_pth, 'stations_clean')
# for name in problem_station:
#     df = pd.read_csv(os.path.join(stations_clean_data_save_pth, name))
#     df = df.drop_duplicates(subset='date')
#     print(df.shape[0])
#     df.to_csv(os.path.join(stations_clean_data_save_pth, name), index=False, header=True)
#
#
# # check clean stations samples again
# stations_clean_data_save_pth = os.path.join(_base_pth, 'stations_clean')
# name_lis = os.listdir(stations_clean_data_save_pth)
# clean_num = []
# problem_station = []
# for name in name_lis:
#     df = pd.read_csv(os.path.join(stations_clean_data_save_pth, name))
#     clean_num.append(df.shape[0])
#     if df.shape[0] != 263403:
#         problem_station.append(name)
#         print(name)
# print(pd.DataFrame(clean_num).value_counts())
#
#
# # 数据不一致，有两种，前者比后者少0.01
# stations_clean_data_save_pth = os.path.join(_base_pth, 'stations_clean')
# name_lis = os.listdir(stations_clean_data_save_pth)
# for name in name_lis:
#     pth = os.path.join(stations_clean_data_save_pth, name)
#     df = pd.read_csv(pth)
#     df.lat = df.lat.max()
#     df.lon = df.lon.max()
#     df.to_csv(pth, index=False, header=True)
#     print(name)
#
#
# # split by var
# miss_var = pd.read_excel(os.path.join(_base_pth, 'station_miss_variable_statistic.xlsx'))
# variables = ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']
# var_pth = os.path.join(_base_pth, 'vars')
# if os.path.exists(var_pth):
#     shutil.rmtree(var_pth)
# if not os.path.exists(var_pth):
#     os.makedirs(var_pth)
# stations_clean_data_save_pth = os.path.join(_base_pth, 'stations_clean')
# clean_stations = [i.split('.')[0] for i in os.listdir(stations_clean_data_save_pth)]
# var_count_lis = [[], [], [], [], [], [], []]
# for clean_station in clean_stations:
#     df = pd.read_csv(os.path.join(stations_clean_data_save_pth, clean_station + '.csv'))
#     for v in range(len(variables)):
#         var = variables[v]
#         if (miss_var[miss_var.number_sta == int(clean_station)][var] != 'miss').bool():  # if station's var no miss
#             if len(var_count_lis[v]) == 0:
#                 header = True
#             else:
#                 header = False
#             df[['number_sta', 'lat', 'lon', 'date', var]].to_csv(os.path.join(var_pth, var + '.csv'), mode='a', index=False, header=header)  # add to var.csv
#             var_count_lis[v].append(1)
#
#
# # count vars date num
# var_pth = os.path.join(_base_pth, 'vars')
# variables = ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']
# for v in variables:
#     pth = os.path.join(var_pth, v + '.csv')
#     df = pd.read_csv(pth)
#     print(v)
#     print(df.shape)
#     print(len(df.date.unique()))
#
#
# # split by month
# var_pth = os.path.join(_base_pth, 'vars')
# variables = ['dd', 'ff', 'precip', 'hu', 'td', 't', 'psl']
# years = ['2016', '2017', '2018']
# months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
# for var in variables:
#     df = pd.read_csv(os.path.join(var_pth, var + '.csv'))
#     var_split_pth = os.path.join(_base_pth, 'vars_split_by_month', var)
#     if not os.path.exists(var_split_pth):
#         os.makedirs(var_split_pth)
#     for year in years:
#         for month in months:
#             df_month = df[df.date.map(lambda x: x[0:7]).isin([year + '-' + month])]
#             df_month.to_csv(os.path.join(var_split_pth, year + '-' + month + '.csv'), header=True, index=False)
#
#
