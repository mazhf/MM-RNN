import numpy as np
import os


variables = ['t']
# variables = ['psl']
years = ['2016', '2017', '2018']
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
for var in variables:
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
    max_value = None
    min_value = None
    count = 0
    for year in years:
        for month in months:
            read_pth = os.path.join(element_base_pth, var, year, month)
            file_names = os.listdir(read_pth)
            for i in range(len(file_names)):
                file_name = file_names[i]
                file_pth = os.path.join(read_pth, file_name)
                file = np.load(file_pth)
                if count == 0:  # init
                    max_value = file.max()
                    min_value = file.min()
                else:
                    if max_value < file.max():
                        max_value = file.max()
                    if min_value > file.min():
                        min_value = file.min()
                count += 1
    print(var, max_value, min_value)
