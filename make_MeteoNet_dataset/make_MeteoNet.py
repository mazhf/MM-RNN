# new version!
import numpy as np
import os
import cv2


years = ['2016', '2017', '2018']
root_pth = r'/mnt/d1/MeteoNet'
month_all = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
month_part = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
month_split = ['1', '2', '3']
resize = False

if resize:
    img_size = 120
    save_pth = r'/home/mazhf/Precipitation-Nowcasting/dataset/MeteoNet-120'
else:
    save_pth = r'/home/mazhf/Precipitation-Nowcasting/dataset/MeteoNet/raw_radar_imgs'

if not os.path.exists(save_pth):
    os.makedirs(save_pth)


def dBZ2Pixel(dBZ):
    dBZ = np.where(dBZ == 255, 0, dBZ)  # radar blind to black, 255 -> 0
    P = np.floor(255 * dBZ / 70)
    return P.astype(np.uint8)


count = 1
for i in range(len(years)):
    year = years[i]
    data_pth_year = os.path.join(root_pth, 'NW_reflectivity_old_product_' + year, 'NW_reflectivity_old_product_' + year)
    if year == '2018':
        months = month_part
    else:
        months = month_all
    for j in range(len(months)):
        month = months[j]
        data_pth_year_month = os.path.join(data_pth_year, 'reflectivity-old-NW-' + year + '-' + month, 'reflectivity-old-NW-' + year + '-' + month)
        for k in range(len(month_split)):
            split = month_split[k]
            data_pth = os.path.join(data_pth_year_month, 'reflectivity_old_NW_' + year + '_' + month + '.' + split + '.npz')
            data = np.load(data_pth, allow_pickle=True)
            # print(data.files)
            radar_data = data['data']
            radar_data = dBZ2Pixel(radar_data)
            dates = data['dates']
            # radar_data_all.append(radar_data)
            for l in range(radar_data.shape[0]):
                img_save_pth = os.path.join(save_pth, str(dates[l]) + '.png')
                if resize:
                    cv2.imwrite(img_save_pth, cv2.resize(radar_data[l], (img_size, img_size)))
                else:
                    cv2.imwrite(img_save_pth, radar_data[l])
                print(count)
                count += 1


# # test
# import numpy as np
# import os
# import cv2
#
#
# data_pth = '/mnt/d1/MeteoNet/NW_reflectivity_old_product_2016/NW_reflectivity_old_product_2016/reflectivity-old-NW-2016-01/reflectivity-old-NW-2016-01/reflectivity_old_NW_2016_01.1.npz'
# data = np.load(data_pth, allow_pickle=True)
# print(data.files)
# radar_data = data['data']
# a = radar_data[0]


# old version!
# import numpy as np
# import os
# import cv2
#
#
# years = ['2016', '2017', '2018']
# root_pth = r'/home/mazhf/Precipitation-Nowcasting/dataset/MeteoNet'
# month_all = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
# month_part = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
# month_split = ['1', '2', '3']
# resize = True
#
# if resize:
#     img_size = 120
#     save_pth = r'/home/mazhf/Precipitation-Nowcasting/dataset/MeteoNet-120'
# else:
#     save_pth = r'/home/mazhf/Precipitation-Nowcasting/dataset/MeteoNet/raw_radar_imgs'
#
#
# def dBZ2Pixel(dBZ):
#     P = np.floor(255 * dBZ / 70)
#     return P.astype(np.uint8)
#
#
# # radar_data_all = []
# dates_all = []
# miss_dates_all = []
# count = 1
# for i in range(len(years)):
#     year = years[i]
#     data_pth_year = os.path.join(root_pth, 'NW_reflectivity_old_product_' + year, 'NW_reflectivity_old_product_' + year)
#     if year == '2018':
#         months = month_part
#     else:
#         months = month_all
#     for j in range(len(months)):
#         month = months[j]
#         data_pth_year_month = os.path.join(data_pth_year, 'reflectivity-old-NW-' + year + '-' + month, 'reflectivity-old-NW-' + year + '-' + month)
#         for k in range(len(month_split)):
#             split = month_split[k]
#             data_pth = os.path.join(data_pth_year_month, 'reflectivity_old_NW_' + year + '_' + month + '.' + split + '.npz')
#             data = np.load(data_pth, allow_pickle=True)
#             # print(data.files)
#             radar_data = data['data']
#             radar_data = dBZ2Pixel(radar_data)
#             # radar_data_all.append(radar_data)
#             for l in range(radar_data.shape[0]):
#                 if count <= 260785:
#                     save_pth_train_test = os.path.join(save_pth, 'train')
#                 else:
#                     save_pth_train_test = os.path.join(save_pth, 'test')
#                 if not os.path.exists(save_pth_train_test):
#                     os.makedirs(save_pth_train_test)
#                 img_save_pth = os.path.join(save_pth_train_test, str(count) + '.png')
#                 if resize:
#                     cv2.imwrite(img_save_pth, cv2.resize(radar_data[l], (img_size, img_size)))
#                 else:
#                     cv2.imwrite(img_save_pth, radar_data[l])
#                 print(count)
#                 count += 1
#             dates = data['dates']
#             dates_all.append(dates)
#             miss_dates = data['miss_dates']
#             miss_dates_all.append(miss_dates)
#
# # radar_data_all = np.concatenate(radar_data_all, axis=0)  # 298041 radar imgs
# dates_all = np.concatenate(dates_all, axis=0)
# miss_dates_all = np.concatenate(miss_dates_all, axis=0)
#
