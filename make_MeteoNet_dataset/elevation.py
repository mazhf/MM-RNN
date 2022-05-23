import netCDF4 as nc
import cv2
import numpy as np
import cmaps
import matplotlib.pyplot as plt


read_pth = '/home/mazhf/Precipitation-Nowcasting/dataset/MeteoNet/Masks/Masks/NW_masks.nc'
save_gray_pth = '/home/mazhf/Precipitation-Nowcasting/dataset/MeteoNet/Masks/Masks/NW_elevation.png'
save_rgb_resize_pth = '/home/mazhf/Precipitation-Nowcasting/dataset/MeteoNet-120/MeteoNet_elevation_120.png'
dataset = nc.Dataset(read_pth)
print(dataset.variables.keys())
land_sea_mask = dataset.variables['lsm'][:]
elevation = dataset.variables['p3008'][:]
elevation = np.array(elevation * land_sea_mask)
elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min()) * 255
elevation = elevation.astype(np.uint8)
cv2.imwrite(save_gray_pth, elevation)
elevation = cv2.resize(elevation, (120, 120))
plt.imsave(save_rgb_resize_pth, elevation, cmap=cmaps.matlab_jet)
