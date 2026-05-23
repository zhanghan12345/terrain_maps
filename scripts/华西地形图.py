# -*- coding: utf-8 -*-
"""
华西地形图 —— West China 区域 (100-111°E, 25-36°N)
需要: cartopy, cmaps, maskout
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
import cmaps
import maskout
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm
import copy
import os

# ==================== 可调参数 ====================
RESOLUTION = 0.1   # 插值分辨率(°): 0.1(快) / 0.05(精) / 0.025(极精)
OUTPUT_DPI = 600
# ================================================

# 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据读取
file = os.path.join(DATA_DIR, 'world_geo.nc')
data = xr.open_dataset(file)
height = data['z'].sel(y=slice(25, 36), x=slice(100, 111))


def interp_(data):
    new_lat = np.arange(25, 36.01, RESOLUTION)
    new_lon = np.arange(100, 111.01, RESOLUTION)
    return data.interp(y=new_lat, x=new_lon)


height1 = interp_(height)

# 画图
plt.rcParams['font.family'] = 'Times New Roman'

fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax = fig.add_subplot(1, 1, 1, projection=proj)

# 经纬度范围 (华西区域)
leftlon, rightlon, lowerlat, upperlat = (100, 111, 25, 36)
ax.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())

# 地形填色
cmap = copy.copy(cmaps.MPL_terrain)
cmap.set_under('lightskyblue')

levels = np.linspace(0, 5001, 100)
olon = height1.x.values
olat = height1.y.values
olon, olat = np.meshgrid(olon, olat)
cs = ax.contourf(
    olon, olat, height1,
    levels=levels, cmap=cmap, extend='both',
    transform=ccrs.PlateCarree(), zorder=1
)

# 地图要素
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=3)

# 读取中国边界 (含藏南、台湾、阿克塞钦)
china = shpreader.Reader(os.path.join(DATA_DIR, 'bou2_4l.shp'))
ax.add_geometries(
    china.geometries(), crs=ccrs.PlateCarree(),
    facecolor='none', edgecolor='black', lw=2.5
)

# 坐标轴
ax.set_xticks(np.arange(leftlon, rightlon + 1, 2), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(lowerlat, upperlat + 1, 2), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params(direction="out", length=4, width=1.2)

tick_font = fm.FontProperties(family='Times New Roman', weight='bold', size=14)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(tick_font)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.8)

# Colorbar
cb = fig.colorbar(
    cs, ax=ax, orientation='horizontal',
    pad=0.04, fraction=0.05, aspect=30,
    ticks=np.arange(0, 5001, 500), label='m'
)
cb.ax.tick_params(labelsize=14, direction='out', width=1.2, length=4)
cb.outline.set_linewidth(1.2)
for label in cb.ax.get_xticklabels():
    label.set_fontproperties(tick_font)
cb.set_label('m', fontproperties=tick_font)

fig.subplots_adjust(bottom=0.12, top=0.95, left=0.08, right=0.95)

plt.show()

output_path = os.path.join(OUTPUT_DIR, 'Fig_WestChina_Topography.png')
fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
print(f"华西地形图已保存: {output_path}")
