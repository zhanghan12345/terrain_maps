# -*- coding: utf-8 -*-
"""
组合地形图：中国地形（仅中国区域）+ West China 地形 + 南海小地图
从大区域到小区域的嵌套展示，用于论文引言

边界数据: bou2_4p.shp / bou2_4l.shp（含藏南、台湾、阿克塞钦、九段线、省界）
南海小地图: 参考frykit风格，右下角展示南海诸岛
配色方案: 多种可选（custom / terrain / gist_earth / nature / geo）
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import geopandas as gpd
from shapely.ops import unary_union
from shapely.vectorized import contains as shapely_contains
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 可调参数 ====================
GRID_RESOLUTION = 0.05   # 插值分辨率(°): 0.1(快) / 0.05(精) / 0.025(极精)
OUTPUT_DPI = 300          # 输出DPI
SHOW_CONTOURS = True      # 地形等高线
SHOW_NINE_DASH = True     # 九段线
SHOW_SCS_INSET = True     # 南海小地图

# 配色方案选择:
#   'custom'     - 自定义: 深海蓝→低地绿→金黄→高山棕→白 (当前)
#   'terrain'    - matplotlib terrain: 经典地形色
#   'gist_earth' - matplotlib gist_earth: 类似ETOPO风格
#   'nature'     - Nature风格: 柔和的绿→黄→棕→灰白过渡
#   'geo'        - USGS地质风格: 鲜明的地形分层色
COLORMAP = 'custom'

# 数据与输出路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中国大区域范围
CN_LON = (70, 140)
CN_LAT = (10, 55)

# West China 小区域范围 (四川盆地及周边)
WC_LON = (100, 111)
WC_LAT = (25, 36)

# 南海小地图范围
SCS_LON = (105, 123)
SCS_LAT = (2, 24)

# ==================== 数据读取 ====================
print("=" * 60)
print("组合地形图 - China + West China + 南海")
print(f"精度: {GRID_RESOLUTION}° | DPI: {OUTPUT_DPI} | 配色: {COLORMAP}")
print("=" * 60)

print("\n[1/7] 读取地形数据...")
data = xr.open_dataset(os.path.join(DATA_DIR, 'world_geo.nc'), engine='scipy')

china_elev = data['z'].sel(y=slice(CN_LAT[0], CN_LAT[1]),
                           x=slice(CN_LON[0], CN_LON[1]))
westchina_elev = data['z'].sel(y=slice(WC_LAT[0], WC_LAT[1]),
                                x=slice(WC_LON[0], WC_LON[1]))
scs_elev = data['z'].sel(y=slice(SCS_LAT[0], SCS_LAT[1]),
                          x=slice(SCS_LON[0], SCS_LON[1]))


def interp_to_grid(da):
    new_lat = np.arange(float(da.y.min()), float(da.y.max()) + GRID_RESOLUTION / 2,
                        GRID_RESOLUTION)
    new_lon = np.arange(float(da.x.min()), float(da.x.max()) + GRID_RESOLUTION / 2,
                        GRID_RESOLUTION)
    return da.interp(y=new_lat, x=new_lon)


china_elev_interp = interp_to_grid(china_elev)
westchina_elev_interp = interp_to_grid(westchina_elev)
scs_elev_interp = interp_to_grid(scs_elev)

# ==================== 中国边界数据 ====================
print("[2/7] 加载中国边界数据...")

china_poly_gdf = gpd.read_file(os.path.join(DATA_DIR, 'bou2_4p.shp'))
china_poly_all = unary_union(china_poly_gdf.geometry.values)

china_line_gdf = gpd.read_file(os.path.join(DATA_DIR, 'bou2_4l.shp'))


def is_nine_dash(geom):
    coords = list(geom.coords)
    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    return max(lats) < 25 and min(lons) > 105


nine_dash_lines = []
land_border_lines = []
province_lines = []
coastline_lines = []

for _, row in china_line_gdf.iterrows():
    gbcode = row['GBCODE']
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    if gbcode == 61010:
        (nine_dash_lines if is_nine_dash(geom) else land_border_lines).append(geom)
    elif gbcode == 26010:
        coastline_lines.append(geom)
    elif gbcode in [61030, 61031, 61032, 61033, 61034, 61035]:
        province_lines.append(geom)

print(f"  国界(陆): {len(land_border_lines)} | 九段线: {len(nine_dash_lines)}")
print(f"  海岸线: {len(coastline_lines)} | 省界: {len(province_lines)}")

# ==================== 创建中国mask ====================
print("[3/7] 创建中国区域mask...")
clon = china_elev_interp.x.values
clat = china_elev_interp.y.values
clon2d, clat2d = np.meshgrid(clon, clat)

n_points = clon2d.size
batch_size = 500000
china_mask = np.zeros(n_points, dtype=bool)
for i in range(0, n_points, batch_size):
    end = min(i + batch_size, n_points)
    china_mask[i:end] = shapely_contains(
        china_poly_all, clon2d.ravel()[i:end], clat2d.ravel()[i:end]
    )
china_mask = china_mask.reshape(clon2d.shape)
print(f"  中国区域: {china_mask.sum()}/{china_mask.size} "
      f"({100 * china_mask.sum() / china_mask.size:.1f}%)")

# 南海区域mask
print("  创建南海区域mask...")
slon = scs_elev_interp.x.values
slat = scs_elev_interp.y.values
slon2d, slat2d = np.meshgrid(slon, slat)
n_scs = slon2d.size
scs_mask = np.zeros(n_scs, dtype=bool)
for i in range(0, n_scs, batch_size):
    end = min(i + batch_size, n_scs)
    scs_mask[i:end] = shapely_contains(
        china_poly_all, slon2d.ravel()[i:end], slat2d.ravel()[i:end]
    )
scs_mask = scs_mask.reshape(slon2d.shape)
print(f"  南海区域: {scs_mask.sum()}/{scs_mask.size} "
      f"({100 * scs_mask.sum() / scs_mask.size:.1f}%)")

# ==================== 配色方案 ====================
print("[4/7] 构建配色方案...")

COLORMAP_PRESETS = {
    'custom': [
        (0.00, (0.18, 0.53, 0.67)),
        (0.04, (0.64, 0.84, 0.98)),
        (0.08, (0.78, 0.90, 0.79)),
        (0.15, (0.49, 0.70, 0.26)),
        (0.30, (0.83, 0.66, 0.13)),
        (0.45, (0.98, 0.66, 0.15)),
        (0.60, (0.90, 0.32, 0.00)),
        (0.75, (0.55, 0.43, 0.39)),
        (0.88, (0.74, 0.74, 0.74)),
        (0.96, (0.94, 0.94, 0.94)),
        (1.00, (1.00, 1.00, 1.00)),
    ],
    'nature': [
        (0.00, (0.20, 0.45, 0.60)),   # 深海蓝
        (0.04, (0.55, 0.75, 0.85)),   # 浅海
        (0.10, (0.85, 0.92, 0.80)),   # 极低-淡绿
        (0.18, (0.35, 0.60, 0.28)),   # 低地-森林绿
        (0.30, (0.60, 0.72, 0.35)),   # 丘陵-黄绿
        (0.45, (0.78, 0.72, 0.38)),   # 中海拔-暖黄
        (0.58, (0.70, 0.52, 0.30)),   # 较高-棕色
        (0.72, (0.55, 0.38, 0.28)),   # 高山-深棕
        (0.85, (0.72, 0.68, 0.65)),   # 很高-灰
        (0.95, (0.90, 0.88, 0.85)),   # 极高-浅灰
        (1.00, (0.98, 0.98, 0.97)),   # 纯白
    ],
    'geo': [
        (0.00, (0.15, 0.30, 0.55)),
        (0.05, (0.50, 0.72, 0.90)),
        (0.12, (0.60, 0.80, 0.45)),
        (0.22, (0.85, 0.85, 0.30)),
        (0.35, (0.75, 0.55, 0.20)),
        (0.50, (0.60, 0.35, 0.15)),
        (0.65, (0.45, 0.28, 0.18)),
        (0.78, (0.55, 0.50, 0.45)),
        (0.90, (0.75, 0.73, 0.70)),
        (1.00, (0.95, 0.95, 0.94)),
    ],
    'etopo': [
        (0.00, (0.10, 0.25, 0.55)),
        (0.03, (0.30, 0.55, 0.80)),
        (0.06, (0.55, 0.75, 0.90)),
        (0.12, (0.30, 0.65, 0.35)),
        (0.20, (0.55, 0.75, 0.30)),
        (0.35, (0.80, 0.75, 0.40)),
        (0.50, (0.65, 0.45, 0.20)),
        (0.68, (0.50, 0.35, 0.22)),
        (0.82, (0.62, 0.58, 0.55)),
        (0.93, (0.82, 0.80, 0.78)),
        (1.00, (0.97, 0.97, 0.96)),
    ],
}


def build_colormap(name):
    """根据名称构建colormap"""
    if name in ['terrain', 'gist_earth']:
        cmap = plt.cm.get_cmap(name).copy()
        cmap.set_under((0.83, 0.93, 1.00))
        return cmap
    elif name in COLORMAP_PRESETS:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            name, COLORMAP_PRESETS[name], N=256
        )
        cmap.set_under((0.83, 0.93, 1.00))
        return cmap
    else:
        print(f"  未知配色 '{name}'，使用custom")
        return build_colormap('custom')


cmap_terrain = build_colormap(COLORMAP)
print(f"  当前配色: {COLORMAP}")
print(f"  可选配色: {list(COLORMAP_PRESETS.keys())} + terrain, gist_earth")

# ==================== 绘图 ====================
print("[5/7] 设置绘图参数...")
plt.rcParams.update({
    'font.family': 'Arial',
    'font.weight': 'bold',
    'font.size': 12,
    'axes.linewidth': 1.2,
})


def draw_lines(ax, lines, color, lw, alpha=1.0, style='-', zorder=5):
    for geom in lines:
        if geom.is_empty:
            continue
        if geom.geom_type == 'LineString':
            x, y = geom.xy
            ax.plot(x, y, color=color, linewidth=lw, alpha=alpha,
                    linestyle=style, zorder=zorder)
        elif geom.geom_type == 'MultiLineString':
            for g in geom.geoms:
                x, y = g.xy
                ax.plot(x, y, color=color, linewidth=lw, alpha=alpha,
                        linestyle=style, zorder=zorder)


def draw_map_content(ax, lon2d, lat2d, elev_data, mask=None, is_china=True,
                     lon_range=None, lat_range=None):
    """在指定axes上绘制地形图内容"""
    levels = np.concatenate([
        np.array([0, 10, 20, 50]),
        np.linspace(100, 1000, 19),
        np.linspace(1200, 6000, 17),
    ])
    levels = np.unique(levels)

    plot_data = np.where(mask, elev_data, np.nan) if mask is not None else elev_data

    cs = ax.contourf(lon2d, lat2d, plot_data,
                     levels=levels, cmap=cmap_terrain, extend='both')

    if SHOW_CONTOURS:
        ax.contour(lon2d, lat2d, plot_data,
                   levels=[200, 500, 1000, 2000, 3000, 4000, 5000],
                   colors='black', linewidths=0.15, alpha=0.25, linestyles='solid')

    # 省界
    draw_lines(ax, province_lines, color='#888888', lw=0.3, alpha=0.7, zorder=4)
    # 海岸线
    draw_lines(ax, coastline_lines, color='#333333', lw=0.7, zorder=5)
    # 国界线
    draw_lines(ax, land_border_lines, color='#111111', lw=1.2, zorder=6)

    # 九段线
    if SHOW_NINE_DASH:
        for geom in nine_dash_lines:
            coords = list(geom.coords)
            lats_ = [c[1] for c in coords]
            if lon_range and lat_range:
                if max(lats_) >= lat_range[0] and min(lats_) <= lat_range[1]:
                    x, y = geom.xy
                    ax.plot(x, y, color='#8B0000', linewidth=1.0,
                            linestyle='--', dashes=(8, 4), alpha=0.85, zorder=8)
            else:
                x, y = geom.xy
                ax.plot(x, y, color='#8B0000', linewidth=1.0,
                        linestyle='--', dashes=(8, 4), alpha=0.85, zorder=8)

    return cs


print("[6/7] 开始绘图...")

fig_width = 24
fig_height = 8.5
fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)

panel_height = 0.76
panel_width = 0.38
bottom_cn = 0.14
left_cn = 0.03
left_wc = left_cn + panel_width + 0.04

# ===== 左图：中国地形 =====
ax_cn = fig.add_axes([left_cn, bottom_cn, panel_width, panel_height])

elev_masked = np.where(china_mask, china_elev_interp.values, np.nan)
cs_cn = draw_map_content(ax_cn, clon2d, clat2d, china_elev_interp.values,
                          mask=china_mask, lon_range=CN_LON, lat_range=CN_LAT)

# West China 矩形框
rect_wc = Rectangle(
    (WC_LON[0], WC_LAT[0]),
    WC_LON[1] - WC_LON[0], WC_LAT[1] - WC_LAT[0],
    linewidth=2.8, edgecolor='#C62828', facecolor='none',
    linestyle='-', zorder=10
)
ax_cn.add_patch(rect_wc)

ax_cn.annotate(
    'West China', xy=(WC_LON[0] + 5.5, WC_LAT[1]),
    xytext=(WC_LON[0] + 5.5, WC_LAT[1] + 5.5),
    fontsize=14, fontweight='bold', color='#C62828',
    ha='center', va='center',
    arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.0),
    zorder=11
)

# ===== 南海小地图 (右下角，参考frykit风格) =====
if SHOW_SCS_INSET:
    # 南海小地图位于中国地图面板右下角
    scs_box_left = left_cn + panel_width - 0.088
    scs_box_bottom = bottom_cn + 0.02
    scs_box_width = 0.075
    scs_box_height = 0.24

    ax_scs = fig.add_axes([scs_box_left, scs_box_bottom,
                           scs_box_width, scs_box_height])

    # slon2d, slat2d 已在创建mask时定义，此处直接使用
    scs_elev_masked = np.where(scs_mask, scs_elev_interp.values, np.nan)

    cs_scs = ax_scs.contourf(
        slon2d, slat2d, scs_elev_masked,
        levels=np.concatenate([
            np.array([0, 10, 20, 50]),
            np.linspace(100, 1000, 19),
            np.linspace(1200, 6000, 17),
        ]),
        cmap=cmap_terrain, extend='both'
    )
    # 南海边界线
    draw_lines(ax_scs, coastline_lines, color='#333333', lw=0.6, zorder=5)
    draw_lines(ax_scs, land_border_lines, color='#111111', lw=1.0, zorder=6)

    # 九段线
    if SHOW_NINE_DASH:
        for geom in nine_dash_lines:
            coords = list(geom.coords)
            lats_ = [c[1] for c in coords]
            if max(lats_) >= SCS_LAT[0] and min(lats_) <= SCS_LAT[1]:
                x, y = geom.xy
                ax_scs.plot(x, y, color='#8B0000', linewidth=1.2,
                            linestyle='--', dashes=(8, 4), alpha=0.9, zorder=8)

    ax_scs.set_xlim(SCS_LON[0], SCS_LON[1])
    ax_scs.set_ylim(SCS_LAT[0], SCS_LAT[1])

    # 南海不显示刻度，只显示边框
    ax_scs.set_xticks([])
    ax_scs.set_yticks([])
    for spine in ax_scs.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    # 标注"南海诸岛"
    ax_scs.text(0.5, 0.02, 'South China Sea Islands',
                transform=ax_scs.transAxes, fontsize=8,
                ha='center', va='bottom', style='italic', color='#444444')

    # 南海小地图不额外画虚线框标注（本身已足够清晰）

# 经纬度
ax_cn.set_xlim(CN_LON[0], CN_LON[1])
ax_cn.set_ylim(CN_LAT[0], CN_LAT[1])
ax_cn.set_xticks(np.arange(70, 141, 10))
ax_cn.set_yticks(np.arange(10, 56, 10))
ax_cn.set_xticklabels([f'{int(t)}°E' for t in np.arange(70, 141, 10)], fontsize=12)
ax_cn.set_yticklabels([f'{int(t)}°N' for t in np.arange(10, 56, 10)], fontsize=12)
ax_cn.grid(True, linestyle='--', linewidth=0.3, alpha=0.4, color='gray')
for spine in ax_cn.spines.values():
    spine.set_linewidth(1.5)
ax_cn.text(0.015, 0.985, '(a)', transform=ax_cn.transAxes,
           fontsize=18, fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                     edgecolor='black', linewidth=0.8, alpha=0.9))

# ===== 右图：West China 地形 =====
ax_wc = fig.add_axes([left_wc, bottom_cn, panel_width, panel_height])

hlon = westchina_elev_interp.x.values
hlat = westchina_elev_interp.y.values
hlon2d, hlat2d = np.meshgrid(hlon, hlat)

cs_wc = draw_map_content(ax_wc, hlon2d, hlat2d,
                          westchina_elev_interp.values,
                          mask=None, lon_range=WC_LON, lat_range=WC_LAT)

# 省界加粗
draw_lines(ax_wc, province_lines, color='#555555', lw=0.9, alpha=0.85, zorder=4)

ax_wc.set_xlim(WC_LON[0], WC_LON[1])
ax_wc.set_ylim(WC_LAT[0], WC_LAT[1])
ax_wc.set_xticks(np.arange(WC_LON[0], WC_LON[1] + 1, 2))
ax_wc.set_yticks(np.arange(WC_LAT[0], WC_LAT[1] + 1, 2))
ax_wc.set_xticklabels([f'{int(t)}°E' for t in np.arange(WC_LON[0], WC_LON[1] + 1, 2)],
                      fontsize=12)
ax_wc.set_yticklabels([f'{int(t)}°N' for t in np.arange(WC_LAT[0], WC_LAT[1] + 1, 2)],
                      fontsize=12)
ax_wc.grid(True, linestyle='--', linewidth=0.3, alpha=0.4, color='gray')
for spine in ax_wc.spines.values():
    spine.set_linewidth(1.5)
ax_wc.text(0.015, 0.985, '(b)', transform=ax_wc.transAxes,
           fontsize=18, fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                     edgecolor='black', linewidth=0.8, alpha=0.9))

# ===== Colorbar =====
cbar_left = left_wc
cbar_bottom = bottom_cn - 0.09
cbar_width = panel_width
cbar_height = 0.025
cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
cbar_ticks = [0, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]
cb = fig.colorbar(cs_wc, cax=cbar_ax, orientation='horizontal',
                  ticks=cbar_ticks, extend='both')
cb.set_label('Elevation (m)', fontsize=14, fontweight='bold', labelpad=8)
cb.ax.tick_params(labelsize=11, direction='out', width=1.2, length=5)
cb.outline.set_linewidth(1.2)

# ==================== 保存 ====================
print("[7/7] 保存图片 (PNG + PDF + EPS)...")
output_base = os.path.join(OUTPUT_DIR, 'Fig_Topography_China_WestChina')

fig.savefig(output_base + '.png', dpi=OUTPUT_DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.3)
fig.savefig(output_base + '.pdf', dpi=OUTPUT_DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.3)
fig.savefig(output_base + '.eps', dpi=OUTPUT_DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.3,
            format='eps')
plt.close()

print(f"\n图片已保存:")
print(f"  {output_base}.png")
print(f"  {output_base}.pdf")
print(f"  {output_base}.eps")
print(f"\n配色方案: {COLORMAP}")
print(f"可选配色: {list(COLORMAP_PRESETS.keys())} + terrain, gist_earth")
print(f"(修改脚本顶部 COLORMAP 变量即可切换)")
print("\n完成!")
