# Terrain Maps - China & West China Topography

Combined terrain map for paper introduction: China overview + West China detail + South China Sea inset.

## Quick Start

```bash
# 1. Place terrain data in this directory
#    Download world_geo.nc (~233MB) and put it here

# 2. Run
cd scripts
python 组合地形图.py
```

## Files

| File | Description |
|------|-------------|
| `scripts/组合地形图.py` | Combined figure (no cartopy needed) |
| `scripts/中国地形图.py` | China-only map (requires cartopy/cmaps/maskout) |
| `scripts/华西地形图.py` | West China map (requires cartopy/cmaps/maskout) |
| `bou2_4p.*` | China polygon boundaries (incl. Taiwan, South Tibet, Aksai Chin) |
| `bou2_4l.*` | China line boundaries + nine-dash line + province boundaries |

## Colormap Options

Edit `COLORMAP` variable at the top of the script:

`custom` | `nature` | `geo` | `etopo` | `terrain` | `gist_earth`

## Output

Generated in `output/`: PNG, PDF, EPS formats.

## Dependencies

`numpy` `xarray` `matplotlib` `scipy` `geopandas` `shapely`
