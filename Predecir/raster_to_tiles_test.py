import os
import json
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box
from rasterio.transform import xy
import geopandas as gpd


def compute_global_percentiles_stream_per_band(tif_path: str, pmin=2, pmax=98, bands=[1,2,3], nbins=10000):
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        nodata = src.nodata
        block_size = 1024
        n_bands = len(bands)
        
        # Min/max por banda
        global_min = np.full(n_bands, np.inf)
        global_max = np.full(n_bands, -np.inf)
        
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                win = Window(x, y, min(block_size, w-x), min(block_size, h-y))
                block = src.read(bands, window=win).astype(np.float32)  # shape: (n_bands, rows, cols)
                
                for b in range(n_bands):
                    band_data = block[b]
                    if nodata is not None:
                        valid_values = band_data[band_data != nodata]
                    else:
                        valid_values = band_data.flatten()
                    
                    if valid_values.size > 0:
                        global_min[b] = min(global_min[b], valid_values.min())
                        global_max[b] = max(global_max[b], valid_values.max())
        
        # Histograma por banda
        hist = np.zeros((n_bands, nbins), dtype=np.int64)
        bin_edges = [np.linspace(global_min[b], global_max[b], nbins+1) for b in range(n_bands)]
        
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                win = Window(x, y, min(block_size, w-x), min(block_size, h-y))
                block = src.read(bands, window=win).astype(np.float32)
                
                for b in range(n_bands):
                    band_data = block[b]
                    if nodata is not None:
                        values = band_data[band_data != nodata]
                    else:
                        values = band_data.flatten()
                    
                    hist_block, _ = np.histogram(values, bins=bin_edges[b])
                    hist[b] += hist_block
        
        # Percentiles por banda
        lo = np.zeros(n_bands)
        hi = np.zeros(n_bands)
        
        for b in range(n_bands):
            cdf = np.cumsum(hist[b])
            cdf = cdf / cdf[-1]
            idx_lo = np.searchsorted(cdf, pmin/100)
            idx_hi = np.searchsorted(cdf, pmax/100)
            lo[b] = bin_edges[b][min(idx_lo, len(bin_edges[b])-1)]
            hi[b] = bin_edges[b][min(idx_hi, len(bin_edges[b])-1)]
    
    return lo, hi  # Ahora son arrays de n_bands elementos

def normalize_percentiles_per_band(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, nodata_value=None):
    """x shape: (height, width, n_bands)"""
    x = x.astype(np.float32)
    x_norm = np.zeros_like(x)
    
    for b in range(x.shape[-1]):
        band = x[..., b]
        if nodata_value is not None:
            valid = band != nodata_value
            x_norm[valid, b] = np.clip((band[valid] - lo[b]) / (hi[b] - lo[b] + 1e-6), 0, 1)
        else:
            x_norm[..., b] = np.clip((band - lo[b]) / (hi[b] - lo[b] + 1e-6), 0, 1)
    
    out = (x_norm * 255).astype(np.uint8)
    
    if nodata_value is not None:
        valid_mask = np.all(x != nodata_value, axis=-1)
        out[~valid_mask] = 0
    
    return out

def tile_geotiff(
    tif_path: str,
    out_dir: str,
    tile_size: int = 512,
    overlap: float = 0.5,
    nodata_threshold: float = 0.9,
    black_tile_threshold: float = 5.0
):
    os.makedirs(out_dir, exist_ok=True)

    # Calcular percentiles globales
    lo, hi = compute_global_percentiles_stream_per_band(tif_path)

    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        bands = src.count
        transform = src.transform
        nodata_value = src.nodata
        stride = int(tile_size * (1 - overlap))

        metadata = {
            "tile_size": tile_size,
            "overlap": overlap,
            "stride": stride,
            "tiles": [],
            "transform": list(transform),
            "crs": str(src.crs),
            "width": W,
            "height": H,
            "nodata_value": nodata_value
        }

        features = []
        tile_id = 0

        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)
                tile = src.read(window=window)

                # Detectar nodata
                if nodata_value is not None:
                    nodata_mask = (tile == nodata_value)
                    nodata_fraction = nodata_mask.sum() / tile.size
                else:
                    nodata_fraction = 0

                if nodata_fraction > nodata_threshold:
                    continue

                if bands >= 3:
                    tile_rgb = np.stack([tile[0], tile[1], tile[2]], axis=-1)
                else:
                    raise ValueError("TIF must have at least 3 bands (RGB).")

                tile_rgb_8bit = normalize_percentiles_per_band(tile_rgb, lo, hi, nodata_value)

                # Detectar tiles negros
                mean_intensity = np.mean(tile_rgb_8bit)
                if mean_intensity < black_tile_threshold:
                    #print(f"Skipping tile_{tile_id} at ({x},{y}). Mean intensity {mean_intensity:.2f} < {black_tile_threshold}")
                    continue
                
                tile_transform = window_transform(window, src.transform)

                tile_path = os.path.join(out_dir, f"tile_{tile_id}.tif")

                profile = src.profile.copy()
                profile.update({
                    "driver": "GTiff",
                    "height": tile_size,
                    "width": tile_size,
                    "transform": tile_transform,
                    "count": 3,
                    "dtype": rasterio.uint8,
                    "nodata": 0
                })

                with rasterio.open(tile_path, "w", **profile) as dst:
                    # rasterio usa (band, row, col)
                    dst.write(tile_rgb_8bit[:, :, 0], 1)
                    dst.write(tile_rgb_8bit[:, :, 1], 2)
                    dst.write(tile_rgb_8bit[:, :, 2], 3)


                x_min, y_max = xy(transform, y, x, offset='ul')
                x_max, y_min = xy(transform, y + tile_size, x + tile_size, offset='lr')

                geom = box(x_min, y_min, x_max, y_max)

                features.append({
                    "geometry": geom,
                    "tile_id": tile_id,
                    "img_path": os.path.basename(tile_path),
                    "pixel_x": x,
                    "pixel_y": y,
                    "nodata_frac": float(nodata_fraction)
                })

                metadata["tiles"].append({
                    "id": tile_id,
                    "x": x,
                    "y": y,
                    "path": tile_path,
                    "nodata_fraction": float(nodata_fraction)
                })

                tile_id += 1

    gdf = gpd.GeoDataFrame(
        features,
        crs = src.crs
    )
    gpkg_path = os.path.join(out_dir, "tiles_index.gpkg")
    gdf.to_file(gpkg_path, layer="tiles", driver="GPKG")

    with open(os.path.join(out_dir, "tiles_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Tiling terminado.")
