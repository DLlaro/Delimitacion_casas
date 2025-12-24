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


def compute_global_percentiles_stream(tif_path: str, pmin=2, pmax=98, bands=[1,2,3], nbins=10000):
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        nodata = src.nodata
        block_size = 1024

        # Estimar global min y max por streaming
        global_min, global_max = np.inf, -np.inf
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                win = Window(x, y, min(block_size, w-x), min(block_size, h-y))
                block = src.read(bands, window=win).astype(np.float32)
                if nodata is not None:
                    mask = block != nodata
                    valid_values = block[mask]
                else:
                    valid_values = block.flatten()
                if valid_values.size > 0:
                    global_min = min(global_min, valid_values.min())
                    global_max = max(global_max, valid_values.max())

        # Crear histograma acumulativo
        hist = np.zeros(nbins, dtype=np.int64)
        bin_edges = np.linspace(global_min, global_max, nbins+1)
        
        # Segundo pase para llenar histograma
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                win = Window(x, y, min(block_size, w-x), min(block_size, h-y))
                block = src.read(bands, window=win).astype(np.float32)
                if nodata is not None:
                    mask = block != nodata
                    values = block[mask]
                else:
                    values = block.flatten()
                hist_block, _ = np.histogram(values, bins=bin_edges)
                hist += hist_block
        
        # Percentiles globales
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]
        def percentile_from_cdf(p):
            idx = np.searchsorted(cdf, p/100)
            return bin_edges[min(idx, len(bin_edges)-1)]
        lo = percentile_from_cdf(pmin)
        hi = percentile_from_cdf(pmax)
    
    return lo, hi

def normalize_percentiles(x: np.ndarray, lo: float, hi: float, nodata_value=None):
    x = x.astype(np.float32)
    if nodata_value is not None:
        mask = x != nodata_value
        x_norm = np.zeros_like(x)
        x_norm[mask] = np.clip((x[mask] - lo) / (hi - lo + 1e-6), 0, 1)
    else:
        x_norm = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    return (x_norm * 255).astype(np.uint8)

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
    lo, hi = compute_global_percentiles_stream(tif_path)

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

                tile_rgb_8bit = normalize_percentiles(tile_rgb, lo, hi, nodata_value)

                # Detectar tiles negros
                mean_intensity = np.mean(tile_rgb_8bit)
                if mean_intensity < black_tile_threshold:
                    print(f"Skipping tile_{tile_id} at ({x},{y}). Mean intensity {mean_intensity:.2f} < {black_tile_threshold}")
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
