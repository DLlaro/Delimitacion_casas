import os
import json
import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine
from skimage.morphology import(
    binary_closing,
    disk,
    binary_dilation
)
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
from tqdm import tqdm

def stitch_tiles(
    tiles_dir: str,
    pred_dir: str,
    output_tif: str,
    test: bool = False
) -> None:

    json_path = "tiles_metadata.json"

    if test:
        json_path = "meta_test.json"
        
    with open(os.path.join(tiles_dir, json_path)) as f:
        meta = json.load(f)

    W = meta["width"]
    H = meta["height"]
    tile_size = meta["tile_size"]
    stride = meta["stride"]

    # Manejar caso cuando stride = 0 (sin solapamiento)
    if stride == 0:
        stride = tile_size
        margin = 0
    else:
        margin = (tile_size - stride) // 2

    # Canvas y weight para acumular predicciones
    canvas = np.zeros((H, W), dtype=np.float32) 
    weight = np.zeros((H, W), dtype=np.uint8)

    print(f"Procesando {len(meta['tiles'])} tiles...")
    
    # Iterar con barra de progreso
    for tile in tqdm(meta["tiles"], desc="Stitching tiles", unit="tile"):
        # Leer archivo TIF
        pred_path = os.path.join(pred_dir, f"tile_{tile['id']}.tif")
        
        # Leer con rasterio
        with rasterio.open(pred_path) as src:
            pred = src.read(1)  # Leer primera banda
            pred = pred.astype(np.float32)

        x, y = tile["x"], tile["y"]

        # Si hay margen, hacer crop central; si no, usar toda la predicción
        if margin > 0:
            valid = pred[margin:-margin, margin:-margin]
            y0 = y + margin
            x0 = x + margin
        else:
            valid = pred
            y0 = y
            x0 = x

        y1 = y0 + valid.shape[0]
        x1 = x0 + valid.shape[1]

        # Asegurar que no excedemos los límites del canvas
        y1 = min(y1, H)
        x1 = min(x1, W)
        
        # Ajustar valid si es necesario
        valid_h = y1 - y0
        valid_w = x1 - x0
        valid = valid[:valid_h, :valid_w]

        canvas[y0:y1, x0:x1] += valid
        weight[y0:y1, x0:x1] += 1

    print("Calculando promedio final...")
    # Cálculo del promedio final
    final = canvas / np.maximum(weight, 1)

    # Transform desde metadata
    transform_list = meta["transform"][:6]
    transform = Affine(*transform_list)

    print("Guardando GeoTIFF...")
    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="uint8",
        crs=meta["crs"],
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(final.astype(np.uint8), 1)

    print(f"✓ GeoTIFF rearmado guardado en: {output_tif}")
