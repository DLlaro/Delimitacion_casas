import os
import json
import numpy as np
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm

def stitch_tiles_multiclase(
    tiles_dir: str,
    pred_dir: str,
    output_tif: str,
    num_classes: int = 3,
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

    if stride == 0:
        stride = tile_size
        margin = 0
    else:
        margin = (tile_size - stride) // 2

    # Canvas multiclase
    canvas = np.zeros((H, W, num_classes), dtype=np.uint16)

    print(f"Procesando {len(meta['tiles'])} tiles...")

    for tile in tqdm(meta["tiles"], desc="Stitching tiles", unit="tile"):
        pred_path = os.path.join(pred_dir, f"tile_{tile['id']}.tif")

        with rasterio.open(pred_path) as src:
            pred = src.read(1).astype(np.uint8)  # clases 0,1,2

        x, y = tile["x"], tile["y"]

        if margin > 0:
            valid = pred[margin:-margin, margin:-margin]
            y0 = y + margin
            x0 = x + margin
        else:
            valid = pred
            y0 = y
            x0 = x

        y1 = min(y0 + valid.shape[0], H)
        x1 = min(x0 + valid.shape[1], W)

        valid = valid[: y1 - y0, : x1 - x0]

        # VOTACIÓN POR CLASE
        for c in range(num_classes):
            canvas[y0:y1, x0:x1, c] += (valid == c)

    print("Calculando clase final por píxel...")
    priority = np.array([0, 1, 2])  # bg < road < building
    final = np.argmax(canvas + priority, axis=-1)

    transform = Affine(*meta["transform"][:6])

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
        compress="lzw"
    ) as dst:
        dst.write(final, 1)

    print(f"GeoTIFF multiclase rearmado: {output_tif}")

def stitch_tiles_by_class(tiles_dir, pred_dir, output_dir, num_classes=3, test=False):
    json_path = "tiles_metadata.json" if not test else "meta_test.json"
    
    with open(os.path.join(tiles_dir, json_path)) as f:
        meta = json.load(f)
    
    H, W = meta["height"], meta["width"]
    tile_size = meta["tile_size"]
    stride = meta["stride"] if meta["stride"] > 0 else tile_size
    margin = (tile_size - stride) // 2 if stride < tile_size else 0

    # Canvas por clase
    canvases = [np.zeros((H, W), dtype=np.uint8) for _ in range(num_classes)]

    for tile in tqdm(meta["tiles"], desc="Stitching tiles"):
        pred_path = os.path.join(pred_dir, f"tile_{tile['id']}.tif")
        with rasterio.open(pred_path) as src:
            pred = src.read(1).astype(np.uint8)
        
        x, y = tile["x"], tile["y"]
        valid = pred[margin:-margin, margin:-margin] if margin > 0 else pred
        y0, x0 = (y + margin, x + margin) if margin > 0 else (y, x)
        y1 = min(y0 + valid.shape[0], H)
        x1 = min(x0 + valid.shape[1], W)
        valid = valid[: y1 - y0, : x1 - x0]

        for c in range(num_classes):
            canvases[c][y0:y1, x0:x1] = np.maximum(canvases[c][y0:y1, x0:x1], (valid == c).astype(np.uint8))

    transform = Affine(*meta["transform"][:6])

    # Guardar cada clase por separado
    os.makedirs(output_dir, exist_ok=True)
    for c in range(1,num_classes):
        out_path = os.path.join(output_dir, f"class_{c}.tif")
        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=H, width=W,
            count=1, dtype="uint8",
            crs=meta["crs"],
            transform=transform,
            compress="lzw"
        ) as dst:
            dst.write(canvases[c], 1)
        print(f"Clase {c} guardada en {out_path}")

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

    print(f"GeoTIFF rearmado guardado en: {output_tif}")
