import os
import json
import numpy as np
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm

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
