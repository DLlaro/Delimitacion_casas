import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import random

# ----------------------------------------------------
# 1. Dividir raster en tiles y contar píxeles válidos
# ----------------------------------------------------
def crear_tiles_con_validos(ruta_raster, n_rows=5, n_cols=5):

    raster_id = ruta_raster.split("_")[-1].split(".")[0]

    with rasterio.open(ruta_raster) as src:
        mask = src.dataset_mask() > 0  # True en píxeles válidos
        height, width = mask.shape
        transform = src.transform
        crs = src.crs

    row_edges = np.linspace(0, height, n_rows + 1, dtype=int)
    col_edges = np.linspace(0, width, n_cols + 1, dtype=int)

    tiles = []
    idx = 1


    for i in range(n_rows):
        for j in range(n_cols):
            tile_id = raster_id+"_"+str(idx)
            
            r0, r1 = row_edges[i], row_edges[i+1]
            c0, c1 = col_edges[j], col_edges[j+1]

            valid_px = int(mask[r0:r1, c0:c1].sum())

            corners_px = [(c0, r0), (c1, r0), (c1, r1), (c0, r1)]
            corners_real = [transform * (c, r) for (c, r) in corners_px]
            poly = Polygon(corners_real)

            tiles.append({
                "tile_id": tile_id,
                "row": i,
                "col": j,
                "valid_px": valid_px,
                "split": "train",
                "geometry": poly,
            })
            idx += 1

    return tiles, crs



# ----------------------------------------------------
# 2. Agrupar tiles según densidad (alto/medio/bajo)
# ----------------------------------------------------
def estratificar_por_densidad(tiles):
    # Extraer valores
    vals = [t["valid_px"] for t in tiles]
    q33, q66 = np.quantile(vals, [0.33, 0.66])

    for t in tiles:
        if t["valid_px"] >= q66:
            t["densidad"] = "alta"
        elif t["valid_px"] >= q33:
            t["densidad"] = "media"
        else:
            t["densidad"] = "baja"

    return tiles



# ----------------------------------------------------
# 3. Vecindad: tiles vecinos NO pueden ir al mismo split
# ----------------------------------------------------
def vecinos(tile, tiles):
    """Devuelve lista de IDs de tiles adyacentes (8-direcciones)."""
    neigh = []
    for t in tiles:
        if t["tile_id"] == tile["tile_id"]:
            continue
        
        # Si están pegados en filas o columnas o diagonales
        if abs(t["row"] - tile["row"]) <= 1 and abs(t["col"] - tile["col"]) <= 1:
            neigh.append(t["tile_id"])
    return neigh



# ----------------------------------------------------
# 4. Algoritmo final de asignación
# ----------------------------------------------------
def asignar_splits(tiles, frac_val=0.20, frac_test=0.10, random_seed = 42):
    random.seed = random_seed

    total_valid = sum(t["valid_px"] for t in tiles)
    objetivo_val = frac_val * total_valid
    objetivo_test = frac_test * total_valid

    # Agrupar por densidad
    densidades = {"alta": [], "media": [], "baja": []}
    for t in tiles:
        densidades[t["densidad"]].append(t)

    # Aleatorizar dentro de cada grupo
    for grupo in densidades.values():
        random.shuffle(grupo)

    # Crear lista global mezclada "nivelada"
    mezcla = []
    for a, m, b in zip(
        densidades["alta"] + [None]*10,
        densidades["media"] + [None]*10,
        densidades["baja"] + [None]*10,
    ):
        if a: mezcla.append(a)
        if m: mezcla.append(m)
        if b: mezcla.append(b)

    # Filtrar None
    mezcla = [x for x in mezcla if x]

    # Auxiliar para saber vecinos
    id_to_tile = {t["tile_id"]: t for t in tiles}

    def tiene_vecino_en_split(tile, split):
        for vid in vecinos(tile, tiles):
            if id_to_tile[vid]["split"] == split:
                return True
        return False

    # ---- Asignación de VALIDACIÓN ----
    acc_val = 0
    for t in mezcla:
        if acc_val >= objetivo_val:
            break
        if not tiene_vecino_en_split(t, "val"):
            t["split"] = "val"
            acc_val += t["valid_px"]

    # ---- Asignación de TEST ----
    acc_test = 0
    for t in mezcla:
        if acc_test >= objetivo_test:
            break
        if t["split"] == "train" and not tiene_vecino_en_split(t, "test"):
            t["split"] = "test"
            acc_test += t["valid_px"]

    return tiles, acc_val, acc_test, total_valid



# ----------------------------------------------------
# 5. Función principal
# ----------------------------------------------------
def generar_tiles_spatial_split(
    ruta_raster,
    out_dir: str,
    n_rows=5,
    n_cols=5,
    frac_val=0.20,
    frac_test=0.10,
    shuffle_seed = 42
) -> None:

    tiles, crs = crear_tiles_con_validos(ruta_raster, n_rows, n_cols)
    tiles = estratificar_por_densidad(tiles)
    tiles, acc_val, acc_test, total_valid = asignar_splits(tiles, frac_val, frac_test, shuffle_seed)

    # Convertir a GeoDataFrame
    gdf = gpd.GeoDataFrame(tiles, crs=crs)

    print("Píxeles válidos totales:", total_valid)
    print("Val   asignado:", acc_val, f"({acc_val/total_valid*100:.1f}%)")
    print("Test asignado:", acc_test, f"({acc_test/total_valid*100:.1f}%)")
    print("Train:", total_valid - acc_val - acc_test)

    gdf.to_file(filename=f"{out_dir}/division_en_tile.gpkg",driver='GPKG')