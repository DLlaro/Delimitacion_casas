import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.transform import xy
from rasterio import features
from shapely.geometry import box
from rasterio.mask import mask
import geopandas as gpd
from scipy.ndimage import binary_closing

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
        # pixel válido solo si TODAS las bandas son válidas
        valid = np.all(x != nodata_value, axis=-1)
        x_norm = np.zeros_like(x)
        x_norm[valid] = np.clip((x[valid] - lo) / (hi - lo + 1e-6), 0, 1)
    else:
        x_norm = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    out =  (x_norm * 255).astype(np.uint8)

    # nodata → negro (consistente)
    out[~valid] = 0
    return out

def data_split(
    tif_path: str,
    gpkg_casas_path: str,
    gpkg_caminos_path: str,
    gpkg_division_tiles: str,
    out_dir_path: str,
    semilla = 42,
    gpkg_area_excluida_path: str = None,
) -> None:
    """
        Función para generar los parches y máscaras respectivas de la imagen satelital

        Args:
            tif_path (float): Ruta de imagen satelital (.tif)
            gpkg_casas_path (str): Ruta de capa vectorial de las casas (.gpkg)
            gpkg_caminos_path (str): Ruta de capa vectorial de los caminos (.gpkg)
            gpkg_division_tiles (str): Ruta de capa vectorial de los tiles de train, val, test (.gpkg)
            out_dir (str): Carpeta donde se guardara train, val, test
            semilla (int): Semilla para configurar el número aleatorio generado al momento de guardar parches completamente vacíos
            gpkg_area_excluida_path (str): Ruta de capa vectorial de area que se desea excluir al realizar los parches (.gpkg)
    """

    np.random.seed = semilla
    # =============================================================
    #  1. CONFIGURACIÓN
    # =============================================================
    ruta_TIF = tif_path

    ##Identificador de tif
    ID_TIF = ruta_TIF.split("_")[-1].split(".")[0]

    # GEOMETRÍAS CASAS
    ruta_casas = gpkg_casas_path

    # GEOMETRÍAS CAMINOS
    ruta_caminos = gpkg_caminos_path

    if gpkg_area_excluida_path is not None:
        # GEOMETRÍAS AREA FUERA DE INTERES (CIUDAD)
        ruta_area_excluida = gpkg_area_excluida_path

    # TILES ESPACIALES (train/valid/test)
    ruta_tiles = gpkg_division_tiles

    # Carpeta de salida
    carpeta_salida = out_dir_path

    os.makedirs(carpeta_salida, exist_ok=True)

    # Crear subcarpetas
    splits = ["train", "val", "test"]
    for sp in splits:
        os.makedirs(os.path.join(carpeta_salida, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(carpeta_salida, sp, "masks"), exist_ok=True)

    ## Leer valores de imagen entera para normalizar
    print("Leyendo raster entero para calcular percentiles para posterior normalización de parches")
    lo, hi = compute_global_percentiles_stream(tif_path)

    # Patch params
    PATCH_SIZE = 512
    STRIDE = 256
    NOINFO_VALUE = 0   # clase 0 = no información
    BACKGROUND = 1
    BUILDING = 255
    ROAD = 125
    EXCLUDE = 10
    # =============================================================
    # 2. CARGAR INSUMOS
    # =============================================================

    print("Cargando raster...")
    src = rasterio.open(ruta_TIF)

    print("Cargando geometrías...")
    
    gdf_casas_geoms = gpd.read_file(ruta_casas)
    gdf_casas_geoms = gdf_casas_geoms.to_crs(src.crs)
    print("Geometria de : Casas")

    
    gdf_caminos_geoms = gpd.read_file(ruta_caminos)
    gdf_caminos_geoms = gdf_caminos_geoms.to_crs(src.crs)
    print("Geometria de : Caminos")

    if gpkg_area_excluida_path is not None:
        gdf_exclusion_geoms = gpd.read_file(ruta_area_excluida)
        gdf_exclusion_geoms = gdf_exclusion_geoms.to_crs(src.crs)
        print("Geometria de : Area fuera de interes")

    print("Cargando tiles...")
    tiles = gpd.read_file(ruta_tiles)
    tiles = tiles.to_crs(src.crs)

    # Campo que indica split: debe ser "train", "valid", "test"
    campo_split = "split"    # ajusta si tiene otro nombre

    #Crea gpkg de tiles

    out_gpkg = f"{out_dir_path}/tiles_debug_{ID_TIF}.gpkg"
    layer_name = "tiles"
    # crear GPKG vacío solo una vez
    if not os.path.exists(out_gpkg):
        gdf_empty = gpd.GeoDataFrame(
            columns=["tile_id", "patch_id", "split", "geometry"],
            geometry="geometry",
            crs=src.crs
        )

        gdf_empty.to_file(out_gpkg, layer=layer_name, driver="GPKG")

    # =============================================================
    #  3. PROCESAR TILE POR TILE
    # =============================================================
    i = 0
    tiles_usados= []

    for idx, tile in tiles.iterrows():
        
        split = tile[campo_split]   # "train" / "valid" / "test"
        if split not in splits:
            print(f"Tile {idx} tiene split desconocido, saltando…")
            continue

        print(f"\nProcesando tile {idx} → {split}")

            # ==== 3.1 Recorte del TIFF al tile ====
        try:
            img_tile, transform_tile = mask(src, [tile.geometry], crop=True, nodata=NOINFO_VALUE)
        except:
            print("Tile vacío, saltando…")
            continue

        # img_tile tiene shape (C, H, W)
        tile_h, tile_w = img_tile.shape[1], img_tile.shape[2]

        # ==== 3.2 Filtrar geometrías que intersectan tile ====
        geoms_casas = gdf_casas_geoms[gdf_casas_geoms.intersects(tile.geometry)]
        geoms_caminos = gdf_caminos_geoms[gdf_caminos_geoms.intersects(tile.geometry)]
        if gpkg_area_excluida_path is not None:
            geoms_exclusion = gdf_exclusion_geoms[gdf_exclusion_geoms.intersects(tile.geometry)]

        roads_tile = np.zeros((tile_h, tile_w), dtype="uint8")
        # ================= RASTERIZAR ROADS DEL TILE =================
        if not geoms_caminos.empty:
            roads_tile = features.rasterize(
                ((g, ROAD) for g in geoms_caminos.geometry),
                out_shape=(tile_h, tile_w),
                transform=transform_tile,
                fill=0,
                dtype="uint8"
            )

        # ================= RASTERIZAR BUILDINGS DEL TILE =================
        buildings_tile = np.zeros((tile_h, tile_w), dtype="uint8")

        if not geoms_casas.empty:
            buildings_tile = features.rasterize(
                ((g, BUILDING) for g in geoms_casas.geometry),
                out_shape=(tile_h, tile_w),
                transform=transform_tile,
                fill=0,
                dtype="uint8"
            )

        if gpkg_area_excluida_path is not None:
            # ================= RASTERIZAR EXCLUSION DEL TILE =================
            exclude_tile = np.zeros((tile_h, tile_w), dtype="uint8")

            if not geoms_exclusion.empty:
                exclude_tile = features.rasterize(
                    ((geom, EXCLUDE) for geom in geoms_exclusion.geometry),
                    out_shape=(tile_h, tile_w),
                    transform=transform_tile,
                    fill=0,
                    dtype="uint8"
                )

        ys = list(range(0, tile_h - PATCH_SIZE + 1, STRIDE))
        xs = list(range(0, tile_w - PATCH_SIZE + 1, STRIDE))

        # asegurar último parche pegado al borde
        if ys[-1] != tile_h - PATCH_SIZE:
            ys.append(tile_h - PATCH_SIZE)

        if xs[-1] != tile_w - PATCH_SIZE:
            xs.append(tile_w - PATCH_SIZE)
        # =============================================================
        #  4. GENERAR PATCHES
        # =============================================================

        for y in ys:
            for x in xs:
                
                patch_id = f"{ID_TIF}_{i:06d}"
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                win_transform = rasterio.windows.transform(window, transform_tile)

                # ---- Extraer parche imagen ----
                img_patch = img_tile[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                # ---- Detectar nodata (0) ----
                nodata_mask = (img_patch[0] == NOINFO_VALUE)
                if nodata_mask.all():
                    continue
                
                if gpkg_area_excluida_path is not None:
                    # ---- Excluir Tiles que abarquen area de exclusion
                    exclude_patch = exclude_tile[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                    # ejemplo: excluir si >30% del patch
                    
                    if np.mean(exclude_patch == EXCLUDE) > 0.3:
                        #print("Se excluyo el tile: "+ patch_id) #debug
                        continue

                # Mascara base
                mask_patch = np.full(shape=(PATCH_SIZE, PATCH_SIZE), fill_value=BACKGROUND, dtype="uint8")
                 # Asignar nodata (0)
                mask_patch[nodata_mask] = NOINFO_VALUE

                roads_patch = roads_tile[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                build_patch = buildings_tile[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                # Caminos
                mask_patch[(mask_patch != NOINFO_VALUE) & (roads_patch == ROAD)] = ROAD
                # Buildings despues de caminos por prioridad (sobreescriba)
                mask_patch[(mask_patch != NOINFO_VALUE) & (build_patch == BUILDING)] = BUILDING

                # Suavizado buildings
                b = (mask_patch == BUILDING)
                b = binary_closing(b, structure=np.ones((3,3)))
                mask_patch[(b) & (mask_patch != ROAD)] = BUILDING

                # Suavizado roads (más alargado)
                r = (mask_patch == ROAD)
                r = binary_closing(r, structure=np.ones((1,5)))
                mask_patch[(r) & (mask_patch != BUILDING)] = ROAD

                # ---- Evitar guardar patches 100% vacío (solo fondo y/o nodata) ----
                valid_classes = set(np.unique(mask_patch)) - {BACKGROUND, NOINFO_VALUE}

                if len(valid_classes) == 0:
                    if np.random.rand() > 0.03:
                        continue

                # ---- Guardar TIFF georreferenciado ----
                out_img = os.path.join(carpeta_salida, split, "images", f"img_{patch_id}.tif")
                out_mask = os.path.join(carpeta_salida, split, "masks", f"mask_{patch_id}.tif")

                # Codigo para visualizar las imagenes en rgb realizando normalizado
                # Normalizar imagen a 0-255 si es necesario
                tile_rgb = np.stack([img_patch[0], img_patch[1], img_patch[2]], axis=-1)
                tile_rgb_8bit = normalize_percentiles(tile_rgb, lo, hi, NOINFO_VALUE)
                
                tile_saved = tile_rgb_8bit

                #print("shape tile:", tile_saved.shape)

                #geometria del patch_id
                patch_geom = box(*rasterio.windows.bounds(window, transform_tile))

                # Guardamos datos del tile procesado
                tiles_usados.append({
                    "tile_id": tile["tile_id"],
                    "patch_id": patch_id,
                    "split": split,
                    "geometry": patch_geom
                })

                # cada 250 tiles guardamos en el GPKG
                if len(tiles_usados) >= 250:
                    gpd.GeoDataFrame(tiles_usados, crs=src.crs).to_file(
                        out_gpkg,
                        layer=layer_name,
                        driver="GPKG",
                        mode="a"
                    )
                    tiles_usados = []  # reset

                # Guardar imagen
                with rasterio.open(
                    out_img, "w",
                    driver="GTiff",
                    height=PATCH_SIZE,
                    width=PATCH_SIZE,
                    count=3,
                    dtype=tile_saved.dtype,
                    crs=src.crs,
                    transform=win_transform
                ) as dst:
                    dst.write(tile_saved[:,:,0], 1)
                    dst.write(tile_saved[:,:,1], 2)
                    dst.write(tile_saved[:,:,2], 3)

                # Guardar máscara
                with rasterio.open(
                    out_mask, "w",
                    driver="GTiff",
                    height=PATCH_SIZE,
                    width=PATCH_SIZE,
                    count=1,
                    dtype="uint8",
                    crs=src.crs,
                    transform=win_transform,
                    nodata=NOINFO_VALUE
                ) as dst:
                    dst.write(mask_patch, 1)

                i += 1

    if len(tiles_usados) > 0:
        gpd.GeoDataFrame(tiles_usados, crs=src.crs).to_file(
            out_gpkg,
            layer=layer_name,
            driver="GPKG",
            mode="a"
        )
        tiles_usados = []

    print("\n LISTO: Dataset generado correctamente.")
    print(f"Total de parches generados: {patch_id}")