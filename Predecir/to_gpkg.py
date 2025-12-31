import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import numpy as np

def raster_to_vector(mask_path: str, output_vector: str, background_value: int = 0):
    """
    Convierte una máscara raster a un vector (GPKG o Shapefile).

    Args:
        mask_path (str): Ruta al archivo raster de la máscara.
        output_vector (str): Ruta donde guardar el vector (GPKG/Shapefile).
        background_value (int): Valor del fondo a ignorar (por defecto 0).
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    # Crear máscara booleana ignorando el fondo
    mask_bool = mask != background_value
    shapes_generator = shapes(mask, mask=mask_bool, transform=transform)

    # Convertir a GeoDataFrame
    geoms = [{'geometry': shape(geom), 'class_value': int(value)} 
             for geom, value in shapes_generator]

    gdf = gpd.GeoDataFrame(geoms, crs=crs)

    # Guardar a archivo vectorial
    gdf.to_file(output_vector, driver="GPKG")
    print(f"Vectorizado guardado en {output_vector}")
