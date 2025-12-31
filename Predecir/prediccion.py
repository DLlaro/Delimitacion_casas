import rasterio
import numpy as np
import os
from glob import glob
from tqdm import tqdm

# Si entrenaste con ImageNet mean/std
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def predict_tiles_multiclase(input_dir, output_dir, model):
    """
    Predice máscaras multiclase para todos los tiles TIF en una carpeta.

    Clases:
    0 = background
    1 = road
    2 = building
    """
    os.makedirs(output_dir, exist_ok=True)
    tile_paths = glob(os.path.join(input_dir, "*.tif"))
    
    print(f"Encontrados {len(tile_paths)} tiles para predecir")
    
    for image_path in tqdm(tile_paths, desc="Prediciendo tiles"):
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Leer imagen
            with rasterio.open(image_path) as src:
                img = src.read()                  # (bands, H, W)
                img = np.moveaxis(img, 0, -1)     # (H, W, bands)
                transform = src.transform
                crs = src.crs
                height = src.height
                width = src.width
            
            # Preparar imagen (solo 3 bandas)
            img_rgb = img[..., :3].astype(np.float32) / 255.0
            img_rgb = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD

            img_ready = np.expand_dims(img_rgb, axis=0)

            # Predicción
            pred = model.predict(img_ready, verbose=0)  # (1, H, W, n_classes)
            
            # Máscara final usando argmax
            mask_class = np.argmax(pred[0], axis=-1).astype(np.uint8)

            # Guardar máscara
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="uint8",
                crs=crs,
                transform=transform,
                compress="lzw"
            ) as dst:
                dst.write(mask_class, 1)
                
        except Exception as e:
            print(f"Error procesando {filename}: {str(e)}")
            continue
    
    print(f"\nPredicción completada. Máscaras guardadas en: {output_dir}")



def predict_tiles(input_dir, output_dir, model, threshold=0.5):
    """
    Predice máscaras binarias para todos los tiles TIF en una carpeta.
    
    Args:
        input_dir: Carpeta con los tiles TIF de entrada
        output_dir: Carpeta donde guardar las máscaras predichas
        model: Modelo de segmentación
        threshold: Umbral para binarizar la predicción (default: 0.5)
    """
    # Crear carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener todos los archivos TIF
    tile_paths = glob(os.path.join(input_dir, "*.tif"))
    
    print(f"Encontrados {len(tile_paths)} tiles para predecir")
    
    # Procesar cada tile
    for image_path in tqdm(tile_paths, desc="Prediciendo tiles"):
        # Nombre del archivo de salida
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Leer imagen y metadata geoespacial
            with rasterio.open(image_path) as src:
                img = src.read()          # shape → (bands, H, W)
                img = np.moveaxis(img, 0, -1)  # → (H, W, bands)
                
                # Guardar metadata para la máscara
                transform = src.transform
                crs = src.crs
                height = src.height
                width = src.width
            
            # Preparar imagen para predicción
            img_rgb = img[..., :3] 
            img_rgb = img_rgb.astype("float32") / 255.0
            img_ready = np.expand_dims(img_rgb, axis=0)
            
            # Predecir
            pred = model.predict(img_ready, verbose=0)
            mask_pred = pred[0, :, :, 0]
            
            # Binarizar máscara (0 o 255)
            mask_bin = (mask_pred > threshold).astype("uint8") * 255
            
            # Guardar máscara binaria como TIF georreferenciado
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="uint8",
                crs=crs,
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(mask_bin, 1)
                
        except Exception as e:
            print(f"Error procesando {filename}: {str(e)}")
            continue
    
    print(f"\nPredicción completada. Máscaras guardadas en: {output_dir}")
