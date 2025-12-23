import os
from sklearn.model_selection import train_test_split

def split_dataset(
        carpeta=""):
    """
    Divide las imágenes en train/val/test agrupando por CCPP.
    Devuelve 6 listas:
    - train_imgs, train_masks
    - val_imgs, val_masks
    - test_imgs, test_masks
    """

    # ------------------------------
    # Construir listas finales
    # ------------------------------
    train_imgs = []
    train_imgs = os.listdir(carpeta+"/train/images")

    val_imgs = []
    val_imgs = os.listdir(carpeta+"/val/images")

    test_imgs = []
    test_imgs = os.listdir(carpeta+"/test/images")

    train_masks = []
    train_masks = os.listdir(carpeta+"/train/masks")

    val_masks = []
    val_masks = os.listdir(carpeta+"/val/masks")

    test_masks = []
    test_masks = os.listdir(carpeta+"/test/masks")

    # Agregar EMPTY a train
    #if agregar_empty_a_train:
    #    train_imgs.extend(empty_tiles)

    # ------------------------------
    # Convertir a rutas completas
    # ------------------------------
    train_imgs_full = [os.path.join(carpeta+"/train/images", f) for f in train_imgs]
    val_imgs_full   = [os.path.join(carpeta+"/val/images", f) for f in val_imgs]
    test_imgs_full  = [os.path.join(carpeta+"/test/images", f) for f in test_imgs]

    train_masks_full = [os.path.join(carpeta+"/train/masks", f) for f in train_masks]
    val_masks_full   = [os.path.join(carpeta+"/val/masks", f) for f in val_masks]
    test_masks_full  = [os.path.join(carpeta+"/test/masks", f) for f in test_masks]

    # ------------------------------
    # Verificar existencia de máscaras
    # ------------------------------
    errores = False
    for f in train_masks_full + val_masks_full + test_masks_full:
        if not os.path.exists(f):
            print("ERROR: Máscara no encontrada:", f)
            errores = True

    if errores:
        print("Hay errores. Revisa los nombres o regenera los tiles.")
    else:
        print("\nDivisión correcta sin errores de archivos.")

    # ------------------------------
    # Estadísticas
    # ------------------------------
    print("-" * 50)
    print(f"TRAIN     → imágenes: {len(train_imgs_full)} | máscaras: {len(train_masks_full)}")
    print(f"VALIDATION → imágenes: {len(val_imgs_full)} | máscaras: {len(val_masks_full)}")
    print(f"TEST      → imágenes: {len(test_imgs_full)} | máscaras: {len(test_masks_full)}")
    print("-" * 50)
    print(f"TOTAL IMGS: {len(train_imgs_full) + len(val_imgs_full) + len(test_imgs_full)}")
    print(f"TOTAL MASKS: {len(train_masks_full) + len(val_masks_full) + len(test_masks_full)}")
    print("-" * 50)



    return (
        train_imgs_full, train_masks_full,
        val_imgs_full, val_masks_full,
        test_imgs_full, test_masks_full
    )