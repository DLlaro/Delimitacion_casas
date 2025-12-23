import numpy as np
import tensorflow as tf
import os
import math
import cv2
import rasterio
import albumentations as A

# -------- CONSTANTES IMAGENET --------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# class values
NOINFO_VALUE = 0   # clase 0 = no información
BACKGROUND = 1
BUILDING = 255
ROAD = 125

class TIFFDataGeneratorAug(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths,
                 batch_size=2, shuffle=True,
                 normalize='imagenet',
                 target_size=(512, 512),
                 n_channels=3,
                 augment=False):
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.target_size = target_size
        self.n_channels = n_channels
        self.augment = augment
        
        # -------- DATA AUGMENTATION --------
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.05,
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4
            ),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(var_limit=(3, 10), p=0.1)
        ], additional_targets={'mask': 'mask'})
        
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):

        batch_indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_imgs, batch_masks = [], []

        for i in batch_indexes:

            # -------- LEER IMAGEN --------
            with rasterio.open(self.image_paths[i]) as src:
                img = src.read()  # (C, H, W)
                img = np.transpose(img, (1, 2, 0))  # (H, W, C)

            # -------- LEER MÁSCARA --------
            with rasterio.open(self.mask_paths[i]) as src:
                mask = src.read(1)

                mask_mc = np.full_like(mask, 255, dtype=np.uint8)
                # opcional: ignorar nodata
                mask_mc[mask == BACKGROUND] = 0
                mask_mc[mask == ROAD] = 1
                mask_mc[mask == BUILDING] = 2 

            img = img.astype(np.float32)
            #Reasignacion de mask (entero ya que es multiclase)
            mask = mask_mc.astype(np.uint8)

            # -------- AJUSTE DEL NÚMERO DE CANALES --------
            if img.shape[-1] > self.n_channels:
                img = img[..., :self.n_channels]
            elif img.shape[-1] < self.n_channels:
                img = np.concatenate([img] * (self.n_channels // img.shape[-1] + 1), axis=-1)
                img = img[..., :self.n_channels]

            # -------- REDIMENSIONAR --------
            if img.shape[:2] != self.target_size:
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
            if mask.shape[:2] != self.target_size:
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

            # -------- APLICAR DATA AUGMENTATION --------
            if self.augment:
                augmented = self.augmentation(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]

            # -------- NORMALIZACIÓN --------
            if self.normalize == 'imagenet':
                # Normalización ImageNet
                img = img / 255.0  # Escalar a [0, 1]
                img = (img - IMAGENET_MEAN) / IMAGENET_STD

            # -------- FORMATO FINAL DE LA MÁSCARA --------
            mask = np.expand_dims(mask, axis=-1)

            batch_imgs.append(img)
            batch_masks.append(mask)

        X = np.stack(batch_imgs, axis=0)
        y = np.stack(batch_masks, axis=0)

        print(X.shape)  # DEBUG
        return X, y