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
        bs = len(batch_indexes)
        H, W = self.target_size

        X = np.empty((bs, H, W, self.n_channels), dtype=np.float32)
        y = np.empty((bs, H, W, 1), dtype=np.int32)

        for j, i in enumerate(batch_indexes):

            with rasterio.open(self.image_paths[i]) as src:
                img = src.read(
                    out_shape=(src.count, H, W),
                    resampling=rasterio.enums.Resampling.bilinear
                )
                img = np.transpose(img, (1, 2, 0))

            with rasterio.open(self.mask_paths[i]) as src:
                mask = src.read(
                    1,
                    out_shape=(H, W),
                    resampling=rasterio.enums.Resampling.nearest
                )

            mask_mc = np.full_like(mask, 255, dtype=np.int32)
            mask_mc[mask == BACKGROUND] = 0
            mask_mc[mask == ROAD] = 1
            mask_mc[mask == BUILDING] = 2

            img = img.astype(np.float32)

            if img.shape[-1] != self.n_channels:
                img = img[..., :self.n_channels]

            if self.augment:
                augmented = self.augmentation(image=img, mask=mask_mc)
                img = augmented["image"]
                mask_mc = augmented["mask"]

            img *= (1.0 / 255.0)
            img = (img - IMAGENET_MEAN) / IMAGENET_STD

            X[j] = img
            y[j, ..., 0] = mask_mc

        return X, y
