#src/dataset_manager.py

from config import TRAIN_DIR,IMG_SIZE,BATCH_SIZE,BASE_LEARNING_RATE,EPOCHS,MODEL_DIR,VALIDATION_SPLIT
import tensorflow as tf
from pathlib import Path



class Dataset:
    """ Dataset manager class that returns a dataset of images and labels for training and validation and the number of classes in the dataset"""
    def __init__(self):
        self.train_dir=Path(TRAIN_DIR)
        self.img_size=IMG_SIZE
        self.batch_size=BATCH_SIZE
        self.base_learning_rate=BASE_LEARNING_RATE
        self.validation_split=VALIDATION_SPLIT
        self.epochs=EPOCHS
        self.model_dir=Path(MODEL_DIR)
        self.num_classes=None
        self.AUTOTUNE=tf.data.AUTOTUNE


    def dataset(self)-> tf.keras.utils.image_dataset_from_directory:
          """Returns a dataset of images and labels for training and validation and the number of classes in the dataset"""
          train_ds = tf.keras.utils.image_dataset_from_directory(
               self.train_dir,
               validation_split=self.validation_split,
               subset="training",
               seed=123,
               image_size=(self.img_size [0], self.img_size[1]),
               batch_size=self.batch_size,
               color_mode="rgb",
               shuffle=True   
          )
          val_ds = tf.keras.utils.image_dataset_from_directory(
               self.train_dir,
               validation_split=self.validation_split,
               subset="validation",
               seed=123,
               image_size=(self.img_size [0], self.img_size[1]),
               batch_size=self.batch_size,
               color_mode="rgb"
          )
          num_classes=len(train_ds.class_names)
          print(f"Found {num_classes} classes, classes: {train_ds.class_names}")
          class_names=train_ds.class_names
          train_ds = train_ds.cache().prefetch(buffer_size=self.AUTOTUNE)
          val_ds = val_ds.cache().prefetch(buffer_size=self.AUTOTUNE)

          return train_ds, val_ds, num_classes,class_names