### src/model.py

from config import TRAIN_DIR,IMG_SIZE,BATCH_SIZE,BASE_LEARNING_RATE,EPOCHS,MODEL_DIR
import tensorflow as tf
from pathlib import Path
from tensorflow.python.keras.saving import save


class CNNmodel:
     """  Model for image classification  provides methods for building, traning, evaluating, saving and predicting """
     def __init__(self)-> None:
          self.train_dir=Path(TRAIN_DIR)
          self.img_size=IMG_SIZE
          self.batch_size=BATCH_SIZE
          self.base_learning_rate=BASE_LEARNING_RATE
          self.epochs=EPOCHS
          self.model_dir=Path(MODEL_DIR)
          self.num_classes=None
          self.AUTOTUNE=tf.data.AUTOTUNE

          
          
          self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            #tf.keras.layers.RandomZoom(0.1),
            #tf.keras.layers.RandomContrast(0.1),
            #tf.keras.layers.RandomBrightness(0.1),
        ], name="data_augmentation")


     def build_model(self,num_classes)-> tf.keras.Sequential:
          """Builds the model returns a keras sequential module"""
          base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(self.img_size[0], self.img_size[1], 3))
          base_model.trainable = False
          model = tf.keras.Sequential([
               tf.keras.layers.Rescaling(1/127.5, offset=-1, input_shape=(self.img_size[0], self.img_size[1], 3)),
               self.data_augmentation,
               base_model,
               tf.keras.layers.GlobalAveragePooling2D(),
               tf.keras.layers.Dense(2000,activation='relu'),
               tf.keras.layers.Dense(1000,activation='relu'),

               
               
               tf.keras.layers.Dense(num_classes,activation='softmax')
         ])
        
          return model
     

     def training(self,train_ds,val_ds,model)->tf.keras.callbacks.History:
          """Trains the model and returns the history of the model training"""
           #Data pipeline optimizations
          
          model.compile(
               optimizer=tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate), 
               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 metrics=['accuracy']
                 )
          early_stopping=tf.keras.callbacks.EarlyStopping(
               monitor='val_loss', 
               patience=10, 
               verbose=1, 
               mode='min',
               restore_best_weights=True
               )
          history=model.fit(
               train_ds, 
               validation_data=val_ds,
               epochs=self.epochs,
               callbacks=[early_stopping]
               )

          
          return history 
        
     
     def save_model(self, model)-> None:
          """Converts the model to Tensorflow Lite and Saves the model to the specified path"""

          self.model_dir.mkdir(parents=True, exist_ok=True)
          keras_filepath = self.model_dir / "classifier.keras"
          model.save(keras_filepath)

          model_filepath = self.model_dir / "classifier.tflite"
          tf_lite_model = tf.lite.TFLiteConverter.from_keras_model(model)
          tf_lite_model = tf_lite_model.convert()

          with open(model_filepath, "wb") as f:
            f.write(tf_lite_model)
          
          
     