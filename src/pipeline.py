### src/pipeline.py
import yaml 
import os 
import matplotlib.pyplot as plt
from data_processing import ImageProcessor
from model import CNNmodel
from dataset_manager import Dataset
import tensorflow as tf 
from PIL import Image
import numpy as np

from config import MODEL_DIR,TRAIN_DIR,TEST_DIR_CARDBOARD,MODEL_PATH,TEST_DIR_PLASTIC,TEST_DIR_PAPER,TEST_DIR_EWASTE

class ClassificationPipeline:
    """ Pipeline for image classification  orchestrates the preprocessing, training and evaluation of the model"""
    def __init__(self):
      
        self.model_builder = CNNmodel()
        self.dataset_builder=Dataset()
        self.model_dir= MODEL_DIR
        
        
        
        self.train_ds=None
        self.val_ds=None
        self.history=None
        self.number_of_classes=None
        self.model=None
        self.class_names=None
        
    def preprocess_images(self):
        """run the image/data processing pipeline. Loads the traning and validate the images and populates the valid and invalid files""" 
        print("scanning directory...")
        self.image_processor = ImageProcessor(TRAIN_DIR)
        self.image_processor.scan_directory()
        self.image_processor.delete_invalid_files()

    def train_model(self):
        """Trains the model and saves the model to the specified path"""
        self.train_ds, self.val_ds, self.number_of_classes,self.class_names = self.dataset_builder.dataset()
        
        self.model= self.model_builder.build_model(self.number_of_classes)

        self.history=self.model_builder.training(
            train_ds=self.train_ds,
            val_ds=self.val_ds,
            model=self.model)
        
        self.model_builder.save_model(self.model)
        print(f"Model saved to {self.model_dir}")
        print("Training Completed")
        self.plot_results()

        class_names_path = os.path.join(self.model_dir, "class_names.yaml")
        try:
            with open(class_names_path, 'w') as f:
                yaml.dump(self.class_names, f)  
            print(f"Class names saved to {class_names_path}")
        except Exception as e:
            print(f"Error saving class names: {e}")
       
       
    def inference(self)->dict:
        """Runs inference on the test dataset and returns the predicted class"""
        self.image_processor=ImageProcessor(TEST_DIR_PLASTIC)
        self.image_processor.scan_directory()
        self.image_processor.delete_invalid_files()
        self.model = tf.keras.models.load_model(MODEL_PATH)
        class_names_path = os.path.join(self.model_dir, "class_names.yaml")
        
        try:
            with open(class_names_path, 'r') as f:
                # Use yaml.safe_load() for security when loading
                class_names = yaml.safe_load(f) 
        except FileNotFoundError:
            print(f"Error: 'class_names.yaml' not found in {self.model_dir}.")
            raise

        all_prediction=[]
        percentage_each_class=dict()
        for image_path in self.image_processor.valid_files:
         try: 
            img=Image.open(image_path)
            img=img.convert('RGB')
            img=img.resize((self.model_builder.img_size[0], self.model_builder.img_size[1]))
            img_array=np.array(img)
            img_batch=tf.expand_dims(img_array,0)
       

            predictions=self.model.predict(img_batch)
            predicted_index=tf.argmax(predictions[0]).numpy()
            predicted_text=class_names[predicted_index]
            all_prediction.append(predicted_text)

         except OSError as e:
            print(f"Corrupted file: {image_path}- Error: {e}")

        try:
            for i in range(len(class_names)):
                percentage_each_class.update({class_names[i]:all_prediction.count(class_names[i])/len(all_prediction)*100})
        
        except OSError as e:
            print(f"Error: {e}")

        

        return percentage_each_class
        


        
        


        

    def plot_results(self)-> None:
          """Plots the training and validation accuracy and loss"""
          

          if not self.history:
           print("Model has not been trained yet. Please train model")
           return
          

          print("Plotting results...")
          acc = self.history.history['accuracy']
          val_acc = self.history.history['val_accuracy']
          loss = self.history.history['loss']
          val_loss = self.history.history['val_loss']
          epochs = range(1, len(acc) + 1)
          plt.plot(epochs, acc, 'r', label='Training acc')
          plt.plot(epochs, val_acc, 'b', label='Validation acc')
          plt.title('Training and validation accuracy')
          plt.legend()
          plt.figure()
          plt.plot(epochs, loss, 'r', label='Training loss')
          plt.plot(epochs, val_loss, 'b', label='Validation loss')
          plt.title('Training and validation loss')
          plt.legend()
          plt.show()


         

        
        




