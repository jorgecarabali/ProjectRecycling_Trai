### src/data/processing.py 

from pathlib import Path
from typing import List 
import tensorflow as tf

class ImageProcessor:
     def __init__(
        self,
        data_dir: str,
      
    


     )->None:
          
        """  Initialization of Image Processor with the configuration values """
        self.data_dir=Path(data_dir)
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Directory not found :{self.data_dir}")
        self.valid_extensions={'.jpg', '.jpeg','.png', '.gif', '.bmp'}

 
        self.valid_files: List[Path]=[]
        self.invalid_files: List[Path]=[]


     def validate_images(self, image_path: Path)->bool:
         """ Args: 
             image_path: Path to the single image
        Returns:
            Whether the single image is valid or not 
         """  
         if image_path.suffix.lower() not in self.valid_extensions:
             return False
         try: 
             single_image=tf.io.read_file(str(image_path))
             image_tensor=tf.image.decode_image(single_image)
             if len(image_tensor.shape)==3 and image_tensor.shape[2]==3:
               return True
             else:
                 return False
         except tf.errors.OpError as e:
             print(f"Corrupted file: {image_path}- Error: {e}")
             return False
         
     def scan_directory(self)->None: 
         """ Scans the directories recursevely and populates the valid and invalid files"""
         self.valid_files.clear()
         self.invalid_files.clear()


         for image_path in self.data_dir.rglob('*'):
             if image_path.is_file():
                 if  self.validate_images(image_path):
                     self.valid_files.append(image_path)
                 else:
                     self.invalid_files.append(image_path)
         print("Scan completed.")  


     def delete_invalid_files(self)-> int:
        """ Deletes invalid files
           Return: number of files sucessfully removed"""
        invalid_count=len(self.invalid_files)
        removed_count=0
        if invalid_count ==0:
         print ("No invalid files to remove")   
         return 0
        
        
        else: 
            for file in self.invalid_files:
                try: 
                    file.unlink()
                    print(f"File removed {file}")
                    removed_count+=1
                except OSError as e:
                    print (f"Error: {e}")
        return removed_count                

                        

