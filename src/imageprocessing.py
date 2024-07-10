import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image

class ImageResize:
    def __init__(self, input_folder, output_folder, new_size=(300, 300)):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.new_size = new_size
    def resize(self):
        for filename in os.listdir(self.input_folder):
            input_path = os.path.join(self.input_folder, filename)
            output_path = os.path.join(self.output_folder, filename)
    
            img = cv2.imread(input_path)
            img_resized = cv2.resize(img, self.new_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, img_resized)

        print("Completed.")

# It is for HAM10000 dataset
# Create masked image from ooriginal image and segumentation image
# Original images are from dataframe["filepath"]
class MaskedImage:
    def __init__(self, df, seg_folder, output_folder):
        self.df = df
        self.seg_folder = seg_folder
        self.output_folder = output_folder
    def create(self):
        for _, row in self.df.iterrows():
            filepath = row["filepath"]
            maskpath = f"{self.seg_folder}{row['image_id']}_segmentation.png"
        
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
            skin = (mask == 0).astype(np.uint8) # 0:black, 1: white -> To change black: 1 means active
            masked_image = cv2.bitwise_and(image, image, mask=skin)
        
            new_filepath = f"{self.output_folder}{row['image_id']}.jpg"
            masked_image = Image.fromarray(masked_image)
            masked_image.save(new_filepath)
        
        print("Completed")