import cv2
import os
import pandas as pd

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