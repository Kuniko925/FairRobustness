#https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans/torch/attacks/noise.py

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class RandomNoiseAttack:
    def __init__(self, model, filepath, true_label):
        self.model = model
        self.filepath = filepath
        self.true_label = true_label

    def load_image(self):
        img = Image.open(self.filepath)
        x = np.array(img) / 255.0  # Normalize the image to [0, 1] range
        return x

    def make_noise_image(self, original_image, eps):
        np.random.seed(12)
        eta = np.random.uniform(-eps, eps, original_image.shape).astype(np.float32)
        adv_image = original_image + eta

        clip_min=None
        clip_max=None
        
        if clip_min is not None or clip_max is not None:
            assert clip_min is not None and clip_max is not None
            adv_image = torch.clamp(adv_image, min=clip_min, max=clip_max)
        return adv_image

    def attack_with_image(self, eps):
        
        x = self.load_image()
        adv_x = self.make_noise_image(x, eps)

        pred, pred_class = self.model.predict(np.expand_dims(x, axis=0))
        adv_pred, adv_pred_class = self.model.predict(np.expand_dims(adv_x, axis=0))
        
        print(f"True label: {self.true_label}")
        print(f"Prediction: {pred}")
        print(f"Prediction class: {pred_class}")
        print(f"Adversarial Prediction: {adv_pred}")
        print(f"Adversarial Predicted Class: {adv_pred_class}")

        plt.subplot(1, 2, 1)
        plt.imshow(x)
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(adv_x)
        plt.title("Adversarial Image with Noise")
        plt.axis("off")
        
        plt.show()
        
    def attack(self, eps):

        x = self.load_image()
        adv_x = self.make_noise_image(x, eps)
            
        pred, pred_class = self.model.predict(np.expand_dims(x, axis=0))
        adv_pred, adv_pred_class = self.model.predict(np.expand_dims(adv_x, axis=0))

        return (pred, pred_class, adv_pred, adv_pred_class)