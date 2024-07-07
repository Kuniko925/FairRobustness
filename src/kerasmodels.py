import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch # Due to keras version
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CreateGenerator:
    def __init__(self, image_size, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size
        
    def train_generator(self, df):
        gen=ImageDataGenerator(rescale=(1./255), horizontal_flip=True, rotation_range=20, shear_range=0.2, zoom_range = 0.2)
        generator=gen.flow_from_dataframe(df, x_col="filepath", y_col="labels", target_size=self.image_size, 
                                       class_mode="binary", color_mode="rgb", shuffle=True, batch_size=self.batch_size)
        return generator
        
    def valid_generator(self, df):
        gen=ImageDataGenerator(rescale=(1./255))
        generator=gen.flow_from_dataframe(df, x_col="filepath", y_col="labels", target_size=self.image_size,
                                       class_mode="binary", color_mode="rgb", shuffle=False, batch_size=self.batch_size)
        return generator

class CNNModel:
    def __init__(self):
        self.model = keras.models.Sequential()
        
    def complie_and_summary(self):
        
        # Create Model Structure
        self.model.add(keras.layers.Input(shape=[300, 300, 3]))
        self.model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"))
        self.model.add(keras.layers.MaxPooling2D())
        self.model.add(keras.layers.BatchNormalization())
        
        self.model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"))
        self.model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"))
        self.model.add(keras.layers.MaxPooling2D())
        self.model.add(keras.layers.BatchNormalization())
        
        self.model.add(keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"))
        self.model.add(keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"))
        self.model.add(keras.layers.MaxPooling2D())
        self.model.add(keras.layers.BatchNormalization())
        
        self.model.add(keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"))
        self.model.add(keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"))
        self.model.add(keras.layers.MaxPooling2D())
        
        self.model.add(keras.layers.Flatten())
        
        self.model.add(keras.layers.Dropout(rate=0.2))
        self.model.add(keras.layers.Dense(units=256, activation="relu", kernel_initializer="he_normal"))
        self.model.add(keras.layers.BatchNormalization())
        
        self.model.add(keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"))
        self.model.add(keras.layers.BatchNormalization())
        
        self.model.add(keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"))
        self.model.add(keras.layers.BatchNormalization())
        
        self.model.add(keras.layers.Dense(units=32, activation="relu", kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.L1L2()))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Dense(units=1, activation="sigmoid", kernel_initializer="glorot_uniform", name="classifier"))
        
        self.model.compile(Adam(learning_rate= 0.001), loss= "binary_crossentropy", metrics= ["accuracy"])
        self.model.summary()

    def fit_and_visualization(self, epochs, train_gen, valid_gen):
        
        history = self.model.fit(train_gen, epochs=epochs, validation_data=valid_gen)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy
        ax[0].plot(history.history["accuracy"], label="accuracy")
        ax[0].plot(history.history["val_accuracy"], label="val_accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend(loc="lower right")
        ax[0].set_title("Model Accuracy")
        
        # Loss
        ax[1].plot(history.history["loss"], label="loss")
        ax[1].plot(history.history["val_loss"], label="val_loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].legend(loc="upper right")
        ax[1].set_title("Model Loss")
        
        plt.tight_layout()
        plt.show()


    def evaluate_and_predict(self, test_gen):
        test_loss, test_acc = self.model.evaluate(test_gen)
        print("Test Loss", test_loss)
        print("Test Accuracy", test_acc)
        pred = self.model.predict(test_gen, verbose=0)
        pred_class = "1" if pred[0] >= 0.5 else "0"
        return pred[0], pred_class

    def predict(self, image):
        pred = self.model.predict(image, verbose=0)
        pred_class = "1" if pred[0] >= 0.5 else "0"
        return pred[0], pred_class