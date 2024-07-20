#https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans/torch/attacks/noise.py

import torch
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal

# Common
def load_image(filepath):
    img = Image.open(filepath)
    x = np.array(img) / 255.0  # Normalize the image to [0, 1] range
    x = tf.expand_dims(x, axis=0)
    return x

def attack_visualisation(x, adv_x, true_label, pred, pred_class, adv_pred, adv_class, pattern=None):
    
    print(f"True label: {true_label}")
    print(f"Prediction: {pred}")
    print(f"Prediction class: {pred_class}")
    print(f"Adversarial Prediction: {adv_pred}")
    print(f"Adversarial Predicted Class: {adv_class}")

    if pattern == None:
        plt.subplot(1, 2, 1)
        plt.imshow(x.numpy().squeeze())
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(adv_x.numpy().squeeze())
        plt.title("Adversarial Image")
        plt.axis("off")
    else:
        plt.subplot(1, 3, 1)
        plt.imshow(x.numpy().squeeze())
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.imshow(pattern[0] * 0.5 + 0.5)
        plt.title("Adversarial Pattern")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(adv_x.numpy().squeeze())
        plt.title("Adversarial Image")
        plt.axis("off")  

    plt.tight_layout()
    plt.show()

# Random Noise attack
def make_noise_image(original_image, eps): # esp is less than 1

    eta = np.random.uniform(-eps, eps, original_image.shape).astype(np.float32)
    adv_image = original_image + eta
    adv_image = tf.clip_by_value(adv_image, 0, 1)
    
    return adv_image
    
def random_noise_attack_with_image(model, filepath, true_label, eps):
    
    x = load_image(filepath)
    adv_x = make_noise_image(x, eps)
    
    pred = model.predict(x, verbose=0)
    pred_class = ["1" if p[0] >= 0.5 else "0" for p in pred]

    adv_pred = model.predict(adv_x, verbose=0)
    adv_class = ["1" if p[0] >= 0.5 else "0" for p in adv_pred]
    
    attack_visualisation(x, adv_x, true_label, pred, pred_class, adv_pred, adv_class)
        
def random_noise_attack(model, filepath, true_label, eps):

    x = load_image(filepath)
    adv_x = make_noise_image(x, eps)

    adv_pred = model.predict(adv_x, verbose=0)
    adv_class = ["1" if p[0] >= 0.5 else "0" for p in adv_pred]

    return (adv_pred, adv_class)

# FGSM attack
def create_adversarial_pattern(model, input_image, input_label):
        loss_object = tf.keras.losses.BinaryCrossentropy()
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = model(input_image)
            loss = loss_object(input_label, prediction)
        gradient = tape.gradient(loss, input_image)
        signed_grad = tf.sign(gradient) #[-1, 1]
        return signed_grad

def fgsm_attack_with_image(model, filepath, true_label, eps):

    x = load_image(filepath) # 0-1
    label = tf.convert_to_tensor([float(true_label)]) # list 
    
    perturbations = create_adversarial_pattern(model, x, label)
    adv_x = x + eps * perturbations
    adv_x = tf.clip_by_value(adv_x, 0, 1)

    pred = model.predict(x, verbose=0)
    pred_class = ["1" if p[0] >= 0.5 else "0" for p in pred]
    
    adv_pred = model.predict(adv_x, verbose=0)
    adv_class = ["1" if p[0] >= 0.5 else "0" for p in adv_pred]

    attack_visualisation(x, adv_x, true_label, pred, pred_class, adv_pred, adv_class, pattern=perturbations)

def fgsm_attack(model, filepath, true_label, eps):

    x = load_image(filepath)
    label = tf.convert_to_tensor([float(true_label)]) # list 
    
    perturbations = create_adversarial_pattern(model, x, label)
    adv_x = x + eps * perturbations
    adv_x = tf.clip_by_value(adv_x, 0, 1)

    adv_pred = model.predict(adv_x, verbose=0)
    adv_class = ["1" if p[0] >= 0.5 else "0" for p in adv_pred]
    
    return (adv_pred, adv_class)

# Saleincy map attack
#https://www.kaggle.com/code/kkhandekar/mapping-saliency-in-image-using-tensorflow
# https://usmanr149.github.io/urmlblog/cnn/2020/05/01/Salincy-Maps.html
def compute_saliency_map(model, input_image, input_label, patch_size):
    
    loss_object = tf.keras.losses.BinaryCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        pred = model(input_image)
        loss = loss_object(input_label, pred)
    gradients = tape.gradient(loss, input_image)
    saliency = tf.abs(gradients)

    saliency = saliency.numpy().squeeze()  # shape=(300, 300, 3)
    scharr = np.ones((patch_size, patch_size), dtype=float) / (patch_size * patch_size)
    
    grad1 = signal.convolve2d(saliency[:, :, 0], scharr, boundary='fill', mode='same')
    grad2 = signal.convolve2d(saliency[:, :, 1], scharr, boundary='fill', mode='same')
    grad3 = signal.convolve2d(saliency[:, :, 2], scharr, boundary='fill', mode='same')
    
    grad = np.stack([
        grad1, 
        grad2,
        grad3,
    ])  # shape=(3, 300, 300)
    
    grad = grad.transpose(1, 2, 0)   # shape=(300, 300, 3)
    
    return grad

def calculate_patch_locations(image, patch_size, center_x, center_y, center_c):

    img_array = image.numpy()
    img = Image.fromarray(np.uint8(img_array[0]))
    width, height = img.size
    
    patch_half_size = patch_size // 2
    patch_top = tf.maximum(0, tf.cast(center_y - patch_half_size, dtype=tf.int32))
    if height - patch_top < patch_size:
        err = patch_size - (height - patch_top)
        patch_top = patch_top - err
    patch_bottom = patch_top + patch_size
    patch_left = tf.maximum(0, tf.cast(center_x - patch_half_size, dtype=tf.int32))
    if width - patch_left < patch_size:
        err = patch_size - (width - patch_left)
        patch_left = patch_left - err
    patch_right = patch_left + patch_size
    return (patch_top, patch_bottom, patch_right, patch_left)

def patch_application(image, patch_form, patch_size, eps):

    patch_top = patch_form[0]
    patch_bottom = patch_form[1]
    patch_right = patch_form[2]
    patch_left = patch_form[3]
    
    # Make noise
    noise = np.random.uniform(-eps, eps, (patch_size, patch_size, 3)).astype(np.float32)
    
    image_copy = image.numpy().squeeze().copy()
    image_copy[patch_left:patch_right, patch_top:patch_bottom, :] += noise
    adv_x = tf.expand_dims(image_copy, axis=0)
    return adv_x

def sailency_attack_with_image(model, filepath, true_label, eps, patch_size=10):
    x = load_image(filepath)
    label = tf.convert_to_tensor([float(true_label)]) # list 
    saliency = compute_saliency_map(model, x, label, patch_size)

    location = tf.where(saliency == tf.reduce_max(saliency))
    center_x = location[0][0]
    center_y = location[0][1]
    center_c = location[0][2]

    patch_form = calculate_patch_locations(x, patch_size, center_x, center_y, center_c)
    adv_x = patch_application(x, patch_form, patch_size, eps)
    adv_x = tf.clip_by_value(adv_x, 0, 1)
        
    #print("X")
    #print(x[0, :30])

    #print("ADV_X")
    #print(adv_x[0, :30])

    pred = model.predict(x, verbose=0)
    pred_class = ["1" if p[0] >= 0.5 else "0" for p in pred]

    adv_pred = model.predict(adv_x, verbose=0)
    adv_class = ["1" if p[0] >= 0.5 else "0" for p in adv_pred]

    attack_visualisation(x, adv_x, true_label, pred, pred_class, adv_pred, adv_class)
    

def saliency_attack(model, filepath, true_label, eps, patch_size=10):

    x = load_image(filepath)
    label = tf.convert_to_tensor([float(true_label)]) # list 
    saliency = compute_saliency_map(model, x, label, patch_size)

    location = tf.where(saliency == tf.reduce_max(saliency))
    center_x = location[0][0]
    center_y = location[0][1]
    center_c = location[0][2]

    patch_form = calculate_patch_locations(x, patch_size, center_x, center_y, center_c)
    adv_x = patch_application(x, patch_form, patch_size, eps)
    adv_x = tf.clip_by_value(adv_x, 0, 1)

    adv_pred = model.predict(adv_x, verbose=0)
    adv_class = ["1" if p[0] >= 0.5 else "0" for p in adv_pred]

    return (adv_pred, adv_class)


# Attack success rate
def attack_success_rate(df, attack_method):
    
    df[f"{attack_method} success"] = 0
    df[f"{attack_method} TN success"] = 0
    df[f"{attack_method} FP success"] = 0

    attack_success = (df["pred"] == df["labels"]) & (df["pred"] != df[f"{attack_method} pred class"])
    tn_success = (df["pred"] == df["labels"]) & (df["pred"] == "0") & (df["pred"] != df[f"{attack_method} pred class"])
    fp_success = (df["pred"] == df["labels"]) & (df["pred"] == "1") & (df["pred"] != df[f"{attack_method} pred class"])

    df.loc[attack_success, f"{attack_method} success"] = 1
    df.loc[tn_success, f"{attack_method} TN success"] = 1
    df.loc[fp_success, f"{attack_method} FP success"] = 1

    # Calculate and print success rate
    success_rate = df[f"{attack_method} success"].mean()
    print(f"{attack_method} Success rate: {success_rate}")

    success_rate = df[f"{attack_method} TN success"].mean()
    print(f"{attack_method} TN Success rate: {success_rate}")

    success_rate = df[f"{attack_method} FP success"].mean()
    print(f"{attack_method} FP Success rate: {success_rate}")

    return df