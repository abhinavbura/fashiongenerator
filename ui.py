import tkinter as tk
from tkinter import messagebox
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import numpy as np

import os
# latent dimension of the random noise
LATENT_DIM = 100 
# weight initializer for G per DCGAN paper
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) 
# number of channels, 1 for gray scale and 3 for color images
CHANNELS = 3 # UPDATED from 1
def build_generator():
    # create a Keras Sequential model 
    model = Sequential(name='generator')

    # prepare for reshape: FC => BN => RN layers, note: input shape defined in the 1st Dense layer  
    model.add(layers.Dense(8 * 8 * 512, input_dim=LATENT_DIM))
    # model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # layers.LeakyReLU(alpha=0.2),

    # 1D => 3D: reshape the output of the previous layer 
    model.add(layers.Reshape((8, 8, 512)))

    # upsample to 16x16: apply a transposed CONV => BN => RELU
    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(layers.BatchNormalization())
    model.add((layers.ReLU()))

    # upsample to 32x32: apply a transposed CONV => BN => RELU
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(layers.BatchNormalization())
    model.add((layers.ReLU()))

    # upsample to 64x64: apply a transposed CONV => BN => RELU
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(layers.BatchNormalization())
    model.add((layers.ReLU()))

    # final layer: Conv2D with tanh activation
    model.add(layers.Conv2D(CHANNELS, (4, 4), padding="same", activation="tanh"))

    # return the generator model
    return model

global generator

def build_model():
    global generator

    generator = build_generator() 
    generator.summary()
    status_label.config(text="Model built successfully")

def load_weights():
    global generator
    generator = load_model('dataset.h5')

def generate_images():
    latent_dim=LATENT_DIM
    generator = load_model('dataset.h5')
    seed = tf.random.normal([16, latent_dim])
    generated_images = generator(seed)
    generated_images = (generated_images * 127.5) + 127.5
    generated_images.numpy()
    num_img=8
    fig = plt.figure(figsize=(4, 4))
    for i in range(num_img):
        plt.subplot(4, 4, i+1)
        img = keras.utils.array_to_img(generated_images[i]) 
        plt.imshow(img)
        plt.axis('off')
    plt.savefig('test.png') 
    status_label.config(text="Images generated successfully")
    generated_image_path="test.png"
    generated_image = Image.open(generated_image_path)

    # Upscale the image to desired dimensions (e.g., 128x128)
    upscaled_image = generated_image.resize((250, 250), Image.BICUBIC)

    # Convert the upscaled image to a format compatible with Tkinter
    tk_generated_image = ImageTk.PhotoImage(upscaled_image)

    # Update the placeholder with the upscaled image
    placeholder.config(image=tk_generated_image)
    placeholder.image = tk_generated_image
    # Load the generated image using PIL
    # generated_image = Image.open(generated_image_path)

    # Resize the image to fit in the placeholder
    
    status_label.config(text="Images generated successfully")

def exit_app():
    root.destroy()

root = tk.Tk()
root.title("Model Builder")
root.geometry("500x300")
root.configure(bg='sky blue')

left_frame = tk.Frame(root, bg='sky blue')
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

button_build_model = tk.Button(left_frame, text="Build Model", command=build_model)
button_build_model.pack(fill=tk.X, padx=10, pady=5)

button_load_model = tk.Button(left_frame, text="Load Model", command=load_weights)
button_load_model.pack(fill=tk.X, padx=10, pady=5)

button_generate_images = tk.Button(left_frame, text="Generate Images", command=generate_images)
button_generate_images.pack(fill=tk.X, padx=10, pady=5)

button_exit = tk.Button(left_frame, text="Exit", command=exit_app)
button_exit.pack(fill=tk.X, padx=10, pady=5)

status_label = tk.Label(root, text="Status: Ready", bg='sky blue', fg='black')
status_label.pack(pady=10)

# Placeholder for displaying generated images
placeholder = tk.Label(root, text="Generated Images Placeholder", bg='white', fg='black', width=250, height=250)
placeholder.pack(pady=10)

root.mainloop()
