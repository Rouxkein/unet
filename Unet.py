import keras
import numpy as np
import tensorflow as tf
from keras import layers,models
import numpy
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
import random
from tensorflow.keras.utils import plot_model
########################################################3
img_size=(160,160)
num_classes=3
batch_size=32
input_dir="images/"
target_dir="annotations/trimaps"
# sap xep theo thu tu data
input_img_paths=sorted([
    os.path.join(input_dir,fname)
    for fname in os.listdir(input_dir)
    if fname.endswith(".jpg")
])
target_img_paths=sorted([
    # truy cap file desktop ,thanh phan cua file /destop/file.txt
    # os.path.join(path,name,dinhdang)
    os.path.join(target_dir,fname)
    # os.listdir open a file
    for fname in os.listdir(target_dir)
    if fname.endswith(".png") and not fname.startswith(".")
])
# verify data score
for input_path,target_path in zip(input_img_paths[0:10],target_img_paths[0:10]):
    print(input_path ,"|",target_path)
###############################################################
#vertor and batch,convert picture to x,y img and label

class OxfordPets(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y
#####################################################################
#set aside a validation slip
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256, 512]:

        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)


        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 3, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    for filters in [512 , 256 ,128 ,64 , 32]:
        x=layers.Conv2DTranspose(filters,3,padding="same")(x)
        x=layers.BatchNormalization()(x)
        x=layers.Activation("relu")(x)

        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x=layers.UpSampling2D(2)(x)
        redisual=layers.UpSampling2D(2)(previous_block_activation)
        redisual=layers.Conv2D(filters,1,padding="same")(redisual)
        x=layers.add([x,redisual])
        previous_block_activation=x
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)

    return model
model=get_model(img_size,num_classes)
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",metrics="accuracy")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 15
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
#
# model.summary()
# plot_model(model,to_file="sg.png")
