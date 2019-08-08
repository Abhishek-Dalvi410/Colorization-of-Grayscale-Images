from keras.layers import Conv2D, UpSampling2D, Input, MaxPooling2D
from keras.layers import Activation, InputLayer, concatenate, Reshape
from keras.layers.core import RepeatVector
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.io import imsave
from skimage.transform import resize
import numpy as np
import os
import random
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
# from tqdm import tqdm

DATASET = "ifw(128)"
DATADIR = "../input/"+DATASET+"/"+DATASET+"/train/"
X = []
IMG_SIZE = 128
# TOTAL_IMGS = 4722
TOTAL_IMGS = 10000
i=0
for filename in os.listdir(DATADIR):
  i=i+1
  if i > TOTAL_IMGS:
    break
  try:
    X.append(img_to_array(load_img(DATADIR + filename)))
  except Exception as e:
    pass

X = np.array(X, dtype=float)
split = int(0.90*TOTAL_IMGS)

Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain

Xtest = X[split:]
Xtest = 1.0/255*Xtest

# model architecture
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1,))

#Encoder1
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(input_img)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

#Encoder2
encoder_output2 = Conv2D(64, (3,3), activation='relu', padding='same')(input_img)
encoder_output2 = MaxPooling2D((2, 2), padding='same')(encoder_output2)
encoder_output2 = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output2)
encoder_output2 = MaxPooling2D((2, 2), padding='same')(encoder_output2)
encoder_output2 = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output2)
encoder_output2 = MaxPooling2D((2, 2), padding='same')(encoder_output2)
encoder_output2 = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output2)
encoder_output2 = MaxPooling2D((2, 2), padding='same')(encoder_output2)
encoder_output2 = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output2)
encoder_output2 = MaxPooling2D((2, 2), padding='same')(encoder_output2)
encoder_output2 = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output2)
encoder_output2 = MaxPooling2D((2, 2), padding='same')(encoder_output2)
encoder_output2 = MaxPooling2D((2, 2), padding='same')(encoder_output2)
encoder_output2 = Conv2D(512, (3,3), activation='softmax', padding='same')(encoder_output2)
encoder2=(Reshape((512,), input_shape=(1,1,512)))(encoder_output2)

#Fusion1
fusion_output = RepeatVector(16 * 16)(encoder2)  
fusion_output = Reshape(([16, 16, 512]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
fusion_output = Conv2D(512, (1, 1), activation='relu', padding='same')(fusion_output) 

#Decoder
decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=input_img, outputs=decoder_output)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()

#model check point
checkpointer = ModelCheckpoint(filepath="best.h5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

# Image transformer
datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=True)

# Generate training data
def train_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)
        
def val_gen(batch_size):
    for batch in datagen.flow(Xtest, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_val_batch = lab_batch[:,:,:,0]
        Y_val_batch = lab_batch[:,:,:,1:] / 128
        yield (X_val_batch.reshape(X_val_batch.shape+(1,)), Y_val_batch)

# Train model 
EPOCHS = 150
batch_size = 30
history = model.fit_generator(train_gen(batch_size), 
                              shuffle=True, 
                              steps_per_epoch=split//batch_size, 
                              validation_data=val_gen(batch_size), 
                              validation_steps=(TOTAL_IMGS-split)//batch_size,
                              callbacks=[checkpointer],
                              epochs=EPOCHS, verbose=2)

# #validate
# Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
# Xtest = Xtest.reshape(Xtest.shape+(1,))
# Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
# Ytest = Ytest / 128
# print(model.evaluate(Xtest, Ytest, batch_size=batch_size))

TEST_DATADIR = "../input/"+DATASET+"/"+DATASET+"/test/"
color_me = []
for filename in os.listdir(TEST_DATADIR):
  try:
    color_me.append(img_to_array(load_img(TEST_DATADIR + filename)))
  except Exception as e:
    pass

color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

# Test model
output = model.predict(color_me)
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((IMG_SIZE, IMG_SIZE, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    res_img = lab2rgb(cur)
    imsave("img_result"+str(i)+".jpg", lab2rgb(cur))

# plot graphs  
def  plot_loss_acc(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(EPOCHS)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    filename= 'loss.png'
    plt.savefig(filename)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(EPOCHS)
    plt.figure()
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'g')
    plt.title('Training and validation accuracy')
    plt.legend(['train', 'val'], loc='lower right')
    filename= 'accuracy.png'
    plt.savefig(filename)

plot_loss_acc(history)