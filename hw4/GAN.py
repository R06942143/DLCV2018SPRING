import matplotlib
matplotlib.use('Agg')
import numpy as np 
from skimage import io
import os
import matplotlib.pyplot as plt
import math
from keras.initializers import RandomNormal
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras import regularizers
import sys
NOISE_DIM= 100
BATCH_SIZE=128




def Discriminator():
    input_image = Input(shape=(64,64,3,))

    d = Conv2D(64*2, (3,3), strides=(2,2), padding='same')(input_image)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization(momentum=0.8)(d)

    d = Conv2D(64*2, (3,3), strides=(2,2), padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization(momentum=0.8)(d)
    
    d = Conv2D(64*4, (3,3), strides=(2,2), padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization(momentum=0.8)(d)
    
    d = Conv2D(64*8, (3,3), strides=(2,2), padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization(momentum=0.8)(d)
    
    d = Conv2D(64*16, (3,3), strides=(2,2), padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Flatten()(d)
    out_layer = Dense(1, activation='sigmoid')(d)    
    
    discriminator = Model(input_image, out_layer)
    return discriminator



def Generator():
    
    input_noise = Input(shape=(NOISE_DIM,))
    g = Dense(4*4*1024, activation=LeakyReLU(alpha=0.2))(input_noise)
    g = Reshape((4, 4, 1024))(g)
    g = BatchNormalization(momentum=0.8)(g)
    
    g = UpSampling2D()(g)
    g = Conv2D(64*8, (5,5), padding='same')(g)
    # layer = Conv2DTranspose(64*8,(3,3),strides = (2,2),padding = 'same',use_bias = False)(layer)
    g = BatchNormalization(momentum=0.8)(g)
    g = LeakyReLU(alpha=0.2)(g)
    
    g = UpSampling2D()(g)
    g = Conv2D(64*4, (5,5), padding='same')(g)
    # layer = Conv2DTranspose(64*4,(3,3),strides = (2,2),padding = 'same',use_bias = False)(layer)
    g = BatchNormalization(momentum=0.8)(g)
    g = LeakyReLU(alpha=0.2)(g)
    
    g = UpSampling2D()(g)
    g = Conv2D(64*2, (5,5), padding='same')(g)
    # layer = Conv2DTranspose(64*2,(3,3),strides = (2,2),padding = 'same',use_bias = False)(layer)
    g = BatchNormalization(momentum=0.8)(g)
    g = LeakyReLU(alpha=0.2)(g)
    
    g = UpSampling2D()(g)
    out_layer = Conv2D(3, (5,5), padding='same', activation='tanh')(g)
    # out_layer = Conv2DTranspose(3,(3,3),strides = (2,2),padding = 'same', activation='tanh',use_bias = False)(layer)
    generator = Model(input_noise, out_layer)
    return generator


adam = RMSprop(0.0001)
generator = Generator()
noise_input = Input(shape=(NOISE_DIM,))
noise_img = generator(noise_input) 

sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
discriminator = Discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
discriminator.summary()

discriminator.trainable = False
        
valid = discriminator(noise_img) 

total_model = Model(noise_input, valid)    
total_model.compile(loss='binary_crossentropy', optimizer=adam)
total_model.summary()
total_model.load_weights('./GAN.h5py')



np.random.seed(40)
noise2 = np.random.normal(0, 1, (100, NOISE_DIM))
np.random.seed(26)
noise1 = np.random.normal(0, 1, (100, NOISE_DIM))
noise2 = noise2[[5,6,9,12,22,23,29,30,31,40,44,52,77,89,92,57]]
noise1 = noise1[[0,9,16,31,45,53,70,71,87,90,94,95,14,72,32,7]]
noise = np.concatenate((noise1,noise2),axis = 0)
generate_images = generator.predict(noise)
generate_images = (generate_images + 1)/2
plt.figure(figsize=(10,5))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.axis("off")
    plt.imshow(generate_images[i])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0, hspace=0)
    plt.savefig(os.path.join(sys.argv[2],'fig2_3.jpg'))
plt.clf()
