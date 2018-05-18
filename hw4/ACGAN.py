import numpy as np 
from skimage import io
import os
import matplotlib.pyplot as plt
import csv
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import sys

NOISE_DIM= 100
BATCH_SIZE=128


def Discriminator():

    input_image = Input(shape=(64,64,3,))

    d = Conv2D(64*2, (4,4), strides=(2,2), padding='same')(input_image)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization(momentum=0.8)(d)

    
    d = Conv2D(64*2, (4,4), strides=(2,2), padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization(momentum=0.8)(d)
    
    d = Conv2D(64*4, (4,4), strides=(2,2), padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization(momentum=0.8)(d)
    
    d = Conv2D(64*8, (4,4), strides=(2,2), padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = BatchNormalization(momentum=0.8)(d)
    
    d = Conv2D(64*16, (4,4), strides=(2,2), padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Flatten()(d)
    out_layer = Dense(1, activation='sigmoid')(d)    
    output_feature = Dense(1,activation='sigmoid')(d)
    discriminator = Model(input_image, [out_layer,output_feature])
    
    return discriminator


def Generator():
    
    input_noise = Input(shape=(NOISE_DIM,))
    input_feature = Input(shape=(1,))

    m = Concatenate(axis=1)([input_noise,input_feature])
    
    g = Dense(8*8*1024, activation=LeakyReLU(alpha=0.2))(m)
    g = Reshape((8, 8, 1024))(g)
    g = BatchNormalization(momentum=0.8)(g)
    
    g = UpSampling2D()(g)
    g = Conv2D(64*8, (5,5), padding='same')(g)
    # layer = Conv2DTranspose(64*8,(3,3),strides = (2,2),padding = 'same',use_bias = False)(layer)
    g = BatchNormalization(momentum=0.8)(g)
    g = LeakyReLU(alpha=0.2)(g)
    
    g = UpSampling2D()(g)
    g = Conv2D(64*4, (5,5), padding='same')(g)
    # layer = Conv2DTranspose(64*4,(3,3),strides = (2,2),padding = 'same',use_bias = False)(layer)
    g = LeakyReLU(alpha=0.2)(g)
    g = BatchNormalization(momentum=0.8)(g)

    g = UpSampling2D()(g)
    g = Conv2D(64*2, (5,5), padding='same')(g)
    # layer = Conv2DTranspose(64*2,(3,3),strides = (2,2),padding = 'same',use_bias = False)(layer)
    g = LeakyReLU(alpha=0.2)(g)    
    g = BatchNormalization(momentum=0.8)(g)
    
    out_layer = Conv2D(3, (1,1), padding='same', activation='tanh')(g)
    generator = Model([input_noise,input_feature], out_layer)
    
    return generator



OPT = Adam(0.0002,beta_1=0.5)

generator = Generator()
noise_input = Input(shape=(NOISE_DIM,))
feature_input = Input(shape=(1,))
noise_img = generator([noise_input,feature_input])
generator.summary()

discriminator = Discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=OPT, metrics=['accuracy'])
discriminator.summary()


discriminator.trainable = False
        
[valid,feature] = discriminator(noise_img) 
        
total_model = Model([noise_input,feature_input], [valid,feature])    
total_model.compile(loss='binary_crossentropy', optimizer=OPT)
total_model.summary()
total_model.load_weights('./ACGAN.h5py')
k = 10

mid = np.concatenate([np.ones(k),np.zeros(k)],axis=0)

np.random.seed(40)
noise = np.random.normal(0, 1, (50, NOISE_DIM))
noise = noise[[0,4,5,7,22,25,36,41,42,43]]
noise = np.concatenate([noise,noise],axis=0)
generate_images = generator.predict([noise,mid])
generate_images = (generate_images + 1)/2


g = np.zeros((64*2,64*10,3),dtype =float)
for i in range(20):
    g[(i//10)*64:(i//10+1)*64,64*(i%10):64*((i%10)+1),:] = generate_images[i]
io.imsave(os.path.join(sys.argv[2],'fig3_3.jpg'),g)
