import numpy as np 
import tensorflow as tf
from skimage import io
import os
import glob as gb
import scipy.misc as sio
import random
import matplotlib.pyplot as plt
import csv
from keras.layers import *
from keras.models import *
from keras.optimizers import *


NOISE_DIM= 100




train_data = np.load(r"H:\master\code\python\DLCV_HW4\image\x_train.npy")
test_data = np.load(r"H:\master\code\python\DLCV_HW4\image\x_test.npy")


feature_train = np.zeros([len(train_data)]) 


feature_test = np.zeros([len(test_data)]) 


data = np.concatenate([train_data,test_data],axis=0)
data /= 255
data -=0.5
data *=2



feature_path = r'H:\master\code\python\DLCV_HW4\hw4_data\train.csv'
with open(feature_path, 'r',newline='\n') as csvfile:
    a= csv.reader(csvfile,delimiter=',')
    k=-1
    for i in a:
        if(k>-1):
            feature_train[k]=float(i[10])
        k+=1


feature_path = r'H:\master\code\python\DLCV_HW4\hw4_data\test.csv'
with open(feature_path, 'r',newline='\n') as csvfile:
    a= csv.reader(csvfile,delimiter=',')
    k=-1
    for i in a:
        if(k>-1):
            feature_test[k]=float(i[10])
        k+=1


all_feature = np.concatenate([feature_train,feature_test],axis=0)




def Discriminator():

    input_image = Input(shape=(64,64,3,))

    layer = Conv2D(64*2, (4,4), strides=(2,2), padding='same')(input_image)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = BatchNormalization(momentum=0.8)(layer)

    
    layer = Conv2D(64*2, (4,4), strides=(2,2), padding='same')(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = BatchNormalization(momentum=0.8)(layer)
    
    layer = Conv2D(64*4, (4,4), strides=(2,2), padding='same')(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = BatchNormalization(momentum=0.8)(layer)
    
    layer = Conv2D(64*8, (4,4), strides=(2,2), padding='same')(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = BatchNormalization(momentum=0.8)(layer)
    
    layer = Conv2D(64*16, (4,4), strides=(2,2), padding='same')(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    layer = Flatten()(layer)
    out_layer = Dense(1, activation='sigmoid')(layer)    
    output_feature = Dense(1,activation='sigmoid')(layer)
    discriminator = Model(input_image, [out_layer,output_feature])
    
    return discriminator


def Generator():
    
    input_noise = Input(shape=(NOISE_DIM,))
    input_feature = Input(shape=(1,))

    merge = Concatenate(axis=1)([input_noise,input_feature])
    
    layer = Dense(8*8*1024, activation=LeakyReLU(alpha=0.2))(merge)
    layer = Reshape((8, 8, 1024))(layer)
    layer = BatchNormalization(momentum=0.8)(layer)
    
    # block
    layer = UpSampling2D()(layer)
    layer = Conv2D(64*8, (5,5), padding='same')(layer)
    # layer = Conv2DTranspose(64*8,(3,3),strides = (2,2),padding = 'same',use_bias = False)(layer)
    layer = BatchNormalization(momentum=0.8)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    
    # block
    layer = UpSampling2D()(layer)
    layer = Conv2D(64*4, (5,5), padding='same')(layer)
    # layer = Conv2DTranspose(64*4,(3,3),strides = (2,2),padding = 'same',use_bias = False)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = BatchNormalization(momentum=0.8)(layer)
    
    # block
    layer = UpSampling2D()(layer)
    layer = Conv2D(64*2, (5,5), padding='same')(layer)
    # layer = Conv2DTranspose(64*2,(3,3),strides = (2,2),padding = 'same',use_bias = False)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)    
    layer = BatchNormalization(momentum=0.8)(layer)
    
    # 3是因為RGB有3個channel
    out_layer = Conv2D(3, (1,1), padding='same', activation='tanh')(layer)
    
    generator = Model([input_noise,input_feature], out_layer)
    
    return generator



OPT = Adam(0.0002,beta_1=0.5)

generator = Generator()
noise_input = Input(shape=(NOISE_DIM,))
feature_input = Input(shape=(1,))
noise_img = generator([noise_input,feature_input])    #generator產生的圖片
generator.summary()

discriminator = Discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=OPT, metrics=['accuracy'])
discriminator.summary()


discriminator.trainable = False
        
[valid,feature] = discriminator(noise_img) 
        
total_model = Model([noise_input,feature_input], [valid,feature])    
total_model.compile(loss='binary_crossentropy', optimizer=OPT)
total_model.summary()

BATCH_SIZE=128
ones = np.ones((BATCH_SIZE,1))
zeros = np.zeros((BATCH_SIZE,1))
g_loss_log = []
d_loss_log = []


noise_set = np.random.normal(0, 1, (30000, NOISE_DIM))



BATCH_SIZE = 128
ones = np.ones(BATCH_SIZE)
zeros =np.zeros(BATCH_SIZE)
shuffle= np.arange(len(data))
mid = np.concatenate([np.ones(10),np.zeros(10)],axis=0)
rand = np.concatenate([np.ones(BATCH_SIZE//2),np.zeros(BATCH_SIZE//2)],axis=0)
for epoch in range(20000):
    # 丟noise給generator，產生圖片
    noise = np.random.normal(0, 1, (BATCH_SIZE//2, NOISE_DIM))
    noise = np.concatenate([noise,noise],axis=0)
    np.random.shuffle(rand)
    generate_images = generator.predict([noise,rand])
    np.random.shuffle(shuffle)
    # 單獨train descriminator，ones、zeros代表跟他說是真的還是假的
    a = np.random.randint(data.shape[0], size=BATCH_SIZE)
    b = np.random.randint(data.shape[0], size=BATCH_SIZE)
    
    d_loss_real = discriminator.train_on_batch(data[a], [ones,all_feature[a]])
    d_loss_real = discriminator.test_on_batch(data[a], [ones,all_feature[a]])
    d_loss_fake = discriminator.train_on_batch(generate_images, [zeros,rand])
    d_loss_fake = discriminator.test_on_batch(generate_images, [zeros,rand])
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # 對generator來說，希望他輸入noise
    # 但是經過discriminator之後可以被判斷維是真的
    rand = np.random.randint(0, 2, BATCH_SIZE)
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    g_loss = total_model.train_on_batch([noise,rand], [ones,rand])
    g_loss = total_model.test_on_batch([noise,rand], [ones,rand])
    #g_loss = total_model.test_on_batch([noise,rand], [ones,rand])
    # Plot the progress
    print ("epoch %d [D loss: %f,feature loss: %f, acc.: %.2f%%] [G loss: %f,feature loss: %f ]" % (epoch, d_loss[0],d_loss[1]
                                                                                                    , 100*d_loss[2], g_loss[0]
                                                                                                    ,g_loss[1]))
    noise = np.random.normal(0, 1, (10, NOISE_DIM))
    noise = np.concatenate([noise,noise],axis=0)
    generate_images = generator.predict([noise,mid])
    # np.random.uniform()
    generate_images = (generate_images + 1)/2
    g_loss_log.append(g_loss)
    d_loss_log.append(d_loss)
    if epoch % 50 ==0:
        plt.figure(figsize=(20,10))
        for i in range(20):
            plt.subplot(2,10,i+1)
            plt.axis("off")
            plt.imshow(generate_images[i])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0, hspace=0)
        plt.savefig('H:/master/code/python/DLCV_HW4/ACGAN/img/fig2_%d.jpg'%(epoch))
    plt.clf()
    if epoch % 100 ==0:
        total_model.save_weights('H:/master/code/python/DLCV_HW4/ACGAN/model/gan_{:05d}.h5py'.format(epoch))
        
np.save('H:/master/code/python/DLCV_HW4/ACGAN/model/g_loss_log.npy',g_loss_log)
np.save('H:/master/code/python/DLCV_HW4/ACGAN/model/d_loss_log.npy',d_loss_log)

