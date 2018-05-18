import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.losses import *
from keras.callbacks import *
from keras import metrics
from keras import optimizers
#######  https://github.com/keras-team/keras/issues/5916
#######  https://github.com/keras-team/keras/issues/9459
# def mean_squared_error(y_true, y_pred):
#     return K.mean(K.square(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)
# def kl_loss(z_l,z_m):
#     kl_loss = - 0.5 * K.sum(1 + z_l - K.square(z_m) - K.exp(z_l), axis=-1)#KL-Divergence Loss
#     return kl_loss
# def vae_loss(y_true, y_pred):
#     M = mean_squared_error(y_true,y_pred)
#     k = kl_loss(z_log_var,z_mean)#KL-Divergence Loss
#     return K.mean(M + k)


batch_size = 64
latent_dim = 512
img_row,img_col,img_channel = 64,64,3
intermediate_dim = latent_dim
epsilon_std = 1.0
epochs = 100
f1 = 3
f2 = 16
f3 = 32
f4 = 64
f5 = 128
f6 = 256
lamdba = 0.00001


x = Input(shape=(img_row,img_col,img_channel))

ll = LeakyReLU(alpha=0.01)
conv_1 = Conv2D(f1,kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
B_1 = BatchNormalization()(conv_1)
A_1 =  LeakyReLU(alpha=0.01)(B_1)

conv_2 = Conv2D(f2,kernel_size=(4, 4), strides=(2, 2), padding='same')(A_1)
B_2 = BatchNormalization()(conv_2)
A_2 =  LeakyReLU(alpha=0.01)(B_2)

conv_3 = Conv2D(f3,kernel_size=(4, 4), strides=(2, 2), padding='same')(A_2)
B_3 = BatchNormalization()(conv_3)
A_3 =  LeakyReLU(alpha=0.01)(B_3)

conv_4 = Conv2D(f4,kernel_size=(4, 4), strides=(2, 2), padding='same')(A_3)
B_4 = BatchNormalization()(conv_4)
A_4 =  LeakyReLU(alpha=0.01)(B_4)

conv_5 = Conv2D(f5,kernel_size=(4, 4), strides=(2, 2), padding='same')(A_4)
B_5 = BatchNormalization()(conv_5)
A_5 =  LeakyReLU(alpha=0.01)(B_5)

conv_6 = Conv2D(f6,kernel_size=(4, 4), strides=(2, 2), padding='same')(A_5)
B_6 = BatchNormalization()(conv_6)
A_6 =  LeakyReLU(alpha=0.01)(B_6)

flat = Flatten()(A_6)
hidden = Dense(intermediate_dim)(flat)
hidden_1 = BatchNormalization()(hidden)
hidden_2 =  LeakyReLU(alpha=0.01)(hidden_1)


z_mean = Dense(latent_dim)(hidden_2)
z_mean_1 = BatchNormalization()(z_mean)
z_mean_2 =  LeakyReLU(alpha=0.01)(z_mean_1)

z_log_var = Dense(latent_dim)(hidden_2)
z_log_var_1 = BatchNormalization()(z_log_var)
z_log_var_2 =  LeakyReLU(alpha=0.01)(z_log_var_1)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
z = Lambda(sampling)([z_mean_2, z_log_var_2])
def see(x,y):
    return K.mean(z)
decoder_hid = Dense(intermediate_dim)
decoder_hid_1 = BatchNormalization()
decoder_hid_2 =  LeakyReLU(alpha=0.01)


output_shape = (batch_size, 1, 1, f6)
decoder_upsample = Dense(f6)
decoder_upsample_1 = BatchNormalization()
decoder_upsample_2 =  LeakyReLU(alpha=0.01)
decoder_reshape  = Reshape(output_shape[1:])



de_conv_1 = Conv2DTranspose(f6,kernel_size=(4, 4), strides=(2, 2), padding='same')
B_7 = BatchNormalization()
A_7 =(LeakyReLU(alpha=0.01))

de_conv_2 = Conv2DTranspose(f5,kernel_size=(4, 4), strides=(2, 2), padding='same')
B_8 = BatchNormalization()
A_8 =(LeakyReLU(alpha=0.01))


de_conv_3 = Conv2DTranspose(f4,kernel_size=(4, 4), strides=(2, 2), padding='same')
B_9 = BatchNormalization()
A_9 =(LeakyReLU(alpha=0.01))

de_conv_4 = Conv2DTranspose(f3,kernel_size=(4, 4), strides=(2, 2), padding='same')
B_10 = BatchNormalization()
A_10 = (LeakyReLU(alpha=0.01))

de_conv_5 = Conv2DTranspose(f2,kernel_size=(4, 4), strides=(2, 2), padding='same')
B_11 = BatchNormalization()
A_11 = (LeakyReLU(alpha=0.01))

de_conv_6 = Conv2DTranspose(f1,kernel_size=(4, 4), strides=(2, 2), padding='same')
x_decoded_mean_squash = Activation('tanh')

d_h = decoder_hid(z)
d_h_1 = decoder_hid_1(d_h)
d_h_2 = decoder_hid_2(d_h_1)

d_u = decoder_upsample(d_h_2)
d_u_1 = decoder_upsample_1(d_u)
d_u_2 = decoder_upsample_2(d_u_1)

d_r = decoder_reshape(d_u_2)

d_1 = de_conv_1(d_r)
b7 = B_7(d_1)
a7 = A_7(b7)

d_2 = de_conv_2(a7)
b8 = B_8(d_2)
a8 = A_8(b8)

d_3 = de_conv_3(a8)
b9 = B_9(d_3)
a9 = A_9(b9)

d_4 = de_conv_4(a9)
b10 = B_10(d_4)
a10 = A_10(b10)

d_5 = de_conv_5(a10)
b11 = B_11(d_5)
a11 = A_11(b11)

d_6 = de_conv_6(a11)

y = x_decoded_mean_squash(d_6)

vae = Model(x,y)
vae_monitor = Model(x,y)


xent_loss = mean_squared_error(K.flatten(x), K.flatten(y)) 
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss*lamdba)
vae.add_loss(vae_loss)

vae.compile(optimizer=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
vae_monitor.compile(loss = mean_squared_error,optimizer=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                    ,metrics=['acc'])
vae.summary()

x_train = np.load(r"H:\master\code\python\DLCV_HW4\image\x_train.npy")/255-0.5
x_test = np.load(r"H:\master\code\python\DLCV_HW4\image\x_test.npy")/255-0.5


vae.load_weights(r'H:\master\code\python\DLCV_HW4\0.14.h5py')
# value_train = np.zeros((epochs,3))
# value_test = np.zeros((epochs,3))
# for ee in range(epochs):
#     checkpoint = ModelCheckpoint(r'H:\master\code\python\DLCV_HW4\model'+"\-"+str(ee)+".h5py", monitor='loss',
#                                 verbose=1, mode='min',period=1)
#     hist = vae.fit(x_train,
#             shuffle=True,
#             epochs=1,
#             batch_size=batch_size,
#             # validation_data=(x_test, x_test),verbose = 1)
#             validation_data=(x_test, None),callbacks=[checkpoint],verbose = 2)
#     value_train[ee][:2] = vae_monitor.evaluate(x_train,x_train)
#     value_train[ee][2] = hist.history['loss'][0]
#     value_test[ee][:2] = vae_monitor.evaluate(x_test,x_test)
#     value_test[ee][2] = hist.history['val_loss'][0]
#     print('mse:%s   acc:%s',(value_train[ee][0],value_train[ee][1]))
#     print('mse:%s   acc:%s',(value_test[ee][0],value_test[ee][1]))
   
#     # tfbor = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


# np.save('./value_train.npy',value_train)
# np.save('./value_test',value_test)
vae_predict = vae.predict(x_test[:20],verbose=1)/2+0.5



z_1_4 = Input(shape=(latent_dim,))
_d_h = decoder_hid(z_1_4)
_d_h_1 = decoder_hid_1(_d_h)
_d_h_2 = decoder_hid_2(_d_h_1)

_d_u = decoder_upsample(_d_h_2)
_d_u_1 = decoder_upsample_1(_d_u)
_d_u_2 = decoder_upsample_2(_d_u_1)

_d_r = decoder_reshape(_d_u_2)

_d_1 = de_conv_1(_d_r)
_b7 = B_7(_d_1)
_a7 = A_7(_b7)

_d_2 = de_conv_2(_a7)
_b8 = B_8(_d_2)
_a8 = A_8(_b8)

_d_3 = de_conv_3(_a8)
_b9 = B_9(_d_3)
_a9 = A_9(_b9)

_d_4 = de_conv_4(_a9)
_b10 = B_10(_d_4)
_a10 = A_10(_b10)

_d_5 = de_conv_5(_a10)
_b11 = B_11(_d_5)
_a11 = A_11(_b11)

_d_6 = de_conv_6(_a11)

_y = x_decoded_mean_squash(_d_6)

generate = Model(z_1_4,_y)
a = np.random.normal(0,1,(10,latent_dim))
gen_predict = generate.predict(a,verbose=1)
gen_predict = gen_predict/2+0.5

for i in range(10):
    io.imshow(vae_predict[i])
    plt.show()
    io.imshow(gen_predict[i])
    plt.show()
