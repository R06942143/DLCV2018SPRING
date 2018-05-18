import numpy as np
import matplotlib
matplotlib.use('Agg')
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
import sys
from sklearn.manifold import TSNE

batch_size = 32
latent_dim = 512*2
img_row,img_col,img_channel = 64,64,3
intermediate_dim = latent_dim
epsilon_std = 1.0
epochs = 100
f1 = 3
f2 = 64
f3 = 128
f4 = 256
lamdba = float(0.000003)


data_path = os.path.join(sys.argv[1],'train')
path_list = os.listdir(data_path)
x_train = np.zeros([len(path_list),64,64,3])
for i in range(len(path_list)):
    img = io.imread(os.path.join(data_path,path_list[i]))
    x_train[i] = img


data_path = os.path.join(sys.argv[1],'test')
path_list = os.listdir(data_path)
x_test = np.zeros([len(path_list),64,64,3])
for i in range(len(path_list)):
    img = io.imread(os.path.join(data_path,path_list[i]))
    x_test[i] = img



label = np.genfromtxt(os.path.join(sys.argv[1],'test.csv'),delimiter=',',skip_header=1)





x = Input(shape=(img_row,img_col,img_channel))

conv_1 = Conv2D(f1,kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
B_1 = BatchNormalization()(conv_1)
A_1 =  LeakyReLU(alpha=0.01)(B_1)
D_1 = Dropout(0.4)(A_1)

conv_2 = Conv2D(f2,kernel_size=(4, 4), strides=(2, 2), padding='same')(D_1)
B_2 = BatchNormalization()(conv_2)
A_2 =  LeakyReLU(alpha=0.01)(B_2)
D_2 = Dropout(0.4)(A_2)

conv_3 = Conv2D(f3,kernel_size=(4, 4), strides=(2, 2), padding='same')(D_2)
B_3 = BatchNormalization()(conv_3)
A_3 =  LeakyReLU(alpha=0.01)(B_3)
D_3 = Dropout(0.4)(A_3)

conv_4 = Conv2D(f4,kernel_size=(4, 4), strides=(2, 2), padding='same')(D_3)
B_4 = BatchNormalization()(conv_4)
A_4 =  LeakyReLU(alpha=0.01)(B_4)
D_4 = Dropout(0.4)(A_4)



flat = Flatten()(D_4)
hidden = Dense(intermediate_dim)(flat)
hidden_1 = BatchNormalization()(hidden)
hidden_2 = Activation('relu')(hidden_1)

z_mean = Dense(latent_dim)(hidden_2)
z_log_var = Dense(latent_dim)(hidden_2)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean
z = Lambda(sampling)([z_mean, z_log_var])

decoder_hid = Dense(intermediate_dim)
decoder_hid_1 = BatchNormalization()
decoder_hid_2 =  LeakyReLU(alpha=0.01)


output_shape = (batch_size, 4, 4, f4)
decoder_upsample = Dense(f4*4*4)
decoder_upsample_1 = BatchNormalization()
decoder_upsample_2 =  LeakyReLU(alpha=0.01)
decoder_reshape  = Reshape(output_shape[1:])



up_1 = UpSampling2D()
de_conv_1 = Conv2D(f4,kernel_size=(4, 4), padding='same')
B_7 = BatchNormalization()
A_7 =(LeakyReLU(alpha=0.01))
D_7 = Dropout(0.4)

up_2 = UpSampling2D()
de_conv_2 = Conv2D(f3,kernel_size=(4, 4), padding='same')
B_8 = BatchNormalization()
A_8 =(LeakyReLU(alpha=0.01))
D_8 = Dropout(0.4)

up_3 = UpSampling2D()
de_conv_3 = Conv2D(f2,kernel_size=(4, 4), padding='same')
B_9 = BatchNormalization()
A_9 =(LeakyReLU(alpha=0.01))
D_9 = Dropout(0.4)

up_4 = UpSampling2D()
de_conv_4 = Conv2D(f1,kernel_size=(4, 4), padding='same')
x_decoded_mean_squash = Activation('sigmoid')

d_h = decoder_hid(z)
d_h_1 = decoder_hid_1(d_h)
d_h_2 = decoder_hid_2(d_h_1)

d_u = decoder_upsample(d_h_2)
d_u_1 = decoder_upsample_1(d_u)
d_u_2 = decoder_upsample_2(d_u_1)

d_r = decoder_reshape(d_u_2)

u_1 = up_1(d_r)
d_1 = de_conv_1(u_1)
b7 = B_7(d_1)
a7 = A_7(b7)
d7 = D_7(a7)

u_2 = up_2(d7)
d_2 = de_conv_2(u_2)
b8 = B_8(d_2)
a8 = A_8(b8)
d8 = D_8(a8)

u_3 = up_3(d8)
d_3 = de_conv_3(u_3)
b9 = B_9(d_3)
a9 = A_9(b9)
d9 = D_9(a9)

u_4 = up_4(d9)
d_4 = de_conv_4(u_4)

y = x_decoded_mean_squash(d_4)

vae = Model(x,y)
vae_monitor = Model(x,y)


xent_loss = mean_squared_error(K.flatten(x), K.flatten(y))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(kl_loss*lamdba+ xent_loss)
vae.add_loss(vae_loss)

vae.compile(optimizer='rmsprop')
vae_monitor.compile(loss = mean_squared_error,optimizer = 'rmsprop',metrics=['acc'])
vae.summary()






z_1_4 = Input(shape=(latent_dim,))
_d_h = decoder_hid(z_1_4)
_d_h_1 = decoder_hid_1(_d_h)
_d_h_2 = decoder_hid_2(_d_h_1)

_d_u = decoder_upsample(_d_h_2)
_d_u_1 = decoder_upsample_1(_d_u)
_d_u_2 = decoder_upsample_2(_d_u_1)

_d_r = decoder_reshape(_d_u_2)

_u_1 = up_1(_d_r)
_d_1 = de_conv_1(_u_1)
_b7 = B_7(_d_1)
_a7 = A_7(_b7)
_d7 = D_7(_a7)

_u_2 = up_1(_d7)
_d_2 = de_conv_2(_u_2)
_b8 = B_8(_d_2)
_a8 = A_8(_b8)
_d8 = D_8(_a8)

_u_3 = up_1(_d8)
_d_3 = de_conv_3(_u_3)
_b9 = B_9(_d_3)
_a9 = A_9(_b9)
_d9 = D_9(_a9)

_u_4 = up_1(_d9)
_d_4 = de_conv_4(_u_4)

_y = x_decoded_mean_squash(_d_4)


generate = Model(z_1_4,_y)

vae.load_weights('VAE.h5py')


np.random.seed(1)
a = np.random.normal(0,2,(32,latent_dim))
vae_predict = (vae.predict(x_test[10:20],verbose=1))
gen_predict = (generate.predict(a,verbose=1))
v = np.zeros((64,64*10,3),dtype = float)
g = np.zeros((64*4,64*8,3),dtype = float)
for i in range(10):
    v[:,64*i:64*(i+1),:] = vae_predict[i]
vvv = np.zeros((64,64*10,3))
for i in range(10):
    vvv[:,i*64:(i+1)*64,:] = x_test[i+10]
io.imsave(os.path.join(sys.argv[2],'fig1_3.jpg'),np.concatenate((vvv,v),axis = 0))
      
for i in range(32):
    g[(i//8)*64:(i//8+1)*64,64*(i%8):64*((i%8)+1),:] = gen_predict[i]
io.imsave(os.path.join(sys.argv[2],'fig1_4.jpg'),g)

enco = Model(x,z)

tt = enco.predict(x_test)

tttt = TSNE(n_components=2, random_state=0).fit_transform(tt)
male = label[:,8]
# for i in range(len(tttt)):
#     if(male[i] ==1):
#         plt.scatter(tttt[i][0],tttt[i][1],c='r',label = 'male')
#     else:
#         plt.scatter(tttt[i][0],tttt[i][1],c='y',label = 'female')
plt.scatter(tttt[male ==1][:,0],tttt[male ==1][:,1],c = 'r',label = 'male')
plt.scatter(tttt[male ==0][:,0],tttt[male ==0][:,1],c = 'y',label = 'female')
plt.legend()
plt.savefig(os.path.join(sys.argv[2],'fig1_5.jpg'))

# 
# plt.show()
# # plt.legend('male')
# 
# # plt.legend('female')
# plt.show()
