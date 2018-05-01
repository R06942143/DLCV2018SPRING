import numpy as np
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
import os 
from skimage import io

img_input = Input(shape=(512,512,3))
x = Conv2D(64, (3, 3), padding='same', name='block1_conv1',trainable = False)(img_input)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same', name='block1_conv2',trainable = False)(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
f1 =x
# Block 2
x = Conv2D(128, (3, 3), padding='same', name='block2_conv1',trainable = False)(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), padding='same', name='block2_conv2',trainable = False)(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
f2 =x
# Block 3
x = Conv2D(256, (3, 3), padding='same', name='block3_conv1',trainable = False)(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), padding='same', name='block3_conv2',trainable = False)(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), padding='same', name='block3_conv3',trainable = True)(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
f3 =x
# Block 4
x = Conv2D(512, (3, 3), padding='same', name='block4_conv1',trainable = False)(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', name='block4_conv2',trainable = False)(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', name='block4_conv3',trainable = True)(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
f4 =x
# Block 5
x = Conv2D(512, (3, 3), padding='same', name='block5_conv1',trainable = False)(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', name='block5_conv2',trainable = False)(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', name='block5_conv3',trainable = True)(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
f5 = x 

model = Model(img_input,x)
model.load_weights(r'H:\master\code\python\DLCV_HW3\vgg16_weights_tf_dim_ordering_tf_kernels.h5',by_name= True)
model.summary()

y = x
o = f5

o = ( Conv2D( 2048,(7 , 7), activation='relu' , padding='same'))(o)
o = Dropout(0.5)(o)
o = ( Conv2D(  2048,(1 , 1), activation='relu' , padding='same'))(o)
o = Dropout(0.5)(o)

o = ( Conv2D( 7 ,(1,1),kernel_initializer='he_normal' ))(o)
o = Conv2DTranspose( 7 , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False,padding = 'same')(o)

o2 = f4
o2 = ( Conv2D( 7 ,(1,1),kernel_initializer='he_normal' ))(o2)
o = Add()([ o , o2 ])

o = Conv2DTranspose( 7 , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False ,padding = 'same'  )(o)
o2 = f3 
o2 = ( Conv2D( 7 ,(1,1),kernel_initializer='he_normal' ))(o2)
o  = Add()([ o2 , o ])


o = Conv2DTranspose( 7 , kernel_size=(16,16) ,  strides=(8,8) , use_bias=False  ,padding = 'same' )(o)
	
o_shape = Model(img_input , o ).output_shape


o = (Activation('softmax'))(o)
FCN8 = Model( img_input , o )


FCN8.compile(loss = 'categorical_crossentropy',
              optimizer = 'adadelta',
              metrics = ['accuracy'])

FCN8.summary()

#######################train_image#############################
traing_path=r'H:\master\code\python\DLCV_HW3\hw3-train-validation\train/'
tt = os.listdir(traing_path)

train_data = np.zeros([len(tt)//2,512,512,3])
train_label = np.zeros([len(tt)//2,512,512,1],dtype=np.int8)

for i in range(len(tt)):
    if('mask' in tt[i]):
        img = io.imread(traing_path+tt[i])
        img = img//255
        train_label[i//2] = (img[:,:,0]*1 +img[:,:,1]*2+img[:,:,2]*4).reshape([512,512,1])
    else :
        train_data[i//2] = io.imread(traing_path+tt[i]) 
        
label = np.zeros([len(tt)//2,512*512,7],dtype=np.int8)
train_label = train_label.reshape([len(tt)//2,512*512])
train_label[train_label>0] -=1

for i in range(len(train_label)):
    label[i,np.arange(512*512),train_label[i]] = 1

label =label.reshape([len(tt)//2,512,512,7])
#######################validation_image#############################
validation_path=r'H:\master\code\python\DLCV_HW3\hw3-train-validation\validation/'
vv = os.listdir(validation_path)

va_data = np.zeros([len(vv)//2,512,512,3])
va_label = np.zeros([len(vv)//2,512,512,1],dtype=np.int8)

for i in range(len(vv)):
    if('mask' in vv[i]):
        img = io.imread(validation_path+vv[i])
        img = img//255
        va_label[i//2] = (img[:,:,0]*1 +img[:,:,1]*2+img[:,:,2]*4).reshape([512,512,1])
    else :
        va_data[i//2] = io.imread(validation_path+vv[i]) 
        
v_label = np.zeros([len(vv)//2,512*512,7],dtype=np.int8)

va_label = va_label.reshape([len(vv)//2,512*512])

va_label[va_label>0] -=1

for i in range(len(va_label)):
    v_label[i,np.arange(512*512),va_label[i]] = 1

v_label =v_label.reshape([len(vv)//2,512,512,7])


checkpoint = ModelCheckpoint(r'H:\master\code\python\DLCV_HW3\model_88/'+"\-{epoch:02d}-{val_acc:.2f}.h5py", monitor='val_acc', verbose=1, save_best_only=True, mode='max',period=1)
FCN8.fit(train_data,label,epochs=100,batch_size=2,verbose=1,validation_data=[va_data,v_label],callbacks=[checkpoint])
FCN8.save(r'H:\master\code\python\DLCV_HW3\model_88/FCN32.h5')