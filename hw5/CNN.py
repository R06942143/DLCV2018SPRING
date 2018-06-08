from keras.applications import InceptionV3
from reader import readShortVideo,getVideoList
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.applications.resnet50 import ResNet50


train_X = np.load('/root/data/DL_5/train_X.npy')
train_Y = np.load('/root/data/DL_5/train_Y.npy')
valid_X = np.load('/root/data/DL_5/valid_X.npy')
valid_Y = np.load('/root/data/DL_5/valid_Y.npy')

bb      = np.load('/root/data/DL_5/train_bb.npy')
valid_bb= np.load('/root/data/DL_5/valid_bb.npy')
input_tensor = Input(shape=(240,320,3))
#model = InceptionV3(input_tensor = input_tensor,include_top=False, weights='imagenet')
model = ResNet50(input_tensor=input_tensor,weights='imagenet', include_top=False)
outcome = model.predict(train_X,verbose = 1)
outcome_V = model.predict(valid_X,verbose =1)
feature_average = []
count = 0
for i in bb:
    for j in range(i):
        if(j==0):
            f_a = outcome[j+count]
        else:
            f_a +=outcome[j+count]
    f_a /=i
    count+=i
    feature_average.append(f_a)

feature_X = np.array(feature_average).reshape(len(feature_average),-1)


feature_average = []
count = 0
for i in valid_bb:
    for j in range(i):
        if(j==0):
            f_a = outcome_V[j+count]
        else:
            f_a +=outcome_V[j+count]
    f_a /=i
    count+=i
    feature_average.append(f_a)

feature_V = np.array(feature_average).reshape(len(feature_average),-1)

print(feature_V.shape)


FC_layer = Sequential()
FC_layer.add(Dense(512,input_shape = (2048,),activation = 'relu'))
FC_layer.add(Dense(11,activation = 'softmax'))

FC_layer.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

checkpoint = ModelCheckpoint("./model/model_{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max',period=1)

FC_layer.fit(feature_X, train_Y, epochs=50, batch_size=64,validation_data=(feature_V,valid_Y),callbacks=[checkpoint],verbose=1)
see = FC_layer.predict(feature_V)
np.savetxt('./see.npy',np.round(see),delimiter=',')


