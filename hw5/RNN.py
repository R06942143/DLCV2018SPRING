from keras.applications import InceptionV3
from reader import readShortVideo,getVideoList
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.applications.resnet50 import ResNet50
from keras.regularizers import *

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
feature_X = np.zeros((len(bb),max(bb),2048))
count = 0
for i in range(len(bb)):
    for j in range(bb[i]):
        feature_X[i][j] = outcome[count]
        count+=1


feature_V = np.zeros((len(valid_bb),max(valid_bb),2048))
count = 0
for i in range(len(valid_bb)):
    for j in range(valid_bb[i]):
        feature_V[i][j] = outcome_V[count]
        count+=1


input_features = Input(batch_shape=(None,None,2048))
x = Bidirectional(GRU(512,kernel_regularizer = l2(0.002),return_sequences = True))(input_features)
x = Bidirectional(GRU(256,kernel_regularizer = l2(0.002),return_sequences = True))(x)
x = Dense(11,kernel_regularizer = l2(0.002),activation = 'softmax')(x)
x = Reshape((11,))(x)
RNN_layer = Model(input_features,x)
RNN_layer.summary()

#RNN_layer = Sequential()
#RNN_layer.add(Input(shape=(None,2048)))
#RNN_layer.add(Bidirectional(LSTM(256)))
#RNN_layer.add(Bidirectional(LSTM(128)))
#RNN_layer.add(Dense(11,activation = 'softmax'))






RNN_layer.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

checkpoint = ModelCheckpoint("./model/RNN_model_{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max',period=1)

RNN_layer.fit(feature_X, train_Y, epochs=50, batch_size=64,validation_data=(feature_V,valid_Y),callbacks=[checkpoint],verbose=1)
see = RNN_layer.predict(feature_V)
np.savetxt('./see.npy',np.round(see),delimiter=',')


