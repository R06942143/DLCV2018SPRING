from reader import readShortVideo,getVideoList
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.applications.resnet50 import ResNet50
import sys


#output_folder = sys.argv[2]




a = getVideoList(str(sys.argv[2]))
valid_bb = []
valid_X = []

for i in range(len(list(a.values())[4])):
    b = readShortVideo(str(sys.argv[1]),
                 list(a.values())[4][i],list(a.values())[6][i])
    valid_X.extend(list(np.array(b.tolist())))
    valid_bb.append(len(b))
valid_X = np.array(valid_X).astype(np.uint8)
valid_bb= np.array(valid_bb)
input_tensor = Input(shape=(240,320,3))
model = ResNet50(input_tensor=input_tensor,weights='imagenet', include_top=False)
outcome_V = model.predict(valid_X,verbose =1)

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

FC_layer = load_model('./p1.hdf5')

ans = np.argmax(np.around(FC_layer.predict(feature_V)))
np.save('./ans.npy',ans)
for i in ans:
    print(i)


