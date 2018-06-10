from reader import readShortVideo,getVideoList
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.applications.resnet50 import ResNet50
from keras.regularizers import *
import sys



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
#model = InceptionV3(input_tensor = input_tensor,include_top=False, weights='imagenet')
model = ResNet50(input_tensor=input_tensor,weights='imagenet', include_top=False)
#'''
outcome_V = model.predict(valid_X,verbose =1)

feature_V = np.zeros((len(valid_bb),max(valid_bb),2048))
count = 0
for i in range(len(valid_bb)):
    for j in range(valid_bb[i]):
        feature_V[i][j] = outcome_V[count]
        count+=1
#'''







RNN_layer = load_model('./p2.hdf5')

ans = np.argmax(np.around(RNN_layer.predict(feature_V)),axis =1)

with open(os.path.join(sys.argv[3],'p2_result.txt'), 'w') as f:
    for i in ans:
        f.write('%d\n' %(i))

