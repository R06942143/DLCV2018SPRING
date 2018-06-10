import numpy as np
import sys 


a = np.argmax(np.load(sys.argv[1]),axis=1)
b = np.genfromtxt(sys.argv[2])
count=0
for i in range(len(a)):
    if(a[i]==b[i]):
        count+=1
print(count/len(a)) 

