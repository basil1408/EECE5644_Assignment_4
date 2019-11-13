from PIL import Image
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.misc import toimage
import numpy
img =  Image.open(r"C:\Users\Basil Mir\Downloads\Plane.jpg")
img.show()
np_im = numpy.array(img)
print (np_im.shape)
new_np_im= np_im.transpose(2,0,1)
print (new_np_im.shape)
li=[]
for i in range(321):
    for j in range(481):
        new=[i ,j, new_np_im[0,i,j] ,new_np_im[1,i,j] ,new_np_im[2,i,j]]
        li.append(new)
normalized = preprocessing.normalize(li, norm='l1')
 
for j in range(2,6):
    EM=GaussianMixture(n_components=j).fit(normalized)
    em=EM.predict(normalized)
    la=[]
    for i in range(154401):
        z=li[i][0]
        c=li[i][1]
        if em[i] == 0:
            new=[0,0,255]
            la.append(new)
        elif em[i] == 1:
            new=[255,0,0]
            la.append(new)
        elif em[i] == 3:
            new=[255,255,0]
            la.append(new)
        elif em[i] == 2:
            new=[0,255,0]
            la.append(new)
        elif em[i] == 4:
            new=[256,0,255]
            la.append(new)
    pic=numpy.reshape(la,(321,481,3))
    toimage(pic).show() 