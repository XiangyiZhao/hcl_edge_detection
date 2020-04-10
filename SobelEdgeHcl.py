import heterocl as hcl
from PIL import Image
import numpy as np
import math
import imageio

#================================================================================================================================================
#initialization
#================================================================================================================================================
path = "hcl_img.jpg"                                               # Your image path 
hcl.init(init_dtype=hcl.Float())
width = 280
height = 412

#================================================================================================================================================
#main function
#================================================================================================================================================
def sobelAlgo(A, Fx, Fy):
    B = hcl.compute((height, width), lambda x,y:A[x][y][0]+A[x][y][1]+A[x][y][2], "B",dtype=hcl.Float())
    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    Gx = hcl.compute((height-2, width-2), lambda y,x:hcl.sum(B[y+r, x+c]*Fx[r,c], axis = [r,c]), "Gx", dtype = hcl.Float())
    t = hcl.reduce_axis(0, 3)
    g = hcl.reduce_axis(0, 3)
    Gy = hcl.compute((height-2, width-2), lambda y,x:hcl.sum(B[y+t, x+g]*Fy[t,g], axis = [t,g]), "Gy", dtype = hcl.Float())
    return hcl.compute((height-2, width-2), lambda y,x:(hcl.sqrt(Gx[y][x]*Gx[y][x]+Gy[y][x]*Gy[y][x]))/4328*255, dtype = hcl.Float())

#================================================================================================================================================
#computations
#================================================================================================================================================
#placeholders
A = hcl.placeholder((height, width, 3), "A", dtype = hcl.Float())  #input placeholder
Fx = hcl.placeholder((3,3), "Fx", dtype = hcl.Float())             #output placeholder1
Fy = hcl.placeholder((3,3), "Fy", dtype = hcl.Float())             #output placeholder2

#build the schedule
s = hcl.create_schedule([A, Fx, Fy], sobelAlgo)
f = hcl.build(s)

#numpy inputs
F1 = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
F2 = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
img = np.asarray(imageio.imread(path)) 
#A1 = np.array([[[32, 54, 76],[45, 67, 230],[67, 45, 230]],[[23, 45, 67],[89, 205, 30],[67, 45, 230]],[[32, 54, 76],[230, 67, 155],[67, 45, 230]]])
            
#numpy output 
np_img = np.zeros((height-2, width-2))

#hcl inputs transfer
hcl_F1 = hcl.asarray(F1)
hcl_F2 = hcl.asarray(F2)
hcl_A = hcl.asarray(img)
    
#hcl output transfer
hcl_img = hcl.asarray(np_img)

#call the function
f(hcl_A, hcl_F1, hcl_F2, hcl_img)

#change the type of output back to numpy array
new_img = hcl_img.asnumpy()

#create an array for the final image output
finalimg = np.zeros((height-2, width-2, 3))

#assign (new_img, new_img, new_img) to each pixel
for x in range (0, height-2):
        for y in range (0, width-2):
                for z in range (0, 3):
                        finalimg[x,y,z] = new_img[x,y]
												
#create an image with the array
imageio.imsave('new_image.png', finalimg)
