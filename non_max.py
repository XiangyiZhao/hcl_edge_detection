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
img = Image.open(path)
width, height = img.size

#================================================================================================================================================
#intensity function
#================================================================================================================================================
def sobelAlgo(A, Fx, Fy):
    B = hcl.compute((height, width), lambda x,y :A[x][y][0]+A[x][y][1]+A[x][y][2],"B", dtype=hcl.Float())
    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    Gx = hcl.compute((height, width), lambda x,y: hcl.select(hcl.and_(x>0,x<(height-1),y>0,y<(width-1)), hcl.sum(B[x+r,y+c]*Fx[r,c],axis=[r,c]), B[x,y]), "Gx")
    t = hcl.reduce_axis(0, 3)
    g = hcl.reduce_axis(0, 3)
    Gy = hcl.compute((height, width), lambda x,y: hcl.select(hcl.and_(x>0,x<(height-1),y>0,y<(width-1)), hcl.sum(B[x+t,y+g]*Fy[t,g],axis=[t,g]), B[x,y]), "Gy")
    return hcl.compute((height, width), lambda x,y:(hcl.sqrt(Gx[x][y]*Gx[x][y]+Gy[x][y]*Gy[x][y]))/4328*255, dtype = hcl.Float())

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
img = np.asarray(img) 
#A1 = np.array([[[32, 54, 76],[45, 67, 230],[67, 45, 230]],[[23, 45, 67],[89, 205, 30],[67, 45, 230]],[[32, 54, 76],[230, 67, 155],[67, 45, 230]]])
            
#numpy output 
np_img = np.zeros((height, width))

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
sobel_img = np.zeros((height, width, 3))

#assign (new_img, new_img, new_img) to each pixel
for x in range (0, height):
        for y in range (0, width):
                for z in range (0, 3):
                        sobel_img[x,y,z] = new_img[x,y]
												
#create an image with the array
imageio.imsave('new_image.png', sobel_img)

#================================================================================================================================================
#theta function
#================================================================================================================================================
def sobelAlgo_Gx(A, Fx, Fy):
    B = hcl.compute((height, width), lambda x,y:A[x][y][0]+A[x][y][1]+A[x][y][2], "B",dtype=hcl.Float())
    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    Gx = hcl.compute((height, width), lambda x,y: hcl.select(hcl.and_(x>0,x<(height-1),y>0,y<(width-1)), hcl.sum(B[x+r,y+c]*Fx[r,c],axis=[r,c]), B[x,y]), "Gx")
    return Gx
def sobelAlgo_Gy(A, Fx, Fy):
    B = hcl.compute((height+2, width+2), lambda x,y:A[x][y][0]+A[x][y][1]+A[x][y][2], "B",dtype=hcl.Float())
    t = hcl.reduce_axis(0, 3)
    g = hcl.reduce_axis(0, 3)
    Gy = hcl.compute((height, width), lambda x,y: hcl.select(hcl.and_(x>0,x<(height-1),y>0,y<(width-1)), hcl.sum(B[x+t,y+g]*Fy[t,g],axis=[t,g]), B[x,y]), "Gy")
    return Gy
#================================================================================================================================================
#computations
#================================================================================================================================================
#build the schedule
sx = hcl.create_schedule([A, Fx, Fy], sobelAlgo_Gx)
fx = hcl.build(sx)

sy = hcl.create_schedule([A, Fx, Fy], sobelAlgo_Gy)
fy = hcl.build(sy)

#output
np_x = np.zeros((height, width))
hcl_x = hcl.asarray(np_x)

np_y = np.zeros((height, width))
hcl_y = hcl.asarray(np_y)

#call the function
fx(hcl_A, hcl_F1, hcl_F2, hcl_x)
fy(hcl_A, hcl_F1, hcl_F2, hcl_y)
np_x = hcl_x.asnumpy()
np_y = hcl_y.asnumpy()
theta = np.arctan2(np_y, np_x)
