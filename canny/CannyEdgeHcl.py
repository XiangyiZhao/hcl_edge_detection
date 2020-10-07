import heterocl as hcl
from PIL import Image
import numpy as np
import math
import imageio
#import scipy.special
#import os

hcl.init(init_dtype=hcl.Float())
path = "canny_img3.jpg"
img = Image.open(path).convert('L')
width, height = img.size

#==================================================================================================
# 1. Generate Gaussian Filter
#==================================================================================================
kernel_size = 5
sigma = 1
k = int(kernel_size) // 2
x, y = np.mgrid[-k:k+1, -k:k+1]
normal = 1 / (2.0 * np.pi * sigma**2)
g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

#==================================================================================================
# 2. Apply Gaussian and Sobel Filters
#==================================================================================================
# main function
def Gaussian_Sobel_filters(A, G, Fx, Fy):
  h = hcl.reduce_axis(0, kernel_size)
  w = hcl.reduce_axis(0, kernel_size)
  B = hcl.compute((height, width), lambda y,x: hcl.select(hcl.and_(y>(k-1), y<(width-k), x>(k-1), x<(height-k)), hcl.sum(A[y+w, x+h]*G[w,h], axis=[w,h]), B[y,x]), "B", dtype = hcl.Float())
  # Sobel Filters
  r = hcl.reduce_axis(0, 3)
  c = hcl.reduce_axis(0, 3)
  Gx = hcl.compute((height, width), lambda y,x:hcl.select(hcl.and_(y > (k-1), y < (width-k), x > (k-1), x < (height-k)), hcl.sum(B[y+r, x+c]*Fx[r,c], axis = [r,c]), B[y,x]), "Gx", dtype = hcl.Float())
  t = hcl.reduce_axis(0, 3)
  g = hcl.reduce_axis(0, 3)
  Gy = hcl.compute((height, width), lambda y,x:hcl.select(hcl.and_(y > (k-1), y < (width-k), x > (k-1), x < (height-k)), hcl.sum(B[y+t, x+g]*Fy[t,g], axis = [t,g]), B[y,x]), "Gy", dtype = hcl.Float())
  # return the intensity matrix and the edge direction matrix?
  return hcl.compute((height, width), lambda y,x:(hcl.sqrt(Gx[y][x]*Gx[y][x]+Gy[y][x]*Gy[y][x]))/4328*255, dtype = hcl.Float())
  
# placeholders
A = hcl.placeholder((height, width, 3), "A", dtype = hcl.Float())  #input placeholder1
G = hcl.placeholder((kernel_size, kernel_size), "G", dtype = hcl.Float())        #input placeholder2
Fx = hcl.placeholder((3,3), "Fx", dtype = hcl.Float())             #output placeholder1
Fy = hcl.placeholder((3,3), "Fy", dtype = hcl.Float())             #output placeholder2

#build the schedule
s = hcl.create_schedule([A, G, Fx, Fy], Gaussian_Sobel_filters)
f = hcl.build(s)

#hcl inputs
hcl_A = hcl.asarray(np.asarray(img))
hcl_G = hcl.asarray(g)
hcl_F1 = hcl.asarray(np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]))
hcl_F2 = hcl.asarray(np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]))

#hcl output
hcl_GS = hcl.asarray(np.zeros((height, width)))

#call the function and change the output back to numpy array
f(hcl_A, hcl_G, hcl_F1, hcl_F2, hcl_GS)
npGS = hcl_GS.asnumpy()

imageio.imsave('edge_img3_hcl.png', npGS)
