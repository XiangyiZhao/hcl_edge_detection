import numpy as np
from scipy import ndimage
from PIL import Image
import math
path = "peru.jpeg" 
img = Image.open(path)

# Algorithm is based on grayscale pictures

#==================================================================================================
# 1. Noise Reduction
#==================================================================================================
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

#==================================================================================================
# 2. Gradient Calculation
#==================================================================================================
# Sobel kernels filter
def sobel_filters(img):
  Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
  Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
  Ix = ndimage.filters.convolve(img, Kx)
  Iy = ndimage.filters.convolve(img, Ky)
  G = np.hypot(Ix, Iy)
  G=G/G.max()*255
  theta = np.arctan2(Iy, Ix)
  return (G, theta)
  
#==================================================================================================
# 3. Non-maximum Suppression
#==================================================================================================
def non_max_suppression(img, D):
  M, N = img.shape
  Z = np.zeros((M,N), dtype=np.int32) 
  angle=D*180./np.pi 
  angle[angle < 0] += 180
    for i in range(1,M-1):
      for j in range(1,N-1):
        try:
          q = 255
          r = 255
          
          #angle 0
          if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
            q = img[i, j+1]
            r = img[i, j-1]
          #angle 45
          elif (22.5 <= angle[i,j] < 67.5):
            q = img[i+1, j-1]
            r = img[i-1, j+1]
          #angle 90
          elif (67.5 <= angle[i,j] < 112.5):
            q = img[i+1, j]
            r = img[i-1, j]
          #angle 135
          elif (112.5 <= angle[i,j] < 157.5):
            q = img[i-1, j-1]
            r = img[i+1, j+1]
          if (img[i,j] >= q) and (img[i,j] >= r):
            Z[i,j] = img[i,j]
          else:
            Z[i,j] = 0
            
        except IndexError as e:
          pass
  return Z

