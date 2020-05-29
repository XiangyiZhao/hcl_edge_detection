from PIL import Image
import numpy as np
import math
import imageio

path_org = "home.jpg"                                               # Your image path 
path_edge = "home_float.jpg"
img_org = Image.open(path_org)
width, height = img_org.size
print(width, height)
img_edge = Image.open(path_edge)
np_img_org = np.asarray(img_org)
np_img_edge = np.asarray(img_edge)
MSE = 0

for x in range(0, height):
  for y in range(0, width):
    org_intensity = np_img_org[x][y][0]+np_img_org[x][y][1]+np_img_org[x][y][2]
    MSE += (org_intensity-np_img_edge[x][y])**2

MSE_final = MSE/(height*width)
print(MSE_final)
