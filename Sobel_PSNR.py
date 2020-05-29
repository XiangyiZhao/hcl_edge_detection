from PIL import Image
import numpy as np
import math
import imageio

#================================================================================================================================================
#Read the original and the processed image
#================================================================================================================================================
path_org = "home.jpg"                                               # Your image path 
path_edge = "home_float.jpg"
img_org = Image.open(path_org)
img_edge = Image.open(path_edge)
width, height = img_org.size
int MSE = 0;

for x in range(0, height):
  for y in range(0, width):
    MSE += (img_org[x][y][0]+img_org[x][y][1]+img_org[x][y][2]-img_edge[x][y])^2
    
MSE_final = MSE/(height*width)
print(MSE_final)
