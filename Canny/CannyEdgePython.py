import cannyEdgeClass as ced
from PIL import Image
import imageio
import numpy as np

path = "canny_img2.png"
img = np.asarray(Image.open(path).convert('L'))

detector = ced.cannyEdgeDetector(img, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)

img_final = detector.detect()
imageio.imsave('edge_img1', img_final)
