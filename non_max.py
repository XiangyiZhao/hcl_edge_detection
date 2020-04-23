import heterocl as hcl
from PIL import Image
import numpy as np
import math
import imageio


#initialization
path = "home.jpg"                                               # Your image path 
hcl.init(init_dtype=hcl.Float())
or_img = Image.open(path)
width, height = or_img.size

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

#placeholders
A = hcl.placeholder((height, width, 3), "A", dtype = hcl.Float())  #input placeholder1
Fx = hcl.placeholder((3,3), "Fx", dtype = hcl.Float())             #input placeholder2
Fy = hcl.placeholder((3,3), "Fy", dtype = hcl.Float())             #input placeholder3

#build the schedule
s = hcl.create_schedule([A, Fx, Fy], sobelAlgo)
f = hcl.build(s)

#numpy inputs
F1 = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
F2 = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
ar_img = np.asarray(or_img) 
#A1 = np.array([[[32, 54, 76],[45, 67, 230],[67, 45, 230]],[[23, 45, 67],[89, 205, 30],[67, 45, 230]],[[32, 54, 76],[230, 67, 155],[67, 45, 230]]])
            
#numpy output 
np_img = np.zeros((height, width))

#hcl inputs transfer
hcl_F1 = hcl.asarray(F1)
hcl_F2 = hcl.asarray(F2)
hcl_A = hcl.asarray(ar_img)
    
#hcl output transfer
hcl_img = hcl.asarray(np_img)

#call the function
f(hcl_A, hcl_F1, hcl_F2, hcl_img)

#change the type of output back to numpy array
edge_img = hcl_img.asnumpy()

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
	B = hcl.compute((height, width), lambda x,y:A[x][y][0]+A[x][y][1]+A[x][y][2], "B",dtype=hcl.Float())
	t = hcl.reduce_axis(0, 3)
	g = hcl.reduce_axis(0, 3)
	Gy = hcl.compute((height, width), lambda x,y: hcl.select(hcl.and_(x>0,x<(height-1),y>0,y<(width-1)), hcl.sum(B[x+t,y+g]*Fy[t,g],axis=[t,g]), B[x,y]), "Gy")
	return Gy

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
edge_dir = np.arctan2(np_y, np_x)

#=================================================================================================================================================
# Non-maximum Suppression
#=================================================================================================================================================
def non_max_sup(I, theta, Z):
        D = hcl.compute((height, width), lambda x,y: theta[x][y]*180/np.pi, "D")
        def loop_body(x, y):
                q = 255
                r = 255
                with hcl.if_(D[x][y] < 0):
                        D[x][y] = D[x][y]+180

                with hcl.if_(hcl.or_(hcl.and_(D[x][y]>=0,D[x][y]<22.5),hcl.and_(D[x][y]>=157.5,D[x][y]<=180))):
                        q = I[x][y+1]
                        r = I[x][y-1]
                with hcl.elif_(hcl.and_(22.5 <= D[x][y],D[x][y] < 67.5)):
                        q = I[x+1][y-1]
                        r = I[x-1][y+1]
                with hcl.elif_(hcl.and_(67.5 <= D[x][y],D[x][y] < 112.5)):
                        q = I[x+1][y]
                        r = I[x-1][y]
                with hcl.elif_(hcl.and_(112.5 <= D[x][y],D[x][y] < 157.5)):
                        q = I[x-1][y-1]
                        r = I[x+1][y+1]

                with hcl.if_(hcl.and_(I[x][y]>=q,I[x][y]>=r)):
                        Z[x][y] = I[x][y]
                with hcl.else_():
                        Z[x][y] = 0
        hcl.mutate(Z.shape, lambda x,y: loop_body(x,y))

#placeholders
I = hcl.placeholder((height, width), "I", dtype = hcl.Float())         #input placeholder1
theta = hcl.placeholder((height, width), "theta", dtype = hcl.Float()) #input placeholder2
Z = hcl.placeholder((height, width), "Z", dtype = hcl.Float())         #input placeholder3

#build the schedule
sm = hcl.create_schedule([I, theta, Z], non_max_sup)
fm = hcl.build(sm)

#hcl input transfer
hcl_edge_img = hcl.asarray(edge_img)
hcl_edge_dir = hcl.asarray(edge_dir)
hcl_Z = hcl.asarray(np.zeros((height, width)))

#call the function
fm(hcl_edge_img, hcl_edge_dir, hcl_Z)

#change the type of output back to numpy array
non_max_img = hcl_Z.asnumpy()

#create an array for the final image output
final_img = np.zeros((height, width, 3))

#assign (new_img, new_img, new_img) to each pixel
for x in range (0, height):
        for y in range (0, width):
                for z in range (0, 3):
                        final_img[x,y,z] = non_max_img[x,y]

#create an image with the array
imageio.imsave('home_nonMax.jpg', final_img)
