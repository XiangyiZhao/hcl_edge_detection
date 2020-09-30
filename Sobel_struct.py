import heterocl as hcl
from PIL import Image
import numpy as np
import math

def test_sobel_vivado_hls():
    path = "home.jpg"
    hcl.init(init_dtype=hcl.Float())
    img = Image.open(path)
    width, height = img.size
    Gx = hcl.placeholder((3,3),"Gx")
    Gy = hcl.placeholder((3,3),"Gy")
    ts = hcl.Struct({"fa":hcl.Float(), "fb":hcl.Float(), "fc":hcl.Float()})
    A = hcl.placeholder((height,width,3),dtype =ts)
    B = hcl.placeholder((height,width),dtype =ts)

    def sobel(A,Gx,Gy):
       D = hcl.compute((height,width), lambda x,y: (A[x][y][0], A[x][y][1], A[x][y][2]), dtype = ts)
       B = hcl.compute((height,width), lambda x,y: D[x][y][0] + D[x][y][1] + D[x][y][2], "B", dtype = hcl.Float())
       r = hcl.reduce_axis(0,3)
       c = hcl.reduce_axis(0,3)
       Fx = hcl.compute((height-2, width-2),
            lambda x,y: hcl.sum(B[x+r, y+c]*Gx[r,c], axis=[r,c]), "xx")
       t = hcl.reduce_axis(0, 3)
       g = hcl.reduce_axis(0, 3)
       Fy = hcl.compute((height-2, width-2),
            lambda x,y: hcl.sum(B[x+t, y+g]*Gy[t,g], axis=[t,g]), "yy")
       return  hcl.compute((height-2,width-2),
            lambda x,y:hcl.sqrt(Fx[x][y]*Fx[x][y]+Fy[x][y]*Fy[x][y])*0.05891867,"Fimg")

    s = hcl.create_schedule([A,Gx,Gy],sobel)
    LBX = s.reuse_at(sobel.B._op, s[sobel.xx], sobel.xx.axis[0], "LBX")
    LBY = s.reuse_at(sobel.B._op, s[sobel.yy], sobel.yy.axis[0], "LBY")
    WBX = s.reuse_at(LBX, s[sobel.xx], sobel.xx.axis[1], "WBX")
    WBY = s.reuse_at(LBY, s[sobel.yy], sobel.yy.axis[1], "WBY")
    s.partition(LBX,dim=1)
    s.partition(LBY,dim=1)
    s.partition(WBX)
    s.partition(WBY)
    s.partition(Gx)
    s.partition(Gy)
    s[sobel.B].pipeline(sobel.B.axis[1])
    s[sobel.xx].pipeline(sobel.xx.axis[1])
    s[sobel.yy].pipeline(sobel.yy.axis[1])
    s[sobel.Fimg].pipeline(sobel.Fimg.axis[1])

    target = hcl.platform.zc706
    s.to([A,Gx,Gy], target.xcel)
    s.to(sobel.Fimg, target.host)
    target.config(compile="vivado_hls", mode="csim|csyn")

    npGx = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    npGy = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    hcl_Gx = hcl.asarray(npGx)
    hcl_Gy = hcl.asarray(npGy)

    npF = np.zeros((height-2,width-2))
    hcl_F = hcl.asarray(npF)
    npA = np.array(img)
    hcl_A = hcl.asarray(npA)
    f = hcl.build(s, target)
    f(hcl_A, hcl_Gx, hcl_Gy, hcl_F)
    report = f.report()


test_sobel_vivado_hls()
