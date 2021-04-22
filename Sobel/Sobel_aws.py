import heterocl as hcl
import os
import numpy as np
from PIL import Image

def test_aws_runtime(dtype=hcl.Int()):
    hcl.init(dtype)
    
    path = "/home/xz598/heterocl-1/home.jpg"                                               
    hcl.init(init_dtype=hcl.Float())
    img = Image.open(path)
    width, height = img.size
    RGB = hcl.placeholder((height,width), "RGB", dtype = hcl.UInt(24))
    Gx = hcl.placeholder((3,3),"Gx")
    Gy = hcl.placeholder((3,3),"Gy")
    
    def sobel(RGB,Gx,Gy):   
       B = hcl.compute((height,width), lambda x,y: RGB[x][y][8:0] + RGB[x][y][16:8] + RGB[x][y][24:16], "B")

       r = hcl.reduce_axis(0,3)
       c = hcl.reduce_axis(0,3)
       D = hcl.compute((height-2, width-2), 
            lambda x,y: hcl.sum(B[x+r, y+c]*Gx[r,c], axis=[r,c], name="sum1"), "xx")

       t = hcl.reduce_axis(0, 3)
       g = hcl.reduce_axis(0, 3)
       E = hcl.compute((height-2, width-2), 
            lambda x,y: hcl.sum(B[x+t, y+g]*Gy[t,g], axis=[t,g], name="sum2"), "yy")
       return  hcl.compute((height-2,width-2), 
            lambda x,y:hcl.sqrt(D[x][y]*D[x][y]+E[x][y]*E[x][y])*0.05891867, "Fimg")

    s = hcl.create_schedule([RGB,Gx,Gy],sobel)
#    #WBB = s.reuse_at(RGB, s[sobel.B], sobel.B.axis[1], "WBB")
#    LBX = s.reuse_at(sobel.B._op, s[sobel.xx], sobel.xx.axis[0], "LBX")
#    LBY = s.reuse_at(sobel.B._op, s[sobel.yy], sobel.yy.axis[0], "LBY") 
#    WBX = s.reuse_at(LBX, s[sobel.xx], sobel.xx.axis[1], "WBX")
#    WBY = s.reuse_at(LBY, s[sobel.yy], sobel.yy.axis[1], "WBY")
#    #WBX = s.reuse_at(sobel.B._op, s[sobel.xx], sobel.xx.axis[1], "WBX")
#    #WBY = s.reuse_at(sobel.B._op, s[sobel.yy], sobel.yy.axis[1], "WBY")
#    s.partition(LBX, dim=1)
#    s.partition(LBY, dim=1)
#    s.partition(WBX)
#    s.partition(WBY)
#    s.partition(Gx)
#    s.partition(Gy)
#    s[sobel.B].pipeline(sobel.B.axis[1])
#    s[sobel.xx].pipeline(sobel.xx.axis[1])
#    s[sobel.yy].pipeline(sobel.yy.axis[1])
#    s[sobel.Fimg].pipeline(sobel.Fimg.axis[1])
    
    target = hcl.Platform.aws_f1
    target.config(compile="vitis", mode="hw_exe", project="test_aws")
    f = hcl.build(s, target=target)

    # Requires AWS CLI package
    if os.system("which aws >> /dev/null") != 0:
        return
    
    npGx = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    npGy = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
#    hcl_Gx = hcl.asarray(npGx)
#    hcl_Gy = hcl.asarray(npGy)
    npF = np.zeros((height-2,width-2))
#    hcl_F = hcl.asarray(npF)
    img = np.asarray(img)
    img_RGB = np.zeros((height, width))
    for x in range (0, height):
        for y in range(0, width):
            img_RGB[x,y] = img[x,y,0] << 16 | img[x,y,1] << 8 | img[x,y,2]
    args = (img_RGB, npGx, npGy, npF)

    # Generate local projects
    f.inspect(args)

    # Compile FPGA binary
    key_path = "/home/xz598/test-key.pem"
    f.compile(args, remote=True, 
        aws_key_path=key_path, instance="t2.2xlarge")

    # Execute bitstream on board
    f.execute(args, remote=True, 
        aws_key_path=key_path, instance="f1.2xlarge")


def test_vitis_runtime(dtype=hcl.Int()):
    hcl.init(dtype)
    
    path = "/home/xz598/heterocl-1/home.jpg"                                               
    hcl.init(init_dtype=hcl.Float())
    img = Image.open(path)
    width, height = img.size
    RGB = hcl.placeholder((height,width), "RGB", dtype = hcl.UInt(24))
    Gx = hcl.placeholder((3,3),"Gx")
    Gy = hcl.placeholder((3,3),"Gy")
    
    def sobel(RGB,Gx,Gy):   
       B = hcl.compute((height,width), lambda x,y: RGB[x][y][8:0] + RGB[x][y][16:8] + RGB[x][y][24:16], "B")

       r = hcl.reduce_axis(0,3)
       c = hcl.reduce_axis(0,3)
       D = hcl.compute((height-2, width-2), 
            lambda x,y: hcl.sum(B[x+r, y+c]*Gx[r,c], axis=[r,c], name="sum1"), "xx")

       t = hcl.reduce_axis(0, 3)
       g = hcl.reduce_axis(0, 3)
       E = hcl.compute((height-2, width-2), 
            lambda x,y: hcl.sum(B[x+t, y+g]*Gy[t,g], axis=[t,g], name="sum2"), "yy")
       return  hcl.compute((height-2,width-2), 
            lambda x,y:hcl.sqrt(D[x][y]*D[x][y]+E[x][y]*E[x][y])*0.05891867, "Fimg")

    s = hcl.create_schedule([RGB,Gx,Gy],sobel)
    #WBB = s.reuse_at(RGB, s[sobel.B], sobel.B.axis[1], "WBB")
#    LBX = s.reuse_at(sobel.B._op, s[sobel.xx], sobel.xx.axis[0], "LBX")
#    LBY = s.reuse_at(sobel.B._op, s[sobel.yy], sobel.yy.axis[0], "LBY") 
#    WBX = s.reuse_at(LBX, s[sobel.xx], sobel.xx.axis[1], "WBX")
#    WBY = s.reuse_at(LBY, s[sobel.yy], sobel.yy.axis[1], "WBY")
#    #WBX = s.reuse_at(sobel.B._op, s[sobel.xx], sobel.xx.axis[1], "WBX")
#    #WBY = s.reuse_at(sobel.B._op, s[sobel.yy], sobel.yy.axis[1], "WBY")
#    s.partition(LBX, dim=1)
#    s.partition(LBY, dim=1)
#    s.partition(WBX)
#    s.partition(WBY)
#    s.partition(Gx)
#    s.partition(Gy)
#    s[sobel.B].pipeline(sobel.B.axis[1])
#    s[sobel.xx].pipeline(sobel.xx.axis[1])
#    s[sobel.yy].pipeline(sobel.yy.axis[1])
#    s[sobel.Fimg].pipeline(sobel.Fimg.axis[1])
    
    target = hcl.Platform.u280
    target.config(compile="vitis", mode="hw_exe")
    f = hcl.build(s, target=target)

    # Requires AWS CLI package
    if os.system("which aws >> /dev/null") != 0:
        return
    
    npGx = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    npGy = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
#    hcl_Gx = hcl.asarray(npGx)
#    hcl_Gy = hcl.asarray(npGy)
    npF = np.zeros((height-2,width-2))
#    hcl_F = hcl.asarray(npF)
    img = np.asarray(img)
    img_RGB = np.zeros((height, width))
    for x in range (0, height):
        for y in range(0, width):
            img_RGB[x,y] = img[x,y,0] << 16 | img[x,y,1] << 8 | img[x,y,2]
    args = (img_RGB, npGx, npGy, npF)

    # Generate local projects
    f.inspect(args)

    # Compile FPGA binary
    xpfm = "/opt/xilinx/platforms/xilinx_u280_xdma_201920_3/xilinx_u280_xdma_201920_3.xpfm"
    f.compile(args, xpfm=xpfm)

    # Execute bitstream on board
    outputs, rpt = f.execute(args)
    print(rpt)

if __name__ == "__main__":
    test_vitis_runtime()
    test_aws_runtime()
