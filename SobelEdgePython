from PIL import Image
import math
#path = "peru.jpeg" # Your image path 
#img = Image.open(path)
width = 3
height = 3
A = [[[32, 54, 76],[45, 67, 230],[67, 45, 230]],
     [[23, 45, 67],[89, 205, 30],[67, 45, 230]],
     [[32, 54, 76],[230, 67, 155],[67, 45, 230]]]
B = [[0 for c in range (0, width-2)]for r in range (0, height-2)]
for x in range(1, width-1):  # ignore the edge pixels for simplicity (1 to width-1)
    for y in range(1, height-1): # ignore edge pixels for simplicity (1 to height-1)

        # initialise Gx to 0 and Gy to 0 for every pixel
        Gx = 0
        Gy = 0

        # top left pixel
        p = A[x-1][y-1]
        r = p[0]
        g = p[1]
        b = p[2]

        # intensity ranges from 0 to 765 (255 * 3)
        intensity = r + g + b

        # accumulate the value into Gx, and Gy
        Gx += -intensity
        Gy += -intensity

        # remaining left column
        p = A[x-1][y]
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += -2 * (r + g + b)

        p = A[x-1][y+1]
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += -(r + g + b)
        Gy += (r + g + b)

        # middle pixels
        p = A[x][y-1]
        r = p[0]
        g = p[1]
        b = p[2]

        Gy += -2 * (r + g + b)
        p = A[x][y+1]
        r = p[0]
        g = p[1]
        b = p[2]

        Gy += 2 * (r + g + b)

        # right column
        p = A[x+1][y-1]
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += (r + g + b)
        Gy += -(r + g + b)

        p = A[x+1][y]
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += 2 * (r + g + b)

        p = A[x+1][y+1]
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += (r + g + b)
        Gy += (r + g + b)

        # calculate the length of the gradient (Pythagorean theorem)
        length = math.sqrt((Gx * Gx) + (Gy * Gy))

        # normalise the length of gradient to the range 0 to 255
        length = length / 4328 * 255

        # draw the length in the edge image
        #newpixel = img.putpixel((length,length,length))
        B[x-1][y-1] = length

print(B)
