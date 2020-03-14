from grabcutsmall import GrabCutSmall
import numpy as np
import cv2 as cv
import time

BLUE = [255, 0, 0]        # rectangle color

def imshow_zoom(name, img, scale=10):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    cv.resizeWindow(name, scale*img.shape[1], scale*img.shape[0])

def test(filename, rect):
    print("==================================")
    print("Input image:", filename)
    img = cv.imread(filename)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)
    print("Rectangle:", rect)

    img_rect = np.pad(img.copy(), ((1,1),(1,1),(0,0)), 'constant')
    for i in range(rect[0]-1,rect[0] + rect[2]+1):
        img_rect[rect[1], i+1] = BLUE
        img_rect[rect[1] + rect[3]+1, i+1] = BLUE
    for i in range(rect[1]-1,rect[1] + rect[3]+1):
        img_rect[i+1, rect[0]] = BLUE
        img_rect[i+1, rect[0] + rect[2]+1] = BLUE
    
    print("\nPerforming the sementation with Edmonds–Karp...")
    start = time.time()
    g = GrabCutSmall(img, mask, rect, pushrelabel=False)
    end = time.time()
    print("Done in: ", end - start,"s !")

    print("\nPerforming the sementation with Push-Relabel...")
    start = time.time()
    g = GrabCutSmall(img, mask, rect, pushrelabel=True)
    end = time.time()
    print("Done in: ", end - start,"s !")
    
    print("\nVisualizing... Press anykey to exit")
    img_out = img*(mask[:,:,None]==3)

    imshow_zoom('input', img)
    imshow_zoom("input with rectangle", img_rect)
    imshow_zoom("output", img_out)
    cv.waitKey(0)

test(filename = 'ballon_small.jpg', rect = (2, 1, 14, 16))
test(filename = 'dog_small.jpg', rect = (4, 8, 38, 37))
