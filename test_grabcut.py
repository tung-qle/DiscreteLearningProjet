from grabcut import GrabCut
import numpy as np
import cv2 as cv
import time

BLUE = [255, 0, 0]        # rectangle color

def test(filename, rect, n_iter=1):
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

    img_outs = []

    print("\nPerforming the sementation...")
    start = time.time()
    g = GrabCut(img, mask, rect)
    img_outs.append(img*(mask[:,:,None]==3))
    end = time.time()
    print("Done in: ", end - start,"s !")
    if n_iter > 1:
        for i in range(n_iter-1):
            print("\nIteration",i+2,"..." )
            start = time.time()
            g.run()
            img_outs.append(img*(mask[:,:,None]==3))
            end = time.time()
            print("Done in: ", end - start,"s !")
    
    print("\nVisualizing... Press anykey to exit")

    cv.imshow("input", img)
    cv.imshow("input with rectangle", img_rect)
    cv.imshow("iter1", img_outs[0])
    if n_iter > 1:
        for i in range(n_iter-1):
            cv.imshow("iter"+str(i+2), img_outs[i+1])
    cv.waitKey(0)

test(filename = 'messi5.jpg', rect = (49, 42, 459, 296), n_iter=2)
