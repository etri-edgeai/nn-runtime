import cv2
import numpy as np
import glob

out = cv2.VideoWriter('Lab1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (2048, 2048)) 
images = sorted(glob.glob('data/HABBOF/Lab1/*.jpg'))
for i, filename in enumerate(images):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    out.write(img)
    print(f"{i}/{len(images)}")
out.release()
