import cv2
import numpy as np

# Let's load a simple image with 3 black squares
image = cv2.imread("/home/allenthreee/yuanchongjian_ws/data/ftm4cjy/0thermal.png")
# cv2.waitKey(0)
print("load image")

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
canny_threshold = 5
edged = cv2.Canny(gray, canny_threshold, canny_threshold*3, 3)
cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(edged,
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()