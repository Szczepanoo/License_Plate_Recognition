from skimage.segmentation import clear_border
import cv2
import numpy as np
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread('images\\Cars349.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (3,3), 1)

cv2.imshow("gray",cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)

edged = cv2.Canny(gray, 30, 200) #Wykrywanie krawędzi
edged = clear_border(edged)

# 376
#edged = cv2.dilate(edged,None, iterations=1)

cv2.imshow("edged",cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
location = None
for contour in contours:
  approx = cv2.approxPolyDP(contour, 10, True)
  if len(approx) == 4:
    location = approx
    break
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow('new_image',cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))

# 349
img = cv2.GaussianBlur(img, (1,1), 1)

# 0, 1, 4, 15
#img = cv2.GaussianBlur(img, (5,5), 1)

cropped_image = img[x1:x2+3, y1:y2+3]

cv2.imshow('cropped',cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
print("License number:",pytesseract.image_to_string(cropped_image))