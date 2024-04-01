from skimage.segmentation import clear_border
import cv2
import numpy as np
import imutils
import pytesseract
import xml.etree.ElementTree as ET


def calculate_iou(bbox1, bbox2):
    # Wyznaczenie wspólnego obszaru
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # Brak wspólnego obszaru

    # Wyznaczenie obszaru przecięcia
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Wyznaczenie obszaru obu bbox
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Wyznaczenie obszaru unii
    union_area = bbox1_area + bbox2_area - intersection_area

    # Obliczenie IoU
    iou = intersection_area / union_area
    return iou

image_name = "Cars376"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread('images\\' + image_name + '.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (3, 3), 1)

edged = cv2.Canny(gray, 30, 200)  # Wykrywanie krawędzi
edged = clear_border(edged)

# 376
edged = cv2.dilate(edged, None, iterations=1)

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Znajdowanie konturów

contours = imutils.grab_contours(keypoints)  # Zwraca listę konturów

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Sortowanie

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)  # Uproszczenie konturów
    if len(approx) == 4:  # Sprawdzenie czy jest czworokątem
        location = approx  # 4 wierzchołki konturu
        break

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)  # Rysowanie białych wypełnionych konturów
new_image = cv2.bitwise_and(img, img, mask=mask)  # Nałożenie maski

(x, y) = np.where(mask == 255)  # Znalezienie białych pikseli na masce
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))

# 1, 349, 376
img = cv2.GaussianBlur(img, (1, 1), 1)

# 0, 4, 15
#img = cv2.GaussianBlur(img, (5,5), 1)

cropped_image = img[x1:x2 + 3, y1:y2 + 3]  # Wycięcie z marginesem

cv2.imshow("gray", cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.imshow("edged", cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.imshow('new_image', cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.imshow('cropped', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)

text = pytesseract.image_to_string(cropped_image)

#Usunięcie spacji, małych liter i interpunkcji
text_fixed = ''.join([c for c in text if c.isupper() or c.isdigit() or c.isalpha()])


print("License number:", text_fixed)


def calculate_iou(bbox1, bbox2):
    # Wyznaczenie wspólnego obszaru
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # Brak wspólnego obszaru

    # Wyznaczenie obszaru przecięcia
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Wyznaczenie obszaru obu bbox
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Wyznaczenie obszaru unii
    union_area = bbox1_area + bbox2_area - intersection_area

    # Obliczenie IoU
    iou = intersection_area / union_area
    return iou


tree = ET.parse('annotations\\' + image_name + '.xml')
root = tree.getroot()

# Odczytanie współrzędnych z pliku XML
xmin = int(root.find('object/bndbox/xmin').text)
ymin = int(root.find('object/bndbox/ymin').text)
xmax = int(root.find('object/bndbox/xmax').text)
ymax = int(root.find('object/bndbox/ymax').text)

bbox_xml = (xmin, ymin, xmax, ymax)
bbox_detected = (y1, x1, y2 + 3, x2 + 3)

# Obliczenie IoU
iou = calculate_iou(bbox_xml, bbox_detected)
print("Intersection over Union (IoU):", iou)

