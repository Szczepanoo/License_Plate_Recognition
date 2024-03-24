from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class ANPR:
    def __init__(self, min_ar=4, max_ar=5, debug=False):
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.debug = debug

    def debug_imshow(self, title, image, wait_key=False):
        if self.debug:
            cv2.imshow(title, image)
            if wait_key:
                cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray, keep=5):
        # perform a blackhat morphological operation that will allow
        # us to reveal dark regions (i.e., text) on light backgrounds
        # (i.e., the license plate itself)
        rec_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rec_kernel)

        square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, square_kernel)
        (T, light) = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        print(f'automatically set {T} threshold')

        # compute the Scharr gradient representation of the blackhat
        # image in the x-direction and then scale the result back to
        # the range [0, 255]
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
        grad_x = grad_x.astype("uint8")

        self.debug_imshow('scharr', grad_x)

        # blur the gradient representation, applying a closing
        # operation, and threshold the image using Otsu's method
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rec_kernel)
        thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # perform a series of erosions and dilations to clean up the
        # thresholded image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # take the bitwise AND between the threshold result and the
        # light regions of the image
        final = cv2.bitwise_and(thresh, thresh, mask=light)
        final = cv2.dilate(final, None, iterations=2)
        final = cv2.erode(final, None, iterations=1)

        self.debug_imshow('gray', gray)
        self.debug_imshow('blackhat', blackhat)
        self.debug_imshow('opening', opening)
        self.debug_imshow('light', light)
        self.debug_imshow("scharr Thresh", thresh)
        self.debug_imshow("Final", final)
        # self.plot_histogram(gray)

        cv2.waitKey(0)

        # find contours in the thresholded image and sort them by
        # their size in descending order, keeping only the largest
        # ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
        # return the list of contours
        return cnts

    def locate_license_plate(self, gray, candidates, use_clear_border=False):
        # initialize the license plate contour and ROI
        lpCnt = None
        roi = None

        # loop over the license plate candidate contours
        for c in candidates:
            # compute the bounding box of the contour and then use
            # the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            print('ar', ar)

            # check to see if the aspect ratio is rectangular
            if ar >= self.min_ar and ar <= self.max_ar:
                # store the license plate contour and extract the
                # license plate from the grayscale image and then
                # threshold it
                lpCnt = c
                license_plate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(license_plate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                # check to see if we should clear any foreground
                # pixels touching the border of the image
                # (which typically, not but always, indicates noise)
                if use_clear_border:
                    roi = clear_border(roi)
                # display any debugging information and then break
                # from the loop early since we have found the license
                # plate region
                self.debug_imshow("License Plate", license_plate)
                self.debug_imshow("ROI", roi, wait_key=True)
                break
                # return a 2-tuple of the license plate ROI and the contour
                # associated with it
        return roi, lpCnt

# Inicjalizacja obiektu ANPR
anpr = ANPR(min_ar=4, max_ar=5, debug=True)

# Wczytanie obrazu
image = cv2.imread('images\\Cars1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Lokalizacja kandydatów na tablicę rejestracyjną
candidates = anpr.locate_license_plate_candidates(gray)

# Wykrycie właściwej tablicy rejestracyjnej
roi, lpCnt = anpr.locate_license_plate(gray, candidates, use_clear_border=True)

# Odczyt tekstu z tablicy rejestracyjnej
if roi is not None:
    text = pytesseract.image_to_string(roi, config='--psm 8')
    print("Odczytany tekst z tablicy rejestracyjnej:", text)

# Wyświetlenie obrazu z zaznaczonymi obszarami
cv2.imshow('License Plate Candidates', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
