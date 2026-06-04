import cv2 as cv
import numpy as np
import sys
from shapely.geometry import Polygon

# Load image
pieces_img = cv.imread("many_pieces.jpg")

if pieces_img is None:
    sys.exit("Could not open input image.")

# Find and store contours
imgray = cv.cvtColor(pieces_img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(imgray, 170, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

pieces = []
areas = []
count = 0
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    if (w > pieces_img.shape[1] * 0.025 and h > pieces_img.shape[0] * 0.025 and 
        w < pieces_img.shape[1] * 0.99 and h < pieces_img.shape[0] * 0.99) and np.isclose(w,h,rtol=10):
        pieces.append({
            'img': pieces_img[y:y+h, x:x+w],
            'rect': (x,y,w,h)
        })
        areas.append(w*h)

filtered_pieces = []
i = 0
for piece in pieces:
    x,y,w,h = piece['rect']
    if np.isclose(np.median(areas), w*h, rtol=0.55):
        cv.rectangle(pieces_img, (x,y), (x+w,y+h), (0,255,0), 3)
        font_scale = max(1, int(pieces_img.shape[0] / 1000))
        cv.putText(pieces_img, str(i), (int(x + w/2), int(y + h/2)), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), 10)
        filtered_pieces.append(piece)
        i += 1

cv.imwrite("annotated_pieces.jpg", pieces_img)