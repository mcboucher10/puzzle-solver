import cv2 as cv
import numpy as np
import sys

TOTAL_PIECES = 100

# Load images
pieces_img = cv.imread('example_6.jpg')
box_img = cv.imread('box.jpeg')

if pieces_img is None or box_img is None:
    sys.exit('Could not open one or both input images.')

annotated_box = box_img.copy()

def get_scale_params(img):
    height = img.shape[0]
    font_scale = max(0.5, height / 1000.0) 
    thickness = max(2, int(height / 300))
    return font_scale, thickness

_, thickness = get_scale_params(annotated_box)
for i in range(10):
    cv.line(annotated_box, (0, annotated_box.shape[0] // 10 * i), (annotated_box.shape[1], annotated_box.shape[0] // 10 * i), (0, 255, 0), thickness)
    cv.line(annotated_box, (annotated_box.shape[1] // 10 * i, 0), (annotated_box.shape[1] // 10 * i, annotated_box.shape[0]), (0, 255, 0), thickness)

sift = cv.SIFT_create() # type: ignore

# Compute keypoints and descriptors for the reference box image
box_gray = cv.cvtColor(box_img, cv.COLOR_BGR2GRAY)
kp_box, des_box = sift.detectAndCompute(box_gray, None)

# Model params
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params) # type: ignore

# Find and store contours
imgray = cv.cvtColor(pieces_img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(imgray, 170, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

pieces = []
areas = []

for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    if (w > pieces_img.shape[1] * 0.025 and h > pieces_img.shape[0] * 0.025 and 
        w < pieces_img.shape[1] * 0.99 and h < pieces_img.shape[0] * 0.99) and np.isclose(w, h, rtol=10):
        pieces.append({
            'img': pieces_img[y:y+h, x:x+w],
            'rect': (x, y, w, h)
        })
        areas.append(w * h)

filtered_pieces = []
i = 0
for piece in pieces:
    x, y, w, h = piece['rect']
    if np.isclose(np.median(areas), w * h, rtol=0.55):
        _, thickness = get_scale_params(pieces_img)
        cv.rectangle(pieces_img, (x, y), (x + w, y + h), (0, 255, 0), thickness)
        
        font_scale, text_thickness = get_scale_params(pieces_img)
        cv.putText(pieces_img, str(i), (int(x + w / 2), int(y + h / 2)), 
                   cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
        
        filtered_pieces.append(piece)
        i += 1

cv.imwrite('annotated_pieces.jpg', pieces_img)
print('piece annotations saved')

box_area = box_img.shape[0] * box_img.shape[1]
predicted_piece_area = box_area / TOTAL_PIECES

# Use feature detection to find piece locations
for i, piece_data in enumerate(filtered_pieces):
    piece = piece_data['img']
    piece_gray = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
    
    # Extract features from the individual piece
    kp_piece, des_piece = sift.detectAndCompute(piece_gray, None)
    
    # Match descriptors between the piece and the entire box painting
    matches = flann.knnMatch(des_piece, des_box, k=2)
    
    # Store good matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)
            
    # Calculate transform matrix
    if len(good_matches) >= 4:
        src_pts = np.float32([kp_piece[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) # type: ignore
        dst_pts = np.float32([kp_box[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) # type: ignore
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        
        if M is not None:
            # Map the corners of the piece template to their location on the box
            h_p, w_p = piece_gray.shape
            pts = np.float32([[0, 0], [0, h_p - 1], [w_p - 1, h_p - 1], [w_p - 1, 0]]).reshape(-1, 1, 2) # type: ignore
            dst = cv.perspectiveTransform(pts, M)

            pts_2d = dst.reshape(-1, 2)
            
            # Calculate the 4 side lengths using Euclidean distance 
            dists = []
            for j in range(4):
                pt1 = pts_2d[j]
                pt2 = pts_2d[(j + 1) % 4]
                dists.append(np.linalg.norm(pt2 - pt1))
                
            max_side = max(dists)
            min_side = min(dists)
            aspect_ratio = max_side / min_side
            
            diag1 = np.linalg.norm(pts_2d[0] - pts_2d[2])
            diag2 = np.linalg.norm(pts_2d[1] - pts_2d[3])
            diag_ratio = max(diag1, diag2) / min(diag1, diag2)
            
            # Calculate polygon area using the shoelace formula
            x_coords = pts_2d[:, 0]
            y_coords = pts_2d[:, 1]
            mapped_area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))
            
            # Filter out aspect ratio / shape anomalies OR polygons that are excessively larger than the predicted area
            if aspect_ratio > 2.0 or diag_ratio > 1.3 or mapped_area > (predicted_piece_area * 3):
                continue

            _, thickness = get_scale_params(annotated_box)
            #cv.polylines(annotated_box, [np.int32(dst)], True, (0, 255, 0), thickness) # type: ignore
            
            font_scale, text_thickness = get_scale_params(annotated_box)
            # Find the geometric center of the mapped piece to overlay text
            match_center_x = int(np.mean(dst[:, 0, 0]) // (annotated_box.shape[1] / 10) * (annotated_box.shape[1] / 10) + annotated_box.shape[1] / 20 - font_scale * 20)
            match_center_y = int(np.mean(dst[:, 0, 1]) // (annotated_box.shape[0] / 10) * (annotated_box.shape[0] / 10) + annotated_box.shape[0] / 20 - font_scale * 5)
            
            cv.putText(annotated_box, str(i), (match_center_x, match_center_y), 
                       cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
            
cv.imwrite('annotated_box.jpg', annotated_box)

h_piece, w_piece = pieces_img.shape[:2]
h_box, w_box = annotated_box.shape[:2]

display_pieces = cv.resize(pieces_img, (int(w_piece * (600 / h_piece)), 600))
display_box = cv.resize(annotated_box, (int(w_box * (600 / h_box)), 600))

cv.imshow('Piece Mapping', cv.hconcat((display_pieces, display_box)))
cv.waitKey(0)
cv.destroyAllWindows()