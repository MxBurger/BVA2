import os

import cv2
import numpy as np

from OCRanalysis import OCRanalysis

img_path = 'altesTestament_ArialBlack.png'
out_img_dir = "marked/"
merged_img_path = "merged_overlay.png"

# (row, col, letter, expected_count, image_out_path)
characters = [
    (14, 6, ",", 16, "marked_comma.png"),
    (3, 3, ".", 20, "marked_dot.png"),
    (9, 11, ":", 8, "marked_colon.png"),
    (0, 30, ";", 2, "marked_semicolon.png"),
    (0, 2, "A", 7, "marked_A.png"),
    (0, 0, "I", 1, "marked_I.png"),
    (7, 0, "W", 8, "marked_W.png"),
    (1, 1, "i", 57, "marked_i.png"),
    (13, 0, "j", 2, "marked_j.png"),
    (0, 1, "m", 37, "marked_m.png"),
    (12, 0, "n", 115, "marked_n.png"),
    (2, 1, "o", 38, "marked_o.png"),
    (3, 2, "r", 81, "marked_r.png"),
    (1, 3, "s", 102, "marked_s.png"),
    (4, 0, "u", 39, "marked_u.png"),
    (5, 31, "ö", 7, "marked_oe.png"),
]

os.makedirs(out_img_dir, exist_ok=True)

myAnalysis = OCRanalysis()
threshold = 0.998
print(f"OCR Analysis Results with threshold {threshold}:")
for (row, col, letter, expected_count, out_img_path) in characters:
    full_out_img_path = os.path.join(out_img_dir, out_img_path)

    actual_count = myAnalysis.run(img_path, full_out_img_path, row, col, threshold)
    print(f"Letter: \'{letter}\', Expected Count: {expected_count}, Actual Count: {actual_count}")

print("Merging images...")
image_files = [f for f in os.listdir(out_img_dir)]

base_image_path = os.path.join(out_img_dir, image_files[0])
result_image = cv2.imread(base_image_path, cv2.IMREAD_GRAYSCALE)

for image_file in image_files[1:]:
    image_path = os.path.join(out_img_dir, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    result_float = result_image.astype(float) / 255.0
    image_float = image.astype(float) / 255.0

    result_float = result_float * image_float

    result_image = (result_float * 255.0).astype(np.uint8)

cv2.imwrite(merged_img_path, result_image)
print(f"Ergebnisbild gespeichert unter: {merged_img_path}")