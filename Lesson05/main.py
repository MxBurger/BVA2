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

characters_wip = [
    # SPECIAL CHARACTERS
    (14, 6, ",", 16, "marked_comma.png"),
    (3, 3, ".", 20, "marked_dot.png"),
    (9, 11, ":", 8, "marked_colon.png"),
    (0, 30, ";", 2, "marked_semicolon.png"),

    # NUMBERS
    # (0, 0, "!", 0, "marked_exclamation.png"),
    # (0, 0, "0", 0, "marked_0.png"),
    # (0, 0, "1", 0, "marked_1.png"),
    # (0, 0, "2", 0, "marked_2.png"),
    # (0, 0, "3", 0, "marked_3.png"),
    # (0, 0, "4", 0, "marked_4.png"),
    # (0, 0, "5", 0, "marked_5.png"),
    # (0, 0, "6", 0, "marked_6.png"),
    # (0, 0, "7", 0, "marked_7.png"),
    # (0, 0, "8", 0, "marked_8.png"),
    # (0, 0, "9", 0, "marked_9.png"),

    # UPPER CASE LETTERS
    (0, 2, "A", 7, "marked_A.png"),
    (0, 0, "B", 3, "marked_B.png"),
    # (0, 0, "C", 0, "marked_C.png"),
    (0, 0, "D", 8, "marked_D.png"),
    (0, 0, "E", 8, "marked_E.png"),
    (0, 0, "F", 6, "marked_F.png"),
    (0, 0, "G", 0, "marked_G.png"),
    (0, 0, "H", 0, "marked_H.png"),
    (0, 0, "I", 1, "marked_I.png"),
    (0, 0, "J", 0, "marked_J.png"),
    (0, 0, "K", 0, "marked_K.png"),
    (0, 0, "L", 0, "marked_L.png"),
    (0, 0, "M", 0, "marked_M.png"),
    (0, 0, "N", 0, "marked_N.png"),
    (0, 0, "O", 0, "marked_O.png"),
    (0, 0, "P", 0, "marked_P.png"),
    (0, 0, "Q", 0, "marked_Q.png"),
    (0, 0, "R", 0, "marked_R.png"),
    (0, 0, "S", 0, "marked_S.png"),
    (0, 0, "T", 0, "marked_T.png"),
    (0, 0, "U", 0, "marked_U.png"),
    (0, 0, "V", 0, "marked_V.png"),
    (7, 0, "W", 8, "marked_W.png"),
    (0, 0, "X", 0, "marked_X.png"),
    (0, 0, "Y", 0, "marked_Y.png"),
    (0, 0, "Z", 0, "marked_Z.png"),
    (0, 0, "Ä", 0, "marked_AE.png"),
    (0, 0, "Ö", 0, "marked_OE.png"),
    (0, 0, "Ü", 0, "marked_UE.png"),

    # LOWER CASE LETTERS
    (0, 0, "a", 0, "marked_a.png"),
    (0, 0, "b", 0, "marked_b.png"),
    (0, 0, "c", 0, "marked_c.png"),
    (0, 0, "d", 0, "marked_d.png"),
    (0, 0, "e", 0, "marked_e.png"),
    (0, 0, "f", 0, "marked_f.png"),
    (0, 0, "g", 0, "marked_g.png"),
    (0, 0, "h", 0, "marked_h.png"),
    (1, 1, "i", 57, "marked_i.png"),
    (13, 0, "j", 2, "marked_j.png"),
    (0, 0, "k", 0, "marked_k.png"),
    (0, 0, "l", 0, "marked_l.png"),
    (0, 1, "m", 37, "marked_m.png"),
    (12, 0, "n", 115, "marked_n.png"),
    (2, 1, "o", 38, "marked_o.png"),
    (0, 0, "p", 0, "marked_p.png"),
    (0, 0, "q", 0, "marked_q.png"),
    (3, 2, "r", 81, "marked_r.png"),
    (1, 3, "s", 102, "marked_s.png"),
    (0, 0, "t", 0, "marked_t.png"),
    (4, 0, "u", 39, "marked_u.png"),
    (0, 0, "v", 0, "marked_v.png"),
    (0, 0, "w", 0, "marked_w.png"),
    (0, 0, "x", 0, "marked_x.png"),
    (0, 0, "y", 0, "marked_y.png"),
    (0, 0, "z", 0, "marked_z.png"),
    (0, 0, "ä", 0, "marked_ae.png"),
    (5, 31, "ö", 7, "marked_oe.png"),
    (0, 0, "ü", 0, "marked_ue.png"),
]

os.makedirs(out_img_dir, exist_ok=True)

myAnalysis = OCRanalysis()
threshold = 0.998
print(f"OCR Analysis Results with threshold {threshold}:")
for (row, col, letter, expected_count, out_img_path) in characters:
    full_out_img_path = os.path.join(out_img_dir, out_img_path)

    actual_count = myAnalysis.run(img_path, full_out_img_path, row, col, threshold, shrink_chars=True)
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