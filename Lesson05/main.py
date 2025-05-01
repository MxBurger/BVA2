import os

import cv2
import numpy as np

from OCRanalysis import OCRanalysis

# img_path = 'altesTestament_ArialBlack.png'
# img_path = 'test123.png'
# img_path = 'consolas.png'
img_path = 'more_bold.png'
# img_path = 'less_pixels.png'
# img_path = 'big_picture.png'
out_img_dir = "marked/"
merged_img_path = "merged_overlay.png"

# (row, col, letter, expected_count, image_out_path)
characters = [
    # SPECIAL CHARACTERS
    (0, 56, ",", 16, "marked_comma.png"),
    (1, 60, ".", 20, "marked_dot.png"),
    (2, 10, ":", 8, "marked_colon.png"),
    (0, 30, ";", 2, "marked_semicolon.png"),
    # (0, 0, "!", 0, "marked_exclamation.png"), # no occurrences (so also no template)
    # (0, 0, "?", 0, "marked_question.png"), # no occurrences (so also no template)

    # NUMBERS
    # (0, 0, "0", 0, "marked_0.png"), # no occurrences (so also no template)
    # (0, 0, "1", 0, "marked_1.png"), # no occurrences (so also no template)
    # (0, 0, "2", 0, "marked_2.png"), # no occurrences (so also no template)
    # (0, 0, "3", 0, "marked_3.png"), # no occurrences (so also no template)
    # (0, 0, "4", 0, "marked_4.png"), # no occurrences (so also no template)
    # (0, 0, "5", 0, "marked_5.png"), # no occurrences (so also no template)
    # (0, 0, "6", 0, "marked_6.png"), # no occurrences (so also no template)
    # (0, 0, "7", 0, "marked_7.png"), # no occurrences (so also no template)
    # (0, 0, "8", 0, "marked_8.png"), # no occurrences (so also no template)
    # (0, 0, "9", 0, "marked_9.png"), # no occurrences (so also no template)

    # UPPER CASE LETTERS
    (0, 2, "A", 7, "marked_upper_A.png"),
    (14, 0, "B", 3, "marked_upper_B.png"),
    # (0, 0, "C", 0, "marked_C.png"), # no occurrences (so also no template)
    (5, 10, "D", 8, "marked_upper_D.png"),
    (0, 26, "E", 8, "marked_upper_E.png"),
    (3, 28, "F", 6, "marked_upper_F.png"),
    (0, 13, "G", 23, "marked_upper_G.png"),
    (0, 17, "H", 5, "marked_upper_H.png"),
    (0, 0, "I", 1, "marked_upper_I.png"),
    (20, 8, "J", 1, "marked_upper_J.png"),
    # (0, 0, "K", 0, "marked_K.png"), # no occurrences (so also no template)
    (2, 18, "L", 10, "marked_upper_L.png"),
    (4, 52, "M", 4, "marked_upper_M.png"),
    (4, 24, "N", 2, "marked_upper_N.png"),
    (10, 24, "O", 1, "marked_upper_O.png"),
    (13, 29, "P", 2, "marked_upper_P.png"),
    # (0, 0, "Q", 0, "marked_Q.png"), # no occurrences (so also no template)
    # (0, 0, "R", 0, "marked_R.png"), # no occurrences (so also no template)
    (7, 55, "S", 8, "marked_upper_S.png"),
    (5, 6, "T", 8, "marked_upper_T.png"),
    (1, 20, "U", 2, "marked_upper_U.png"),
    # (0, 0, "V", 0, "marked_V.png"), # no occurrences (so also no template)
    (6, 7, "W", 8, "marked_upper_W.png"),
    # (0, 0, "X", 0, "marked_X.png"), # no occurrences (so also no template)
    # (0, 0, "Y", 0, "marked_Y.png"), # no occurrences (so also no template)
    (19, 20, "Z", 1, "marked_upper_Z.png"),
    # (0, 0, "Ä", 0, "marked_AE.png"), # no occurrences (so also no template)
    # (0, 0, "Ö", 0, "marked_OE.png"), # no occurrences (so also no template)
    # (0, 0, "Ü", 0, "marked_UE.png"), # no occurrences (so also no template)

    # LOWER CASE LETTERS
    (0, 5, "a", 92, "marked_lower_a.png"),
    (0, 39, "b", 22, "marked_lower_b.png"),
    (0, 9, "c", 33, "marked_lower_c.png"),
    (0, 25, "d", 69, "marked_lower_d.png"),
    (0, 21, "e", 169, "marked_lower_e.png"),
    (0, 4, "f", 6, "marked_lower_f.png"),
    (0, 7, "g", 27, "marked_lower_g.png"),
    (0, 10, "h", 45, "marked_lower_h.png"),
    (0, 18, "i", 57, "marked_lower_i.png"),
    (13, 0, "j", 2, "marked_lower_j.png"),
    (10, 39, "k", 2, "marked_lower_k.png"),
    (0, 22, "l", 35, "marked_lower_l.png"),
    (0, 1, "m", 37, "marked_lower_m.png"),
    (0, 3, "n", 115, "marked_lower_n.png"),
    (0, 14, "o", 38, "marked_lower_o.png"),
    (2, 5, "p", 5, "marked_lower_p.png"),
    # (0, 0, "q", 0, "marked_q.png"), # no occurrences (so also no template)
    (0, 27, "r", 81, "marked_lower_r.png"),
    (0, 8, "s", 102, "marked_lower_s.png"),
    (0, 15, "t", 82, "marked_lower_t.png"),
    (0, 11, "u", 39, "marked_lower_u.png"),
    (3, 22, "v", 10, "marked_lower_v.png"),
    (0, 42, "w", 25, "marked_lower_w.png"),
    # (0, 0, "x", 0, "marked_x.png"), # no occurrences (so also no template)
    # (0, 0, "y", 0, "marked_y.png"), # no occurrences (so also no template)
    (19, 0, "z", 6, "marked_lower_z.png"),
    (14, 1, "ä", 2, "marked_lower_ae.png"),
    (5, 31, "ö", 7, "marked_lower_oe.png"),
    (0, 46, "ü", 7, "marked_lower_ue.png"),
]

brainrot_characters = [
    (0, 0, "I", 6, "marked_upper_I.png"),
    (0, 1, "t", 30, "marked_lower_t.png"),
    (0, 2, "a", 35, "marked_lower_a.png"),
    (0, 3, "l", 16, "marked_lower_l.png"),
    (0, 4, "i", 23, "marked_lower_i.png"),
    (0, 5, "n", 21, "marked_lower_n.png"),
    (0, 6, "B", 1, "marked_upper_B.png"),
    (0, 7, "r", 22, "marked_lower_r.png"),
    (0, 12, "o", 23, "marked_lower_o.png"),
    (0, 14, ",", 5, "marked_comma.png"),
    (0, 17, "s", 24, "marked_lower_s.png"),
    (0, 19, "k", 2, "marked_lower_k.png"),
    (3, 0, "w", 3, "marked_lower_w.png"),
    (0, 26, "A", 4, "marked_upper_A.png"),
    (1, 7, "f", 4, "marked_lower_f.png"),
    (6, 1, "y", 6, "marked_lower_y.png"),
    (2, 5, "d", 7, "marked_lower_d.png"),
    (1, 1, "e", 36, "marked_lower_e.png"),
    (2, 6, "m", 12, "marked_lower_m.png"),
    (3, 8, "c", 13, "marked_lower_c.png"),
    (5, 3, "p", 7, "marked_lower_p.png"),
    (3, 27, "u", 4, "marked_lower_u.png"),
    (3, 3, "h", 11, "marked_lower_h.png"),
    (5, 22, ".", 2, "marked_dot.png"),
    (7, 2, "v", 4, "marked_lower_v.png"),
    (6, 18, "-", 3, "marked_dash.png"),
    (7, 27, "'", 1, "marked_apostrophe.png"),
    (7, 13, "g", 4, "marked_lower_g.png"),
    (5, 23, "T", 1, "marked_upper_T.png")
]

def merge_images(image_files, out_dir, merged_output_path):
    if not image_files:
        print("No images to merge.")
        return

    # Load the base image
    base_image_path = os.path.join(out_dir, image_files[0])
    result_image = cv2.imread(base_image_path, cv2.IMREAD_GRAYSCALE)

    if result_image is None:
        print(f"Error: Unable to load base image {base_image_path}")
        return

    # Merge subsequent images
    for image_file in image_files[1:]:
        image_path = os.path.join(out_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Warning: Unable to load image {image_path}. Skipping.")
            continue

        # Normalize to float for pixel-wise multiplication
        result_float = result_image.astype(float) / 255.0
        image_float = image.astype(float) / 255.0

        # Perform overlay operation
        result_float = result_float * image_float

        # Convert back to uint8
        result_image = (result_float * 255.0).astype(np.uint8)

    # Save the merged image
    cv2.imwrite(merged_output_path, result_image)
    print(f"Merged image saved at: {merged_output_path}")

os.makedirs(out_img_dir, exist_ok=True)

myAnalysis = OCRanalysis()
threshold = 0.999
print(f"OCR Analysis Results with threshold {threshold}:")
for (row, col, letter, expected_count, out_img_path) in brainrot_characters:
    full_out_img_path = os.path.join(out_img_dir, out_img_path)

    actual_count = myAnalysis.run(img_path, full_out_img_path, row, col, threshold, shrink_chars=True, only_use_simple_features=False)
    result = "OK" if actual_count == expected_count else "ERROR"
    print(f"Letter: \'{letter}\', Expected Count: {expected_count}, Actual Count: {actual_count}, Result: {result}")

print("Merging images...")

image_files = [f for f in os.listdir(out_img_dir)]
merge_images(image_files, out_img_dir, merged_img_path)

# temp = ["marked_lower_r.png", "marked_lower_t.png"]
# merge_images(temp, out_img_dir, merged_img_path)
