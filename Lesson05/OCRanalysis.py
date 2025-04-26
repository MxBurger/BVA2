import cv2
import numpy as np
import argparse

import ImageFeatures
from SubImageRegion import SubImageRegion

logging_enabled = False


def log(*string: str):
    if logging_enabled:
        print(string)


class OCRanalysis:
    def __init__(self):
        self.F_FGcount = 0
        self.F_MaxDistX = 1
        self.F_MaxDistY = 2
        self.F_AvgDistanceCentroide = 3
        self.F_MaxDistanceCentroide = 4
        self.F_MinDistanceCentroide = 5
        self.F_Circularity = 6
        self.F_CentroideRelPosX = 7
        self.F_CentroideRelPosY = 8
        self.F_HoleCount = 9

    def run(self, img_in_path: str, img_out_path: str, tgtCharRow: int, tgtCharCol: int, threshold: float,
            shrink_chars: bool):
        img = cv2.imread(img_in_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Fehler: Konnte Bild '{img_in_path}' nicht laden.")
            return

        height, width = img.shape
        FG_VAL = 0
        BG_VAL = 255
        MARKER_VAL = 127
        thresholdVal = 127

        _, binaryImgArr = cv2.threshold(img, thresholdVal, BG_VAL, cv2.THRESH_BINARY)
        cv2.imwrite("binaryOut.png", binaryImgArr)

        # define the features to evaluate
        features_to_use = []
        features_to_use.append(ImageFeatures.ImageFeatureF_FGcount())
        features_to_use.append(ImageFeatures.ImageFeatureF_MaxDistX())
        features_to_use.append(ImageFeatures.ImageFeatureF_MaxDistY())
        features_to_use.append(ImageFeatures.ImageFeatureF_AvgDistanceCentroide())
        features_to_use.append(ImageFeatures.ImageFeatureF_MaxDistanceCentroide())
        features_to_use.append(ImageFeatures.ImageFeatureF_MinDistanceCentroide())
        features_to_use.append(ImageFeatures.ImageFeatureF_Circularity())
        features_to_use.append(ImageFeatures.ImageFeatureF_CentroideRelPosX())
        features_to_use.append(ImageFeatures.ImageFeatureF_CentroideRelPosY())
        features_to_use.append(ImageFeatures.ImageFeatureF_HoleCount())

        log("Starte Zeichenerkennung...")
        linked_regions, lines = split_characters(binaryImgArr, width, height, BG_VAL, FG_VAL, shrink_chars)
        log(f"Gefundene Zeilen: {lines}")
        if lines > 0:
            log(f"Zeichen in 7. Zeile: {len(linked_regions[6]) if linked_regions else 0}")

        # define the reference character
        charROI = linked_regions[tgtCharRow][tgtCharCol]

        # test calculate features
        log('features of reference character is: ')
        feature_res_arr = calc_feature_arr(charROI, BG_VAL, features_to_use)
        self.printout_feature_res(feature_res_arr, features_to_use)

        # then normalize
        feature_norm_arr = calculate_norm_arr(linked_regions, BG_VAL, features_to_use)
        log('NORMALIZED features: ')
        self.printout_feature_res(feature_norm_arr, features_to_use)

        # now check all characters and test, if similarity with reference letter is given:
        # assuming that hitCount, binaryImgArr, FG_VAL, MARKER_VAL are defined elsewhere globally
        # as they're not defined in the current code provided

        hitCount = 0  # make sure to initialize hitCount

        binary_img_arr = binaryImgArr.copy()
        for i in range(len(linked_regions)):
            for j in range(len(linked_regions[i])):
                img_reg = linked_regions[i][j]
                curr_feature_arr = calc_feature_arr(img_reg, BG_VAL, features_to_use)
                is_target_char = is_matching_char(curr_feature_arr, feature_res_arr, feature_norm_arr, threshold)
                if is_target_char:
                    hitCount += 1
                    binary_img_arr = self.mark_region_in_image(binary_img_arr, img_reg, BG_VAL, MARKER_VAL)

        cv2.imwrite(img_out_path, binary_img_arr)
        log('num of found characters is = ' + str(hitCount))

        return hitCount

    def printout_feature_res(feature_res_arr, features_to_use):
        log("========== features =========")
        for i in range(len(features_to_use)):
            log("res of F " + str(i) + ", " + features_to_use[i].description + " is " + str(feature_res_arr[i]))

    def mark_region_in_image(self, in_img_arr, img_region, color_to_replace, tgt_color):
        for x in range(img_region.width):
            for y in range(img_region.height):
                if img_region.subImgArr[y][x] == color_to_replace:
                    in_img_arr[y + img_region.startY][x + img_region.startX] = tgt_color
        return in_img_arr

    def printout_feature_res(self, feature_res_arr, features_to_use):
        log("========== features =========")
        for i in range(len(features_to_use)):
            log("res of F", i, ",", features_to_use[i].description, "is", feature_res_arr[i])


def is_empty_column(in_img, height, col_idx, BG_val):
    for row_idx in range(height):
        if in_img[row_idx][col_idx] != BG_val:
            return False
    return True


def is_empty_row(in_img, width, row_idx, BG_val):
    for col_idx in range(width):
        if in_img[row_idx][col_idx] != BG_val:
            return False
    return True


def split_characters_vertically(row_image, BG_val, FG_val, orig_img, shrink_chars):
    return_char_arr = []
    height = row_image.height
    width = row_image.width
    start_col = 0

    col_idx = 0
    while col_idx < width:
        while col_idx < width and is_empty_column(row_image.subImgArr, height, col_idx, BG_val):
            col_idx += 1

        if col_idx >= width:
            break

        start_col = col_idx  # start of the character found

        # find the end of the character (next empty column)
        while col_idx < width and not is_empty_column(row_image.subImgArr, height, col_idx, BG_val):
            col_idx += 1

        end_col = col_idx  # end of the character found
        char_width = end_col - start_col

        # Skip too narrow characters (probably noise)
        min_char_width = 2
        if char_width >= min_char_width:
            # Create a temporary subimage with just the character width
            width_limited_char_region = SubImageRegion(
                row_image.startX + start_col,
                row_image.startY,
                char_width,
                height,
                orig_img
            )

            if shrink_chars:
                # Now limit the character vertically
                char_start_y, char_height = limit_character_vertically(width_limited_char_region, BG_val)

                # Create the final character region with proper horizontal bounds
                limited_char_region = SubImageRegion(
                    row_image.startX + start_col,
                    row_image.startY + char_start_y,
                    char_width,
                    char_height,
                    orig_img
                )

                return_char_arr.append(limited_char_region)
            else:
                return_char_arr.append(width_limited_char_region)

    return return_char_arr


def limit_character_vertically(char_region, BG_val):
    """
    Find the top and bottom bounds of the character within the region.
    Returns start_y and height of the actual character.
    """
    height = char_region.height
    width = char_region.width

    # Find the first non-empty row from top
    start_y = 0
    while start_y < height:
        empty_row = True
        for x in range(width):
            if char_region.subImgArr[start_y][x] != BG_val:
                empty_row = False
                break
        if not empty_row:
            break
        start_y += 1

    # Find the first non-empty row from bottom
    end_y = height - 1
    while end_y >= start_y:
        empty_row = True
        for x in range(width):
            if char_region.subImgArr[end_y][x] != BG_val:
                empty_row = False
                break
        if not empty_row:
            break
        end_y -= 1

    # Calculate actual character height
    char_height = end_y - start_y + 1

    # Ensure minimum height
    if char_height < 2:
        char_height = 2
        if start_y + char_height > height:
            start_y = height - char_height

    return start_y, char_height


def split_characters(in_img, width, height, BG_val, FG_val, shrink_chars):
    return_char_matrix = []
    row_idx = 0
    while row_idx < height:
        # skip empty rows (background)
        while row_idx < height and is_empty_row(in_img, width, row_idx, BG_val):
            row_idx += 1

        # end of image reached
        if row_idx >= height:
            break

        start_row = row_idx  # start of the text line found

        #
        while row_idx < height and not is_empty_row(in_img, width, row_idx, BG_val):
            row_idx += 1

        end_row = row_idx  # End of the text line found
        line_height = end_row - start_row

        min_line_height = 2
        # Skip too narrow lines (probably noise)
        if line_height >= min_line_height:

            # Create a sub-image region for the line
            line_region = SubImageRegion(0, start_row, width, line_height, in_img)

            # Split the line into characters
            chars_in_line = split_characters_vertically(line_region, BG_val, FG_val, in_img, shrink_chars)

            # Add the characters to the return matrix
            if chars_in_line:
                return_char_matrix.append(chars_in_line)

    line_count = len(return_char_matrix)
    return return_char_matrix, line_count


def calculate_norm_arr(input_regions, FG_val, features_to_use):
    # calculate the average per feature to allow for normalization
    return_arr = [0] * len(features_to_use)
    num_of_regions = 0

    for i in range(len(input_regions)):
        curr_row = input_regions[i]
        for j in range(len(curr_row)):
            curr_feature_vals = calc_feature_arr(curr_row[j], FG_val, features_to_use)
            for k in range(len(return_arr)):
                return_arr[k] += curr_feature_vals[k]

            num_of_regions += 1

    for k in range(len(return_arr)):
        return_arr[k] /= num_of_regions if num_of_regions > 0 else 1

    return return_arr


def calc_feature_arr(region, FG_val, features_to_use):
    feature_res_arr = [0] * len(features_to_use)
    for i in range(len(features_to_use)):
        curr_feature_val = features_to_use[i].CalcFeatureVal(region, FG_val)
        feature_res_arr[i] = curr_feature_val

    return feature_res_arr


def is_matching_char(curr_feature_arr, ref_feature_arr, norm_feature_arr, threshold: float):
    norm_curr_arr = []
    norm_ref_arr = []

    for i in range(len(curr_feature_arr)):
        if norm_feature_arr[i] != 0:  # avoid division by zero
            norm_curr = curr_feature_arr[i] / norm_feature_arr[i]
            norm_ref = ref_feature_arr[i] / norm_feature_arr[i]
        else:
            norm_curr = curr_feature_arr[i]
            norm_ref = ref_feature_arr[i]

        norm_curr_arr.append(norm_curr)
        norm_ref_arr.append(norm_ref)

    mean_curr = sum(norm_curr_arr) / len(norm_curr_arr)
    mean_ref = sum(norm_ref_arr) / len(norm_ref_arr)

    numerator = 0
    denominator_curr = 0
    denominator_ref = 0

    for i in range(len(norm_curr_arr)):
        diff_curr = norm_curr_arr[i] - mean_curr
        diff_ref = norm_ref_arr[i] - mean_ref

        numerator += diff_curr * diff_ref
        denominator_curr += diff_curr * diff_curr
        denominator_ref += diff_ref * diff_ref

    # correlation coefficient
    if denominator_curr > 0 and denominator_ref > 0:
        correlation_coefficient = numerator / (np.sqrt(denominator_curr) * np.sqrt(denominator_ref))
    else:
        correlation_coefficient = 0  # avoid division by zero

    if correlation_coefficient > threshold:
        return True

    return False


def main(img_in_path: str, img_out_path: str, row: int, col: int, threshold: float, shrink_chars: bool):
    print("OCR")
    myAnalysis = OCRanalysis()
    myAnalysis.run(img_in_path, img_out_path, row, col, threshold, shrink_chars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Character Matcher")

    parser.add_argument("--image_in_path", type=str, default="altesTestament_ArialBlack.png", help="Input image path")
    parser.add_argument("--image_out_path", type=str, default="markedChars.png", help="Output image path")
    parser.add_argument("--row", "-r", type=int, default=1, help="Row of target character (0-based) -> default: 1")
    parser.add_argument("--column", "-c", type=int, default=3,
                        help="Column of target character (0-based) -> default: 3")
    parser.add_argument("--threshold", "-t", type=float, default=0.999,
                        help="Correlation coefficient limit (confidence) -> default: 0.999")
    parser.add_argument("--logging", "-l", type=bool, default=False, help="Enable logging -> default: False")
    parser.add_argument("--shrink_chars", "-s", type=bool, default=False, help="Shrink characters -> default: False")

    args = parser.parse_args()

    logging_enabled = args.logging
    main(args.image_in_path, args.image_out_path, args.row, args.column, args.threshold, args.shrink_chars)