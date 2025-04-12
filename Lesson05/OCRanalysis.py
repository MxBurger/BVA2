import cv2
import numpy as np

import ImageFeatureBase
from SubImageRegion import SubImageRegion


class OCRanalysis:
    def __init__(self):
        self.F_FGcount = 0
        self.F_MaxDistX = 1
        self.F_MaxDistY = 2
        # self.F_AvgDistanceCentroide= 3
        # self.F_MaxDistanceCentroide= 4
        # self.F_MinDistanceCentroide= 5
        # self.F_Circularity = 6
        # self.F_CentroideRelPosX = 7
        # self.F_CentroideRelPosY = 8

    def run(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Fehler: Konnte Bild '{img_path}' nicht laden.")
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
        features_to_use.append(ImageFeatureF_FGcount())
        features_to_use.append(ImageFeatureF_MaxDistX())
        features_to_use.append(ImageFeatureF_MaxDistY())

        print("Starte Zeichenerkennung...")
        linked_regions, lines = split_characters(binaryImgArr, width, height, BG_VAL, FG_VAL)
        print(f"Gefundene Zeilen: {lines}")
        if lines > 0:
            print(f"Zeichen in erster Zeile: {len(linked_regions[0]) if linked_regions else 0}")

        # define the reference character
        tgtCharRow = 2
        tgtCharCol = 3
        charROI = linked_regions[tgtCharRow][tgtCharCol]

        # test calculate features
        print('features of reference character is: ')
        feature_res_arr = calc_feature_arr(charROI, BG_VAL, features_to_use)
        self.printout_feature_res(feature_res_arr, features_to_use)

        # then normalize
        feature_norm_arr = calculate_norm_arr(linked_regions, BG_VAL, features_to_use)
        print('NORMALIZED features: ')
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
                is_target_char = is_matching_char(curr_feature_arr, feature_res_arr, feature_norm_arr)
                if is_target_char:
                    hitCount += 1
                    binary_img_arr = self.mark_region_in_image(binary_img_arr, img_reg, BG_VAL, MARKER_VAL)

        # TODO: printout result image with all the marked letters
        cv2.imwrite("markedChars.png", binary_img_arr)
        print('num of found characters is = ' + str(hitCount))

    def printout_feature_res(feature_res_arr, features_to_use):
        print("========== features =========")
        for i in range(len(features_to_use)):
            print("res of F " + str(i) + ", " + features_to_use[i].description + " is " + str(feature_res_arr[i]))

    def mark_region_in_image(self, in_img_arr, img_region, color_to_replace, tgt_color):
        adjustedColors = 0
        for x in range(img_region.width):
            for y in range(img_region.height):
                if img_region.subImgArr[y][x] == color_to_replace:
                    in_img_arr[y + img_region.startY][x + img_region.startX] = tgt_color
                    adjustedColors += 1
        print('adjusted colors is ' + str(adjustedColors))
        return in_img_arr

    def printout_feature_res(self, feature_res_arr, features_to_use):
        print("========== features =========")
        for i in range(len(features_to_use)):
            print("res of F", i, ",", features_to_use[i].description, "is", feature_res_arr[i])


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


def split_characters_vertically(row_image, BG_val, FG_val, orig_img):
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

        if char_width >= 2:          # skip too narrow characters (probably noise)
            char_region = SubImageRegion(
                row_image.startX + start_col,
                row_image.startY,
                char_width,
                height,
                orig_img
            )
            return_char_arr.append(char_region)

    return return_char_arr


def split_characters(in_img, width, height, BG_val, FG_val):
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

        end_row = row_idx  # Ende der Textzeile gefunden

        # Berechne die Höhe der Textzeile
        line_height = end_row - start_row

        # Überspringe zu niedrige Zeilen (vermutlich Rauschen)
        if line_height >= 2:  # Minimale Höhe für eine gültige Textzeile
            # Extrahiere die Zeile als SubImageRegion
            line_region = SubImageRegion(0, start_row, width, line_height, in_img)

            # Zerlege die Zeile in einzelne Zeichen
            chars_in_line = split_characters_vertically(line_region, BG_val, FG_val, in_img)

            # Füge die Zeile zur Matrix hinzu, wenn Zeichen gefunden wurden
            if chars_in_line:
                return_char_matrix.append(chars_in_line)

    lines = len(return_char_matrix)  # Anzahl der gefundenen Zeilen
    return return_char_matrix, lines


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


def is_matching_char(curr_feature_arr, ref_feature_arr, norm_feature_arr):
    CORR_COEFFICIENT_LIMIT = 0.999

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

    if correlation_coefficient > CORR_COEFFICIENT_LIMIT:
        return True

    return False


class ImageFeatureF_FGcount(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "Pixelanzahl"

    def CalcFeatureVal(self, imgRegion, FG_val):
        count = 0
        for x in range(imgRegion.width):
            for y in range(imgRegion.height):
                if imgRegion.subImgArr[y][x] == FG_val:
                    count += 1
        return count


class ImageFeatureF_MaxDistX(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "maximale Ausdehnung in X-Richtung"

    def CalcFeatureVal(self, imgRegion, FG_val):
        return imgRegion.width


class ImageFeatureF_MaxDistY(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "maximale Ausdehnung in Y-Richtung"

    def CalcFeatureVal(self, imgRegion, FG_val):
        return imgRegion.height


def main():
    print("OCR")
    inImgPath = "altesTestament_ArialBlack.png"
    myAnalysis = OCRanalysis()
    myAnalysis.run(inImgPath)


if __name__ == "__main__":
    main()