import ImageFeatureBase
import math

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

def get_centroid(imgRegion, FG_val):
    """
    Helper function to calculate the centroid of a region in an image
    for the centroid based features.
    """
    totalX = 0
    totalY = 0
    count = 0

    # Calculate centroid
    for x in range(imgRegion.width):
        for y in range(imgRegion.height):
            if imgRegion.subImgArr[y][x] == FG_val:
                totalX += x
                totalY += y
                count += 1

    # No foreground pixels, return None for centroid
    if count == 0:
        return (None, None, None)

    centerX = totalX / count
    centerY = totalY / count

    return (centerX, centerY, count)

class ImageFeatureF_AvgDistanceCentroide(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "Durchschnittliche Distanz der Pixel zu den Centroiden"

    def CalcFeatureVal(self, imgRegion, FG_val):
        centroid_x, centroid_y, count = get_centroid(imgRegion, FG_val)
        if centroid_x is None:
            return 0

        total_distance = 0

        # calculate the distance of each pixel to the centroid
        for y in range(imgRegion.height):
            for x in range(imgRegion.width):
                if imgRegion.subImgArr[y][x] == FG_val:
                    # euclidean distance of pixel to the centroid
                    distance = ((x - centroid_x) ** 2 + (y - centroid_y) ** 2) ** 0.5
                    total_distance += distance

        return total_distance / count


class ImageFeatureF_MaxDistanceCentroide(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "Maximale Distanz der Pixel zu den Centroiden"

    def CalcFeatureVal(self, imgRegion, FG_val):
        centroid_x, centroid_y, _ = get_centroid(imgRegion, FG_val)
        if centroid_x is None:
            return 0

        max_distance = 0.0

        # find the maximum to centroid
        for y in range(imgRegion.height):
            for x in range(imgRegion.width):
                if imgRegion.subImgArr[y][x] == FG_val:
                    dx = x - centroid_x
                    dy = y - centroid_y

                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    if dist > max_distance:
                        max_distance = dist

        return max_distance


class ImageFeatureF_MaxDistanceCentroide(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "Maximale Distanz der Pixel zu den Centroiden"

    def CalcFeatureVal(self, imgRegion, FG_val):
        centroid_x, centroid_y, _ = get_centroid(imgRegion, FG_val)
        if centroid_x is None:
            return 0

        min_distance = float('inf')

        # find the maximum to centroid
        for y in range(imgRegion.height):
            for x in range(imgRegion.width):
                if imgRegion.subImgArr[y][x] == FG_val:
                    dx = x - centroid_x
                    dy = y - centroid_y

                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    if dist < min_distance:
                        min_distance = dist

        return min_distance


class ImageFeatureF_Circularity(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "Rundheit"

    def CalcFeatureVal(self, imgRegion, FG_val):
        area = 0
        for y in range(imgRegion.height):
            for x in range(imgRegion.width):
                if imgRegion.subImgArr[y][x] == FG_val:
                    area += 1

        # 0 roundness if no foreground pixels
        if area == 0:
            return 0

        # calculate the perimeter
        perimeter = 0
        for y in range(imgRegion.height):
            for x in range(imgRegion.width):
                if imgRegion.subImgArr[y][x] == FG_val:

                    # check the 8 neighbors if pixel is on the border
                    is_boundary = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy

                            # pixel out of the region bounds are considered background
                            if nx < 0 or nx >= imgRegion.width or ny < 0 or ny >= imgRegion.height:
                                is_boundary = True
                                break

                            # if neighbor is background, pixel is on the border
                            if imgRegion.subImgArr[ny][nx] != FG_val:
                                is_boundary = True
                                break

                        if is_boundary:
                            break

                    if is_boundary:
                        perimeter += 1

        if perimeter == 0:
            return 0

        # calculate the circularity
        # is defined as 4π × Area / Perimeter² => should be 1 for a perfect circle
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        return circularity


class ImageFeatureF_CentroideRelPosX(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "Relative X-Position des Zentroids"

    def CalcFeatureVal(self, imgRegion, FG_val):
        centroid_x, _, _ = get_centroid(imgRegion, FG_val)
        if (centroid_x is None) or (imgRegion.width == 0):
            return 0

        return centroid_x / imgRegion.width

class ImageFeatureF_CentroideRelPosY(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "Relative Y-Position des Zentroids"

    def CalcFeatureVal(self, imgRegion, FG_val):
        _, centroid_y, _ = get_centroid(imgRegion, FG_val)
        if (centroid_y is None) or (imgRegion.height == 0):
            return 0

        return centroid_y / imgRegion.height