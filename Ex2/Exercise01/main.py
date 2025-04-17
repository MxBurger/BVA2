import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO


# load sample images
# image-source: https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F05ctyq
def load_images(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        images.append(img)
        filenames.append(filename)
    return images, filenames


# use YOLO11 for ball localisation (https://docs.ultralytics.com/models/)
def detect_balls_yolo(model, images):
    all_boxes = []
    for img in images:
        results = model(img)
        boxes = []
        for r in results:
            for box in r.boxes:
                # only look at sports balls (coco-class 32)
                if box.cls == 32:  # (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    boxes.append({
                        'xyxy': (int(x1), int(y1), int(x2), int(y2)),
                        'conf': confidence,
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    })
        all_boxes.append(boxes)
    return all_boxes


# segment the ball within the bounding box using Hough Circle detection
# (https://theailearner.com/tag/cv2-houghcircles/)
def segment_ball(image, box):
    x1, y1, x2, y2 = box['xyxy']
    roi = image[y1:y2, x1:x2]
    # dimensions
    h, w = roi.shape[:2]
    # convert to grayscale and apply Gaussian blur
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    # parameters for HoughCircles depend on roi size
    min_dim = min(h, w)
    min_radius = max(min_dim // 5, 10)  # min radius is 1/5 of the smallest dimension
    max_radius = min_dim  # max radius is the size of the ROI

    # Hough Circle detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.8,
        minDist=min_dim,
        param1=100,  # sharpness of the edges
        param2=25,  # minimum number of votes to detect a circle
        minRadius=min_radius,
        maxRadius=max_radius
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    circles = np.uint16(np.around(circles))
    # use the first detected circle (highest confidence)
    circle = circles[0, 0]
    center_x, center_y, radius = circle
    # draw the circle on the mask
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    return mask


# find the centroid of the ball using the mask contour
# (https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/)
def find_centroid(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

# orchestrate the analysis
def analyze_tennis_balls(directory):
    images, filenames = load_images(directory)
    print(f"Found: {len(images)} images")
    model = YOLO("yolo11n.pt")
    all_boxes = detect_balls_yolo(model, images)
    results = []
    for i, (img, boxes, filename) in enumerate(zip(images, all_boxes, filenames)):
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box['xyxy']
            yolo_center = box['center']
            mask = segment_ball(img, box)
            centroid = find_centroid(mask)

            if centroid is not None:
                centroid_global = (centroid[0] + x1, centroid[1] + y1)
                results.append({
                    'filename': filename,
                    'ball_id': j,
                    'yolo_center': yolo_center,
                    'segmentation_centroid': centroid_global,
                    'difference_x': yolo_center[0] - centroid_global[0],
                    'difference_y': yolo_center[1] - centroid_global[1],
                    'euclidean_distance': np.sqrt((yolo_center[0] - centroid_global[0]) ** 2 +
                                                  (yolo_center[1] - centroid_global[1]) ** 2)
                })

                plt.figure(figsize=(16, 4))

                # Visualize YOLO bounding box
                plt.subplot(1, 4, 1)
                img_with_box = img.copy()
                cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(img_with_box, yolo_center, 5, (0, 0, 255), -1)
                plt.imshow(img_with_box)
                plt.title('YOLO Bounding Box')

                # Visualize segmentation mask
                plt.subplot(1, 4, 2)
                plt.imshow(mask, cmap='gray')
                plt.title('Segmentation mask')

                # Visualize mask over original roi
                plt.subplot(1, 4, 3)
                roi = img[y1:y2, x1:x2].copy()
                mask_colored = np.zeros_like(roi)
                mask_colored[mask > 0] = [255, 0, 255]
                alpha = 0.5
                overlay_roi = cv2.addWeighted(roi, 1, mask_colored, alpha, 0)
                plt.imshow(overlay_roi)
                plt.title('Mask Overlay')

                # Visualize both centers
                plt.subplot(1, 4, 4)
                img_with_centers = img.copy()
                cv2.rectangle(img_with_centers, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(img_with_centers, yolo_center, 5, (0, 0, 255), -1)  # YOLO in Rot
                cv2.circle(img_with_centers, centroid_global, 5, (0, 255, 0), -1)  # Segmentierung in Grün
                plt.imshow(img_with_centers)
                plt.title('Both centers')

                plt.tight_layout()
                plt.show()

    return results

# analyze the results
def statistical_analysis(results):
    distances = [result['euclidean_distance'] for result in results]

    min_distance = min(distances)
    max_distance = max(distances)
    avg_distance = sum(distances) / len(distances)

    # Median
    sorted_distances = sorted(distances)
    n = len(sorted_distances)
    if n % 2 == 0:
        # even number of result-sets
        median_distance = (sorted_distances[n // 2 - 1] + sorted_distances[n // 2]) / 2
    else:
        # odd number of result-sets
        median_distance = sorted_distances[n // 2]

    print("\nStats:")
    print(f"Minimum: {min_distance:.2f} Pixel")
    print(f"Maximum: {max_distance:.2f} Pixel")
    print(f"Average: {avg_distance:.2f} Pixel")
    print(f"Median: {median_distance:.2f} Pixel")
    print(f"Amount of analyzed Balls: {len(distances)}")

    # Visualisation
    plt.figure(figsize=(8, 8))
    for i, result in enumerate(results):
        yolo_x, yolo_y = result['yolo_center']
        seg_x, seg_y = result['segmentation_centroid']
        plt.scatter(yolo_x, yolo_y, color='red', label='YOLO' if i == 0 else "")
        plt.scatter(seg_x, seg_y, color='green', label='Segmentation' if i == 0 else "")
        plt.plot([yolo_x, seg_x], [yolo_y, seg_y], 'k-', alpha=0.3)

    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')
    plt.title('YOLO and segmentation centers in comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Histogram of euclidian distances
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=10, color='skyblue', edgecolor='black')
    plt.axvline(avg_distance, color='red', linestyle='dashed', linewidth=1, label=f'Average: {avg_distance:.2f}')
    plt.axvline(median_distance, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_distance:.2f}')
    plt.xlabel('Euclidean distance in pixels')
    plt.ylabel('Frequency')
    plt.title('Distribution of Euclidean Distances')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    directory = "images"
    results = analyze_tennis_balls(directory)
    statistical_analysis(results)
