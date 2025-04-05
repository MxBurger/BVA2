import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
from scipy import stats


# 1. load sample images
# source: https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F05ctyq
def load_images(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        images.append(img)
        filenames.append(filename)
    return images, filenames


# 2. use YOLO11 for ball localisation (https://docs.ultralytics.com/models/)
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

# 3. segment ball within the bounding box
def segment_ball_by_color(image, box):
    x1, y1, x2, y2 = box['xyxy']
    roi = image[y1:y2, x1:x2]

    # convert to hsv-model for better segmentation
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # bounds for tennis ball(yellowish-green)
    lower_yellow = np.array([25, 25, 25])
    upper_yellow = np.array([95, 255, 255])

    # create mask
    mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

    # reduce noise with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


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


def analyze_tennis_balls(directory):
    images, filenames = load_images(directory)
    print(f"Anzahl geladener Bilder: {len(images)}")

    model = YOLO("yolo11n.pt")  # oder ein anderes vortrainiertes Modell

    all_boxes = detect_balls_yolo(model, images)

    results = []

    for i, (img, boxes, filename) in enumerate(zip(images, all_boxes, filenames)):
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box['xyxy']
            yolo_center = box['center']

            mask = segment_ball_by_color(img, box)

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

                if i < 5:
                    plt.figure(figsize=(12, 6))

                    plt.subplot(1, 3, 1)
                    img_with_box = img.copy()
                    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(img_with_box, yolo_center, 5, (0, 0, 255), -1)
                    plt.imshow(img_with_box)
                    plt.title('YOLO Bounding Box')

                    plt.subplot(1, 3, 2)
                    plt.imshow(mask, cmap='gray')
                    plt.title('Segmentation mask')

                    plt.subplot(1, 3, 3)
                    img_with_centers = img.copy()
                    cv2.rectangle(img_with_centers, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(img_with_centers, yolo_center, 5, (0, 0, 255), -1)
                    cv2.circle(img_with_centers, centroid_global, 5, (0, 255, 0), -1)
                    plt.imshow(img_with_centers)
                    plt.title('Both centers')

                    plt.tight_layout()
                    plt.show()

    return results


def statistical_analysis(results):
    if not results:
        print("Keine Ergebnisse zur Analyse gefunden.")
        return

    df = pd.DataFrame(results)

    stats_desc = df[['difference_x', 'difference_y', 'euclidean_distance']].describe()
    print("Deskriptive Statistik:")
    print(stats_desc)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(df['difference_x'], bins=20)
    plt.title('Differenz in X-Richtung')

    plt.subplot(1, 3, 2)
    plt.hist(df['difference_y'], bins=20)
    plt.title('Differenz in Y-Richtung')

    plt.subplot(1, 3, 3)
    plt.hist(df['euclidean_distance'], bins=20)
    plt.title('Euklidische Distanz')

    plt.tight_layout()
    plt.show()

    t_stat_x, p_val_x = stats.ttest_1samp(df['difference_x'], 0)
    t_stat_y, p_val_y = stats.ttest_1samp(df['difference_y'], 0)

    print(f"t-Test für X-Differenz: t={t_stat_x:.4f}, p={p_val_x:.4f}")
    print(f"t-Test für Y-Differenz: t={t_stat_y:.4f}, p={p_val_y:.4f}")

    plt.figure(figsize=(8, 8))
    for i, row in df.iterrows():
        yolo_x, yolo_y = row['yolo_center']
        seg_x, seg_y = row['segmentation_centroid']
        plt.scatter(yolo_x, yolo_y, color='red', label='YOLO' if i == 0 else "")
        plt.scatter(seg_x, seg_y, color='green', label='Segmentierung' if i == 0 else "")
        plt.plot([yolo_x, seg_x], [yolo_y, seg_y], 'k-', alpha=0.3)

    plt.xlabel('X-Koordinate')
    plt.ylabel('Y-Koordinate')
    plt.title('Vergleich der Zentren aus YOLO und Segmentierung')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    directory = "images"
    results = analyze_tennis_balls(directory)
    statistical_analysis(results)