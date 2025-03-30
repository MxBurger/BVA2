import cv2
import numpy as np
import os


def main():
    # Input video (0 = webcam)
    video_path = "./epic.mp4"

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Unable to open video source')
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create OpenCV background subtractors
    subtractors = {
        'MOG2': cv2.createBackgroundSubtractorMOG2(
            history=500,  # Number of frames for background model
            varThreshold=16,  # Threshold for foreground detection
            detectShadows=False  # Don't detect shadows (faster)
        ),
        'KNN': cv2.createBackgroundSubtractorKNN(
            history=500,  # Number of frames for background model
            dist2Threshold=400.0,  # Threshold for foreground detection
            detectShadows=False  # Don't detect shadows (faster)
        ),
        'CNT': cv2.bgsegm.createBackgroundSubtractorCNT(
            minPixelStability=15,  # Min frames for pixel stability
            useHistory=True,  # Use history model
            maxPixelStability=15 * 60  # Max frames for pixel stability
        ),
        'GMG': cv2.bgsegm.createBackgroundSubtractorGMG(
            initializationFrames=120,  # Frames for initialization
            decisionThreshold=0.8  # Threshold for foreground detection
        )
    }

    # Initialize heat maps for each subtractor
    heat_maps = {name: np.zeros((height, width), dtype=np.float32)
                 for name in subtractors.keys()}

    # Variables for frame counting
    frame_count = 0

    print("Processing video with different background subtractors...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process with each subtractor
        for name, subtractor in subtractors.items():
            # Apply background subtraction to get foreground mask
            fg_mask = subtractor.apply(frame)

            # Update heat map (accumulate motion intensity)
            heat_maps[name] += fg_mask.astype(np.float32) / 255.0

            # Normalize heat map for visualization
            if np.max(heat_maps[name]) > 0:
                normalized_heat_map = heat_maps[name] / np.max(heat_maps[name]) * 255
                normalized_heat_map = np.uint8(normalized_heat_map)
            else:
                normalized_heat_map = np.zeros_like(heat_maps[name], dtype=np.uint8)

            heat_map_display = cv2.applyColorMap(normalized_heat_map, cv2.COLORMAP_JET)

            # Show results for this subtractor
            cv2.imshow(f"{name} - Original Frame", frame)
            cv2.imshow(f"{name} - Binary Motion Mask", fg_mask)
            cv2.imshow(f"{name} - Heat Map", heat_map_display)

        frame_count += 1

        # Exit on 'q' press
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    # Save final heat maps as images
    for name, heat_map in heat_maps.items():
        if np.max(heat_map) > 0:
            normalized_heat_map = heat_map / np.max(heat_map) * 255
            normalized_heat_map = np.uint8(normalized_heat_map)
        else:
            normalized_heat_map = np.zeros_like(heat_map, dtype=np.uint8)

        heat_map_color = cv2.applyColorMap(normalized_heat_map, cv2.COLORMAP_JET)

        # Save the heat map
        heat_map_path = f"{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_{name}_heatmap.png"
        cv2.imwrite(heat_map_path, heat_map_color)
        print(f"Saved heat map: {heat_map_path}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
