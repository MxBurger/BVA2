import cv2
import numpy as np
import os


def main():
    # Input video (0 = webcam)
    video_path = "./vtest.avi"

    # Create output directory if it doesn't exist
    output_dir = "img/task2"
    os.makedirs(output_dir, exist_ok=True)

    # Activate Erosion and Dilatation
    apply_morphology = False  # Set True to activate Erosion and Dilatation Post-Processing
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion_iterations = 1
    dilation_iterations = 2

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # OpenCV background subtractors
    subtractors = {
        #'MOG2': cv2.createBackgroundSubtractorMOG2(
        #  history=90,  # Number of frames for background model
        #  varThreshold=25,  # Threshold for foreground detection
        #  detectShadows=False  # Don't detect shadows (faster)
        #),
        'KNN': cv2.createBackgroundSubtractorKNN(
          history=90,  # Number of frames for background model
          dist2Threshold=1000.0,  # Threshold for foreground detection
          detectShadows=True  # Don't detect shadows (faster)
        ),
        # 'CNT': cv2.bgsegm.createBackgroundSubtractorCNT(
        #   minPixelStability=10,  # Min frames for pixel stability
        #   useHistory=True,  # Use history model
        #   maxPixelStability=15 * 60  # Max frames for pixel stability
        # ),
        #'GMG': cv2.bgsegm.createBackgroundSubtractorGMG(
        #    initializationFrames=30,  # Frames for initialization
        #    decisionThreshold=0.85  # Threshold for foreground detection
        #)
    }

    # Initialize heat maps for each subtractor
    heat_maps = {name: np.zeros((height, width), dtype=np.float32)
                 for name in subtractors.keys()}

    # Dictionaries to store mask overlays for each subtractor
    mask_overlays = {name: None for name in subtractors.keys()}

    print("Processing video with different background subtractors...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process each subtractor
        for name, subtractor in subtractors.items():
            # Apply background subtraction to get foreground mask
            fg_mask = subtractor.apply(frame)

            # Apply Erosion and Dilatation, if activated
            if apply_morphology:
                # Reduce noise with Erosion
                fg_mask = cv2.erode(fg_mask, kernel, iterations=erosion_iterations)
                # Fill gaps with Dilatation
                fg_mask = cv2.dilate(fg_mask, kernel, iterations=dilation_iterations)

            # Update heat map
            heat_maps[name] += fg_mask.astype(np.float32) / 255.0

            # Normalize heat map for visualization
            if np.max(heat_maps[name]) > 0:
                normalized_heat_map = heat_maps[name] / np.max(heat_maps[name]) * 255
                normalized_heat_map = np.uint8(normalized_heat_map)
            else:
                normalized_heat_map = np.zeros_like(heat_maps[name], dtype=np.uint8)

            heat_map_display = cv2.applyColorMap(normalized_heat_map, cv2.COLORMAP_JET)

            # Create mask overlay on original frame
            overlay = frame.copy()
            # Create a red mask where motion is detected
            red_mask = np.zeros_like(frame)
            red_mask[fg_mask == 255] = [0, 0, 255]  # BGR => red
            # Add the red mask to the original frame with transparency
            alpha = 0.5
            mask_overlay = cv2.addWeighted(overlay, 1, red_mask, alpha, 0)

            # Store the current mask overlay
            mask_overlays[name] = mask_overlay

            # Show results for this subtractor
            cv2.imshow(f"{name} - Binary Motion Mask", fg_mask)
            cv2.imshow(f"{name} - Heat Map", heat_map_display)
            cv2.imshow(f"{name} - Mask Overlay", mask_overlay)

        cv2.imshow(f"Original Frame", frame)

        # Exit on 'q' press
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    # Save final heat maps and mask overlays as images
    for name, heat_map in heat_maps.items():
        if np.max(heat_map) > 0:
            normalized_heat_map = heat_map / np.max(heat_map) * 255
            normalized_heat_map = np.uint8(normalized_heat_map)
        else:
            normalized_heat_map = np.zeros_like(heat_map, dtype=np.uint8)

        heat_map_color = cv2.applyColorMap(normalized_heat_map, cv2.COLORMAP_JET)

        heat_map_path = f"{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_{name}_heatmap_morph.png"
        cv2.imwrite(heat_map_path, heat_map_color)
        print(f"Saved heat map: {heat_map_path}")

        # Save the last mask overlay for each subtractor
        mask_overlay_path = f"{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_{name}_overlay_morph.png"
        cv2.imwrite(mask_overlay_path, mask_overlays[name])
        print(f"Saved mask overlay: {mask_overlay_path}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()