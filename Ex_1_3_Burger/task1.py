from time import sleep

import cv2
import numpy as np
import os


def main():
    # Input video (0 = webcam)
    video_path = "./vtest.avi"

    bg_learning_frames = 30  # Number of frames to use for initial background model
    bg_learning_rate = 0.01  # Background update rate (0-1)
    motion_threshold = 25  # Threshold for motion detection

    # Create output directory if it doesn't exist
    output_dir = "img/task1"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Unable to open video source')
        return

    frame_count = 0
    frame_buffer = []

    print("First stage: collect frames for background initialization")
    while frame_count < bg_learning_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        frame_buffer.append(gray_frame)
        frame_count += 1

    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    # Create initial background model by median filtering the buffer
    if len(frame_buffer) > 0:
        background = np.median(np.array(frame_buffer), axis=0).astype(np.float32)
        print(f"Background model created from {len(frame_buffer)} frames")
    else:
        print("Error: Could not initialize background model")
        return

    # Initialize heat map
    heat_map = np.zeros_like(frame_buffer[0], dtype=np.float32)

    print("Second stage: process video with the initialized background")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Make a copy of the frame and convert to grayscale for processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        # Calculate absolute difference between background and current frame
        frame_delta = cv2.absdiff(background.astype("uint8"), gray_frame)

        # Threshold the delta image to get binary mask of moving areas
        thresh = cv2.threshold(frame_delta, motion_threshold, 255, cv2.THRESH_BINARY)[1]

        # Update heat map (accumulate motion intensity)
        heat_map = heat_map + thresh.astype(np.float32) / 255.0

        # Update background model
        # Check if the pixel is not part of a detected motion (to prevent including moving objects)
        mask = 255 - thresh # Invert
        if bg_learning_rate > 0:
            # Only update pixels that are not part of detected motion
            cv2.accumulateWeighted(gray_frame, background, bg_learning_rate, mask=mask)

        # Prepare visualization frames
        background_display = background.astype("uint8")

        # Create mask overlay on original frame
        overlay = frame.copy()
        # Create a red mask where motion is detected
        red_mask = np.zeros_like(frame)
        red_mask[thresh == 255] = [0, 0, 255]  # BGR => red
        # Add the red mask to the original frame with transparency
        alpha = 0.5  # Transparency factor
        mask_overlay = cv2.addWeighted(overlay, 1, red_mask, alpha, 0)

        # Normalize heat map for visualization
        if np.max(heat_map) > 0:
            # Scale values to use full range (0-255)
            normalized_heat_map = heat_map / np.max(heat_map) * 255
            # Convert to 8-bit unsigned integer format
            normalized_heat_map = np.uint8(normalized_heat_map)
        else:
            # If no motion detected yet, create an empty image
            normalized_heat_map = np.zeros_like(heat_map, dtype=np.uint8)

        heat_map_display = cv2.applyColorMap(normalized_heat_map, cv2.COLORMAP_JET)

        cv2.imshow("Original Frame", frame)
        cv2.imshow("Background Model", background_display)
        cv2.imshow("Binary Motion Mask", thresh)
        cv2.imshow("Heat Map", heat_map_display)
        cv2.imshow("Mask Overlay", mask_overlay)

        frame_count += 1

        # Exit on 'q' press
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    # Save
    original_path = f"{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_original.jpg"
    background_path = f"{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_background.jpg"
    thresh_path = f"{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_threshold.jpg"
    heat_map_path = f"{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_heat_map.jpg"
    mask_path = f"{output_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_mask.jpg"

    cv2.imwrite(original_path, frame)
    cv2.imwrite(background_path, background_display)
    cv2.imwrite(thresh_path, thresh)
    cv2.imwrite(heat_map_path, heat_map_display)
    cv2.imwrite(mask_path, mask_overlay)

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()