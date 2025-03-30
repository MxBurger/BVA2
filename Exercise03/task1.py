import cv2
import numpy as np


def main():
    # Input video (0 = webcam)
    video_path = "./vtest.avi"

    # Parameters
    bg_learning_frames = 30  # Number of frames to use for initial background model
    bg_learning_rate = 0.01  # Background update rate (0-1)
    motion_threshold = 25  # Threshold for motion detection
    erosion_size = 2  # Size for noise removal operations
    dilation_size = 5  # Size for filling gaps in detected motion

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Unable to open video source')
        return

    # Initialize variables
    frame_count = 0
    background = None
    heat_map = None
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

    # Reset video and variables
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

    # Create kernels for morphological operations
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))


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

        # Apply morphological operations to clean up the binary mask
        # First erode to remove small noise, then dilate to fill gaps
        thresh = cv2.erode(thresh, erosion_kernel, iterations=1)
        thresh = cv2.dilate(thresh, dilation_kernel, iterations=2)

        # Update heat map (accumulate motion intensity)
        heat_map = heat_map + thresh.astype(np.float32) / 255.0

        # Update background model
        # Check if the pixel is not part of a detected motion (to prevent assimilating moving objects)
        mask = cv2.bitwise_not(thresh)
        if bg_learning_rate > 0:
            # Only update pixels that are not part of detected motion
            cv2.accumulateWeighted(gray_frame, background, bg_learning_rate, mask=mask)

        # Prepare visualization frames
        background_display = background.astype("uint8")

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

        # Show
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Background Model", background_display)
        cv2.imshow("Binary Motion Mask", thresh)
        cv2.imshow("Heat Map", heat_map_display)

        frame_count += 1

        # Break loop if 'q' key is pressed
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()