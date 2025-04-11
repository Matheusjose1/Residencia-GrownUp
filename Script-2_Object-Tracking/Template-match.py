import cv2
import numpy as np

# Load the reference image
reference_image = cv2.imread('ExemploLixeira.png')

# Load the image to wrap onto the reference image
image_to_wrap = cv2.imread('ExemploLixeira2.png')

# Resize the image to match the size of the reference image
image_to_wrap_resized = cv2.resize(image_to_wrap, (reference_image.shape[1], reference_image.shape[0]))

# Convert reference image to grayscale
gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for the reference image
keypoints_reference, descriptors_reference = sift.detectAndCompute(gray_reference, None)

# Initialize the camera or video stream
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for the frame
    keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)

    if descriptors_frame is not None:
        # Match descriptors between the frame and the reference image
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_reference, descriptors_frame, k=2)

        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            # Get the matched keypoints
            src_pts = np.float32([keypoints_reference[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculate homography
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                # Warp the image to fit the perspective of the reference image
                warped_image = cv2.warpPerspective(image_to_wrap_resized, M, (frame.shape[1], frame.shape[0]))

                # Overlay the warped image onto the frame
                frame_with_overlay = frame.copy()
                frame_with_overlay[np.where(warped_image[:, :, 0] != 0)] = warped_image[
                    np.where(warped_image[:, :, 0] != 0)]

                # Display the frame with overlay
                cv2.imshow('Augmented Reality', frame_with_overlay)
            else:
                cv2.imshow('Augmented Reality', frame)

        else:
            cv2.imshow('Augmented Reality', frame)

    else:
        cv2.imshow('Augmented Reality', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()