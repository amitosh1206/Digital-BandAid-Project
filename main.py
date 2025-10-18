import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import glob

def apply_band_aid(image_path, band_aid_path, show_result=True):
    """
    Apply a band-aid to an arm in the image.
    
    Args:
        image_path (str): Path to the input image
        band_aid_path (str): Path to the band-aid image
        show_result (bool): Whether to show the result window
    
    Returns:
        tuple: (original image, result image) or None if error
    """
    print(f"\nProcessing image: {image_path}")
    
    # Initialize Mediapipe Hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
image = cv2.imread(image_path)

if image is None:
    print("❌ Error: 'original_arm.png' not found or invalid path.")
    sys.exit(1)

# Convert BGR to RGB for Mediapipe
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect hands
results = hands.process(image_rgb)

# Load the band-aid image (✅ Fixed Path)
band_aid = cv2.imread('image/band_aid.png', cv2.IMREAD_UNCHANGED)

if band_aid is None:
    print("❌ Error: 'band_aid.png' not found or invalid path.")
    sys.exit(1)

# ✅ Check if band-aid image has alpha channel; if not, convert it
if band_aid.ndim == 2:
    # single channel -> convert to BGRA
    band_aid = cv2.cvtColor(band_aid, cv2.COLOR_GRAY2BGRA)

elif band_aid.shape[2] == 3:  # BGR only, add alpha
    print("ℹ️ No alpha channel detected in band-aid image. Adding one automatically.")
    b, g, r = cv2.split(band_aid)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    band_aid = cv2.merge((b, g, r, alpha))

if not results.multi_hand_landmarks:
    print("⚠️ No hand detected. Please try another image.")
else:
    for hand_landmarks in results.multi_hand_landmarks:
        # Get coordinates of wrist and middle finger base
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

        h, w, _ = image.shape
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        middle_x, middle_y = int(middle_base.x * w), int(middle_base.y * h)

        # Band-aid placement region
        center_x = (wrist_x + middle_x) // 2
        center_y = (wrist_y + middle_y) // 2

        band_aid_width = int(max(abs(middle_x - wrist_x) * 1.2, 1))
        band_aid_height = int(max(band_aid_width * 0.4, 1))

        # Protect against invalid resize dimensions
        if band_aid_width <= 0 or band_aid_height <= 0:
            print("⚠️ Computed band-aid size is invalid, skipping overlay for this hand.")
            continue

        band_aid_resized = cv2.resize(band_aid, (band_aid_width, band_aid_height))

        # Overlay band-aid with transparency
        y1 = max(center_y - band_aid_height // 2, 0)
        y2 = min(y1 + band_aid_height, image.shape[0])
        x1 = max(center_x - band_aid_width // 2, 0)
        x2 = min(x1 + band_aid_width, image.shape[1])

        band_aid_resized = band_aid_resized[: max(0, y2 - y1), : max(0, x2 - x1)]

        # If cropping results in empty image, skip
        if band_aid_resized.size == 0:
            print("⚠️ Band-aid crop is empty after bounds clipping; skipping overlay.")
            continue

        # Split BGRA and create mask
        b, g, r, a = cv2.split(band_aid_resized)
        overlay_color = cv2.merge((b, g, r)).astype(float)
        mask = cv2.merge((a, a, a)).astype(float) / 255.0

        # Extract region of interest and blend safely using float arithmetic
        roi = image[y1:y2, x1:x2].astype(float)
        blended = (1 - mask) * roi + mask * overlay_color
        # Clip and convert back to uint8
        image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        # Draw landmarks for visualization (optional)
        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Save and show result
    cv2.imwrite('output_with_bandaid.png', image)
    try:
        # Showing images may fail in headless environments; suppress errors
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass

    # close mediapipe resources
    try:
        hands.close()
    except Exception:
        pass

print("✅ Band-aid applied successfully and saved as 'output_with_bandaid.png'")