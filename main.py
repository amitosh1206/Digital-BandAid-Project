import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Load the input image (✅ Fixed Path)
image_path = 'image/original_arm.png'   # Image folder ke andar rakha hua hai
image = cv2.imread(image_path)

if image is None:
    print("❌ Error: 'original_arm.png' not found or invalid path.")
    exit()

# Convert BGR to RGB for Mediapipe
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect hands
results = hands.process(image_rgb)

# Load the band-aid image (✅ Fixed Path)
band_aid = cv2.imread('image/band_aid.png', cv2.IMREAD_UNCHANGED)

if band_aid is None:
    print("❌ Error: 'band_aid.png' not found or invalid path.")
    exit()

# ✅ Check if band-aid image has alpha channel; if not, convert it
if band_aid.shape[2] == 3:  # RGB only, add alpha
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

        band_aid_width = int(abs(middle_x - wrist_x) * 1.2)
        band_aid_height = int(band_aid_width * 0.4)
        band_aid_resized = cv2.resize(band_aid, (band_aid_width, band_aid_height))

        # Overlay band-aid with transparency
        y1 = max(center_y - band_aid_height // 2, 0)
        y2 = min(y1 + band_aid_height, image.shape[0])
        x1 = max(center_x - band_aid_width // 2, 0)
        x2 = min(x1 + band_aid_width, image.shape[1])

        band_aid_resized = band_aid_resized[:y2 - y1, :x2 - x1]

        b, g, r, a = cv2.split(band_aid_resized)
        overlay_color = cv2.merge((b, g, r))
        mask = cv2.merge((a, a, a)) / 255.0

        image[y1:y2, x1:x2] = (1 - mask) * image[y1:y2, x1:x2] + mask * overlay_color

        # Draw landmarks for visualization (optional)
        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Save and show result
    cv2.imwrite('output_with_bandaid.png', image)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("✅ Band-aid applied successfully and saved as 'output_with_bandaid.png'")
b