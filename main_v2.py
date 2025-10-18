import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import glob

def apply_band_aid(image_path, band_aid_path, output_path=None, show_result=True):
    """Process an image to detect hands and apply a band-aid overlay.
    
    Args:
        image_path (str): Path to input image with an arm/hand
        band_aid_path (str): Path to band-aid image (with transparency if possible)
        output_path (str, optional): Where to save the result
        show_result (bool): Whether to display the result window
    
    Returns:
        tuple: (original image, result image) or None if failed
    """
    print(f"\nProcessing: {os.path.basename(image_path)}")
    
    # Initialize Mediapipe with better detection settings
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.2,  # Even lower threshold for better detection
        min_tracking_confidence=0.2,
        model_complexity=1  # Using more complex model for better detection
    )
    mp_draw = mp.solutions.drawing_utils

    # Load input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image: {image_path}")
        return None
    
    # Make a copy for the final comparison
    original = image.copy()
    
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load band-aid image
    band_aid = cv2.imread(band_aid_path, cv2.IMREAD_UNCHANGED)
    if band_aid is None:
        print(f"❌ Error: Could not load band-aid: {band_aid_path}")
        return None

    # Ensure band-aid has alpha channel
    if band_aid.ndim == 2:  # Grayscale
        band_aid = cv2.cvtColor(band_aid, cv2.COLOR_GRAY2BGRA)
    elif band_aid.shape[2] == 3:  # BGR
        print("ℹ️ Adding alpha channel to band-aid image")
        b, g, r = cv2.split(band_aid)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        band_aid = cv2.merge((b, g, r, alpha))

    # Process the image
    print("🔍 Trying to detect hand in image...")
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        print("⚠️ No hand detected in image - कृपया इन बातों को ध्यान में रखें:")
        print("   1. हाथ की फोटो स्पष्ट होनी चाहिए")
        print("   2. हाथ की हथेली या पीछे का हिस्सा सीधा कैमरे की तरफ होना चाहिए")
        print("   3. फोटो में पर्याप्त रोशनी होनी चाहिए")
        print("   4. हाथ फोटो में बड़ा और स्पष्ट दिखना चाहिए")
    else:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get wrist and middle finger base landmarks
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            h, w, _ = image.shape
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            middle_x, middle_y = int(middle_base.x * w), int(middle_base.y * h)

            # Calculate band-aid placement
            center_x = (wrist_x + middle_x) // 2
            center_y = (wrist_y + middle_y) // 2

            # Scale band-aid based on hand size
            band_aid_width = int(max(abs(middle_x - wrist_x) * 1.2, 1))
            band_aid_height = int(max(band_aid_width * 0.4, 1))

            if band_aid_width <= 0 or band_aid_height <= 0:
                print("⚠️ Invalid band-aid size computed")
                continue

            # Resize band-aid
            band_aid_resized = cv2.resize(band_aid, (band_aid_width, band_aid_height))

            # Calculate overlay region
            y1 = max(center_y - band_aid_height // 2, 0)
            y2 = min(y1 + band_aid_height, image.shape[0])
            x1 = max(center_x - band_aid_width // 2, 0)
            x2 = min(x1 + band_aid_width, image.shape[1])

            # Crop band-aid if needed
            band_aid_resized = band_aid_resized[:max(0, y2 - y1), :max(0, x2 - x1)]

            if band_aid_resized.size == 0:
                print("⚠️ Band-aid region invalid after cropping")
                continue

            # Apply the band-aid overlay with alpha blending
            b, g, r, a = cv2.split(band_aid_resized)
            overlay_color = cv2.merge((b, g, r)).astype(float)
            mask = cv2.merge((a, a, a)).astype(float) / 255.0

            roi = image[y1:y2, x1:x2].astype(float)
            blended = (1 - mask) * roi + mask * overlay_color
            image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

            # Draw landmarks for visualization
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Save result if path provided
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"✅ Saved result to: {output_path}")

    # Show the result
    if show_result:
        # Create side-by-side comparison
        h1, w1 = original.shape[:2]
        h2, w2 = image.shape[:2]
        
        # Size them equally
        height = max(h1, h2)
        width = max(w1, w2)
        
        # Resize if needed
        if (h1, w1) != (height, width):
            original = cv2.resize(original, (width, height))
        if (h2, w2) != (height, width):
            image = cv2.resize(image, (width, height))
            
        # Stack side by side
        comparison = np.hstack((original, image))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(comparison, 'With Band-Aid', (width + 10, 30), font, 1, (0, 255, 0), 2)
        
        cv2.imshow('Before and After', comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    try:
        hands.close()
    except:
        pass

    return original, image

def main():
    # Test with all PNG/JPG files in the image folder
    image_dir = 'image'
    band_aid_path = os.path.join(image_dir, 'band_aid.png')
    
    # Find all test images (excluding band_aid.png)
    test_images = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        pattern = os.path.join(image_dir, ext)
        test_images.extend([f for f in glob.glob(pattern) 
                          if 'band_aid' not in f and 'output' not in f])
    
    if not test_images:
        print("❌ No test images found in 'image' folder")
        sys.exit(1)
        
    print(f"Found {len(test_images)} test image(s)")
    
    # Process each test image
    for img_path in test_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = f'output_{base_name}_with_bandaid.png'
        
        result = apply_band_aid(img_path, band_aid_path, output_path)
        if result:
            print(f"✅ Processed {base_name}")
        else:
            print(f"❌ Failed to process {base_name}")

if __name__ == '__main__':
    main()