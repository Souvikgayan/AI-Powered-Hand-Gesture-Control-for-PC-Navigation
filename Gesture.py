import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror image
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            lm = hand_landmark.landmark

            # Get index fingertip (landmark 8) and thumb tip (landmark 4)
            index_finger = lm[8]
            thumb = lm[4]

            index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # Move mouse
            screen_x = np.interp(index_finger.x, [0, 1], [0, screen_width])
            screen_y = np.interp(index_finger.y, [0, 1], [0, screen_height])
            pyautogui.moveTo(screen_x, screen_y)

            # If index and thumb are close → Click
            if distance((index_x, index_y), (int(thumb.x * w), int(thumb.y * h))) < 40:
                pyautogui.click()
                cv2.putText(img, "Click!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Hand Gesture Mouse Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
