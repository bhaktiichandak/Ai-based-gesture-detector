import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Debug print to check if webcam is working
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()
else:
    print("✅ Webcam opened successfully")

# Variables to track finger movement
prev_x, prev_y = 0, 0
threshold = 40
action_cooldown = 20  # Wait frames before next action
cooldown_counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to grab frame")
        break

    img = cv2.flip(img, 1)  # Flip image like mirror
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    h, w, _ = img.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark[8]  # Index fingertip
            x, y = int(lm.x * w), int(lm.y * h)

            # Draw violet circle
            cv2.circle(img, (x, y), 20, (255, 0, 255), cv2.FILLED)

            # Check movement direction
            dx = x - prev_x
            dy = y - prev_y

            direction = ""

            if cooldown_counter == 0:
                if abs(dx) > threshold:
                    direction = "Right" if dx > 0 else "Left"
                    pyautogui.press('right' if dx > 0 else 'left')
                    cooldown_counter = action_cooldown

                elif abs(dy) > threshold:
                    direction = "Up" if dy < 0 else "Down"
                    if dy < 0:
                        pyautogui.press('space')  # jump
                    cooldown_counter = action_cooldown

                if direction:
                    print("👉 Action:", direction)
                    cv2.putText(img, direction, (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)

            prev_x, prev_y = x, y

    if cooldown_counter > 0:
        cooldown_counter -= 1

    cv2.imshow("🎮 Day 3 - Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
