import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
threshold = 10

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = img.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark[8]  # Index finger tip
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (x, y), 8, (255, 0, 0), cv2.FILLED)  # Main violet dot

            dx = x - prev_x
            dy = y - prev_y

            direction = ""
            if abs(dx) > threshold:
                direction = "Right" if dx > 0 else "Left"
            elif abs(dy) > threshold:
                direction = "Down" if dy > 0 else "Up"

            if direction:
                print("Direction:", direction)
                cv2.putText(img, direction, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            prev_x, prev_y = x, y

    cv2.imshow("Day 2 - Hand Gesture Directions", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
