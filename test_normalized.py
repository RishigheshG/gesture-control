import cv2
import mediapipe as mp
from gesture_detector import normalize_landmarks

mp_hands = mp.solutions.hands # type: ignore
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        points = [(lm.x, lm.y, lm.z) for lm in results.multi_hand_landmarks[0].landmark]
        norm = normalize_landmarks(points)
        rx, ry, _ = points[8]
        nx, ny, _ = norm[8]
        print(f"index tip   raw x={rx:.3f} y={ry:.3f}   |   normalized x={nx:.3f} y={ny:.3f}   |   wrist={tuple(round(v, 3) for v in norm[0])}")

    cv2.imshow("Normalize test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()