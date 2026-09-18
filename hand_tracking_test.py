import cv2
import math 
import mediapipe as mp
from gesture_detector import get_drag_displacement_index, get_drag_displacement_middle, is_pinching, is_pinching_middle

mp_hands = mp.solutions.hands # type: ignore
mp_draw = mp.solutions.drawing_utils # type: ignore

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions.")

print("Press 'q' to quit.")

anchor_y = None  # Initialize previous midpoint

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
          mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

          if is_pinching(hand_landmarks.landmark):
              displacement, anchor_y = get_drag_displacement_index(hand_landmarks.landmark, anchor_y)
              print(f"index displacement={displacement:.4f}")
          elif is_pinching_middle(hand_landmarks.landmark):
              displacement, anchor_y = get_drag_displacement_middle(hand_landmarks.landmark, anchor_y)
              print(f"middle displacement={displacement:.4f}")
          else:
              anchor_y = None
              print("no pinch")

    cv2.imshow("Hand Tracking Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()