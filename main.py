import cv2
import mediapipe as mp
import time

from gesture_detector import is_pinching, is_pinching_middle, get_drag_displacement_index, get_drag_displacement_middle
from actions import get_volume, set_volume, get_brightness, set_brightness

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

current_volume = get_volume()
current_brightness = get_brightness()

anchor_y = None
last_action_time = 0

COOLDOWN = 0.15          # seconds between nudges, tune by feel
DEAD_ZONE = 0.02         # ignore tiny drift so a held-still pinch doesn't act on its own
STEP_VOLUME = 5          # how much one nudge changes volume
STEP_BRIGHTNESS = 0.05

print("Press 'q' to quit.")

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
            landmarks = hand_landmarks.landmark

            pinching_volume = is_pinching(landmarks)
            # index takes priority if both would fire; adjust this line if you already
            # solved that conflict differently in gesture_detector.py
            pinching_brightness = is_pinching_middle(landmarks) and not pinching_volume

            if pinching_volume or pinching_brightness:
                if pinching_volume:
                    displacement, anchor_y = get_drag_displacement_index(landmarks, anchor_y)
                else:
                    displacement, anchor_y = get_drag_displacement_middle(landmarks, anchor_y)
                ready = (time.time() - last_action_time) > COOLDOWN

                if abs(displacement) > DEAD_ZONE and ready:
                    moved_up = displacement < 0   # y increases downward, so "up" is negative

                    if pinching_volume:
                        current_volume += STEP_VOLUME if moved_up else -STEP_VOLUME
                        current_volume = max(0, min(100, current_volume))
                        set_volume(current_volume)
                        print(f"volume -> {current_volume}")
                    else:
                        current_brightness += STEP_BRIGHTNESS if moved_up else -STEP_BRIGHTNESS
                        current_brightness = max(0.0, min(1.0, current_brightness))
                        set_brightness(current_brightness)
                        print(f"brightness -> {current_brightness:.2f}")

                    last_action_time = time.time()
            else:
                anchor_y = None
    else:
        anchor_y = None

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()