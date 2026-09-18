import cv2
import mediapipe as mp
import time
import numpy as np
import joblib

from gesture_detector import normalize_landmarks, get_drag_displacement_index, get_drag_displacement_middle, GestureSmoother
from actions import get_volume, set_volume, get_brightness, set_brightness

mp_hands = mp.solutions.hands # type: ignore
mp_draw = mp.solutions.drawing_utils # type: ignore
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# NEW: the trained model and the smoother replace is_pinching / is_pinching_middle
clf = joblib.load("gesture_classifier.pkl")
smoother = GestureSmoother(window_size=5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions.")

current_volume = get_volume()
current_brightness = get_brightness()

anchor_y = None
active_gesture = "idle"       # NEW: remember last frame's gesture so switching pinches resets the drag
last_action_time = 0

COOLDOWN = 0.15
DEAD_ZONE = 0.02
STEP_VOLUME = 5
STEP_BRIGHTNESS = 0.05

print("Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture = "idle"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        landmarks = hand_landmarks.landmark

        # NEW: same normalization as training, then classify, then smooth
        points = [(lm.x, lm.y, lm.z) for lm in landmarks]
        features = np.array(normalize_landmarks(points)).flatten().reshape(1, -1)
        raw_prediction = clf.predict(features)[0]
        gesture = smoother.update(raw_prediction)

        if gesture in ("pinch_index", "pinch_middle"):
            if gesture != active_gesture:
                anchor_y = None                 # NEW: switching from one pinch to the other starts a fresh drag

            if gesture == "pinch_index":
                displacement, anchor_y = get_drag_displacement_index(landmarks, anchor_y)
            else:
                displacement, anchor_y = get_drag_displacement_middle(landmarks, anchor_y)
            ready = (time.time() - last_action_time) > COOLDOWN

            if abs(displacement) > DEAD_ZONE and ready:
                moved_up = displacement < 0

                if gesture == "pinch_index":
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
        gesture = smoother.update("idle")   # NEW: no hand counts as an idle vote, so a stale pinch decays out of the window

    active_gesture = gesture

    # NEW: show the smoothed gesture on screen so you can see what the model thinks in real time
    cv2.putText(frame, f"gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()