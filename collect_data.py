import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands # type: ignore
mp_draw = mp.solutions.drawing_utils # type: ignore
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

LABEL = "pinch_middle"                # change this, then rerun, for each class
OUTPUT_FILE = "gesture_data.csv"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

recording = False
rows_written = 0
print(f"Collecting label: {LABEL}. Press 'r' to start/stop recording, 'q' to quit.")

# "a" = append, so running this once per label builds up one shared file
with open(OUTPUT_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if recording:
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])   # 21 landmarks x 3 = 63 numbers
                row.append(LABEL)                     # the 64th column
                writer.writerow(row)
                rows_written += 1

        status = f"REC {rows_written}" if recording else "paused"
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(frame, f"{LABEL} - {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = not recording
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Done. Wrote {rows_written} rows for label '{LABEL}'.")