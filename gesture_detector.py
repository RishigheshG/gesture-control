import math
from collections import deque, Counter

def normalize_landmarks(points):
    """
    points: list of 21 (x, y, z) tuples, in MediaPipe's landmark order.
    Returns 21 (x, y, z) tuples, relative to the wrist and scaled by hand size.
    Works on live camera data AND on rows loaded from the CSV.
    """
    wrist = points[0]

    # 1. translate: wrist becomes (0, 0, 0), everything else is relative to it
    relative = [(x - wrist[0], y - wrist[1], z - wrist[2]) for x, y, z in points]

    # 2. scale: hand size = distance from wrist to middle-finger knuckle (landmark 9)
    mcp_x, mcp_y, _ = relative[9]
    hand_size = math.sqrt(mcp_x**2 + mcp_y**2)     # same formula shape as is_pinching()
    if hand_size < 1e-6:
        hand_size = 1e-6                            # never divide by zero on a weird frame

    return [(x / hand_size, y / hand_size, z / hand_size) for x, y, z in relative]

class GestureSmoother:
    """Majority vote over the last window_size predictions."""
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)   # automatically drops the oldest when full

    def update(self, prediction):
        self.history.append(prediction)
        return Counter(self.history).most_common(1)[0][0]   # the label with the most votes

def is_pinching(landmarks, threshold=0.07):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)  # your formula from above
    return distance < threshold   # comparison against threshold

def is_pinching_middle(landmarks, threshold=0.07):
    thumb_tip = landmarks[4]
    middle_tip = landmarks[12]
    distance = math.sqrt((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)
    return distance < threshold

def get_drag_displacement_index(landmarks, anchor_y):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    midpoint_y = (thumb_tip.y + index_tip.y) / 2

    if anchor_y is None:
        anchor_y = midpoint_y             # lock in this frame's midpoint_y as the reference
        displacement = 0
    else:
        displacement = midpoint_y - anchor_y        # how far from the anchor, not from a previous midpoint

    return displacement, anchor_y

def get_drag_displacement_middle(landmarks, anchor_y):
    thumb_tip = landmarks[4]
    middle_tip = landmarks[12]
    midpoint_y = (thumb_tip.y + middle_tip.y) / 2

    if anchor_y is None:
        anchor_y = midpoint_y             # lock in this frame's midpoint_y as the reference
        displacement = 0
    else:
        displacement = midpoint_y - anchor_y        # how far from the anchor, not from a previous midpoint

    return displacement, anchor_y