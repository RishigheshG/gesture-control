import math

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