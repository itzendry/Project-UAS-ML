import math

LANDMARK_COUNT = 21
AXES = ("x", "y", "z")


def feature_columns():
    columns = []
    for index in range(LANDMARK_COUNT):
        for axis in AXES:
            columns.append(f"{axis}{index}")
    return columns


def landmarks_to_features(hand_landmarks, normalize=True):
    points = [
        (
            hand_landmarks.landmark[index].x,
            hand_landmarks.landmark[index].y,
            hand_landmarks.landmark[index].z,
        )
        for index in range(LANDMARK_COUNT)
    ]

    if not normalize:
        return [value for point in points for value in point]

    wrist_x, wrist_y, wrist_z = points[0]
    middle_mcp_x, middle_mcp_y, middle_mcp_z = points[9]
    scale = math.dist((wrist_x, wrist_y, wrist_z), (middle_mcp_x, middle_mcp_y, middle_mcp_z))
    if scale == 0:
        scale = 1.0

    normalized = []
    for x, y, z in points:
        normalized.extend(
            [
                (x - wrist_x) / scale,
                (y - wrist_y) / scale,
                (z - wrist_z) / scale,
            ]
        )
    return normalized


def fingers_open(hand_landmarks):
    return {
        "index": hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,
        "middle": hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,
        "ring": hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,
        "pinky": hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y,
    }


def heuristic_right_gesture(hand_landmarks, frame_width=1280, frame_height=720):
    hx4 = int(hand_landmarks.landmark[4].x * frame_width)
    hy4 = int(hand_landmarks.landmark[4].y * frame_height)
    hx8 = int(hand_landmarks.landmark[8].x * frame_width)
    hy8 = int(hand_landmarks.landmark[8].y * frame_height)
    pinch_dist = math.hypot(hx8 - hx4, hy8 - hy4)
    open_state = fingers_open(hand_landmarks)

    if pinch_dist < 35:
        return "right_click"
    if open_state["index"] and open_state["middle"] and open_state["ring"] and not open_state["pinky"]:
        return "right_arrow_right"
    if open_state["index"] and open_state["middle"] and not open_state["ring"] and not open_state["pinky"]:
        return "right_arrow_left"
    if open_state["index"] and not open_state["middle"] and not open_state["ring"] and not open_state["pinky"]:
        return "right_cursor"
    return "idle"
