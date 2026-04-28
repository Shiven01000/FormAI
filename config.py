# Data-driven exercise registry. Adding a new exercise = one new dict entry;
# no changes required in rep_counter.py, form_checker.py, or main.py.
#
# MediaPipe landmark indices used here:
#   11 LEFT_SHOULDER   12 RIGHT_SHOULDER
#   13 LEFT_ELBOW      14 RIGHT_ELBOW
#   15 LEFT_WRIST      16 RIGHT_WRIST
#   23 LEFT_HIP        24 RIGHT_HIP
#   25 LEFT_KNEE       26 RIGHT_KNEE
#   27 LEFT_ANKLE      28 RIGHT_ANKLE
#
# Full map: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

EXERCISES = {

    "bicep_curl": {
        "display_name":    "Bicep Curl",
        "description":     "Curl arm up, squeeze the bicep",
        # Ascending: arm starts curled (low angle), rep counts on extension.
        # Counting on the return to straight is more reliable than requiring
        # a tight curl — avoids missed reps due to hyperextension checks.
        "landmarks":       [11, 13, 15],
        "landmarks_right": [12, 14, 16],
        "bilateral":       True,
        "down_angle":      50,
        "up_angle":        150,
        "joint_label":     "Elbow",
    },

    "squat": {
        "display_name": "Squat",
        "description":  "Lower hips until knees reach 90°",
        "landmarks":    [23, 25, 27],
        "down_angle":   90,
        "up_angle":     160,
        "joint_label":  "Knee",
    },

    "pushup": {
        "display_name": "Pushup",
        "description":  "Lower chest to floor, arms bent",
        "landmarks":    [11, 13, 15],
        "down_angle":   80,
        "up_angle":     160,
        "joint_label":  "Elbow",
    },

    "shoulder_press": {
        "display_name":    "Shoulder Press",
        "description":     "Press arms overhead from shoulder height",
        "landmarks":       [11, 13, 15],
        "landmarks_right": [12, 14, 16],
        "bilateral":       True,
        "down_angle":      70,
        "up_angle":        160,
        "joint_label":     "Elbow",
    },

    "lunge": {
        "display_name": "Lunge",
        "description":  "Step forward, lower back knee toward floor",
        "landmarks":    [23, 25, 27],
        "down_angle":   90,
        "up_angle":     160,
        "joint_label":  "Knee",
    },

    "lateral_raise": {
        "display_name":    "Lateral Raise",
        "description":     "Raise arms to shoulder level, no higher",
        # Landmarks ordered hip → shoulder → elbow so the angle opens
        # from ~10° (arm at side) to ~80° (arm parallel to floor).
        "landmarks":       [23, 11, 13],
        "landmarks_right": [24, 12, 14],
        "bilateral":       True,
        "down_angle":      20,
        "up_angle":        70,
        "joint_label":     "Shoulder",
    },
}

# Ordered keyboard mapping: key "1" → index 0, "2" → index 1, …
EXERCISE_KEYS = [
    "bicep_curl",
    "squat",
    "pushup",
    "shoulder_press",
    "lunge",
    "lateral_raise",
]
