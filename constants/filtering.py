# NOTE First element of tuple is COCO-filter / second is COCOWholebody-filter
KEYPOINT_FILTERS = {
    # 1. [face (hand allowed)]
    1: (
        "nose/left_eye/right_eye/left_ear/right_ear | left_knee/right_knee/left_ankle/right_ankle/left_hip_extra/right_hip_extra",
        "|",
    ),
    # 2. [face (strict)]
    2: (
        "nose/left_eye/right_eye/left_ear/right_ear | left_knee/right_knee/left_ankle/right_ankle/left_hip_extra/right_hip_extra/left_wrist/right_wrist/left_elbow/right_elbow",
        "|",
    ),
    # 3. [full body (one of shoulder, one of hip)]
    3: (
        "left_shoulder,right_shoulder / left_hip_extra,right_hip_extra | ",
        "|",
    ),
    # 4. [full body (one of shoulder, one of hip, one of elbow or hand)]
    4: (
        "left_shoulder,right_shoulder / left_hip_extra,right_hip_extra / left_elbow, right_elbow | ",
        "|",
    ),
    # 5. [full body (one of shoulder, one of hip, one of knees or ankle)]
    5: (
        "left_shoulder,right_shoulder / left_hip_extra,right_hip_extra / left_knee, right_knee, left_ankle, right_ankle | ",
        "|",
    ),
    # 6. [hand (elbow, shoulder allowed: both sides available)]
    6: (
        "left_wrist, right_wrist | left_knee/right_knee/left_ankle/right_ankle/left_hip_extra/right_hip_extra/nose/left_eye/right_eye/left_ear/right_ear",
        "|",
    ),
    # 7. [hand (elbow allowed: both sides available)]
    7: (
        "left_wrist, right_wrist | left_knee/right_knee/left_ankle/right_ankle/left_hip_extra/right_hip_extra/nose/left_eye/right_eye/left_ear/right_ear/left_shoulder/right_shoulder",
        "|",
    ),
    # 8. [hand (strict: both sides available)]
    8: (
        "left_wrist, right_wrist | left_knee/right_knee/left_ankle/right_ankle/left_hip_extra/right_hip_extra/nose/left_eye/right_eye/left_ear/right_ear/left_shoulder/right_shoulder/left_elbow/right_elbow",
        "|",
    ),
    # 9. [hand (elbow allowed: only left)]
    9: (
        "left_wrist | left_knee/right_knee/left_ankle/right_ankle/left_hip_extra/right_hip_extra/nose/left_eye/right_eye/left_ear/right_ear/left_shoulder/right_shoulder/right_wrist/right_elbow",
        "|",
    ),
    # 10. [hand (elbow allowed: only right)]
    10: (
        "right_wrist | left_knee/right_knee/left_ankle/right_ankle/left_hip_extra/right_hip_extra/nose/left_eye/right_eye/left_ear/right_ear/left_shoulder/right_shoulder/left_wrist/left_elbow",
        "|",
    ),
    # 11. [full body (one of shoulder, one of hip, one of knees, one of ankle)]
    11: (
        "left_shoulder,right_shoulder / left_hip_extra,right_hip_extra / left_knee, right_knee / left_ankle, right_ankle | ",
        "|",
    ),
    # 12. [face (hand allowed, side view allowed)]
    12: (
        "left_eye,right_eye/left_ear,right_ear,nose | left_knee/right_knee/left_ankle/right_ankle/left_hip_extra/right_hip_extra",
        "|",
    ),
    # 13. [face (strict, side view allowed)]
    13: (
        "left_eye,right_eye/left_ear,right_ear,nose | left_knee/right_knee/left_ankle/right_ankle/left_hip_extra/right_hip_extra/left_wrist/right_wrist/left_elbow/right_elbow",
        "|",
    ),
    # 14. [face+body]
    14: ("nose/left_eye,right_eye/left_ear,right_ear /left_shoulder,right_shoulder | ", "|"),
    # 15. [no keypoint filter]
    15: ("|", "|"),
}

DEFAULT_KEYPOINT_FILTER_NUM = 3
DEFAULT_FILTER_SETTING_NUM = 3

DO_SHUFFLE_WHEN_NO_KPFILTER = True  # For the "No-Keypoint Filtering" Experiment
