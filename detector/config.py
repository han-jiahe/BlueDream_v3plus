"""
BlueDream v3plus — Soccer Pitch Configuration
Standard FIFA 105×68m pitch. Supports both:
  - 32 keypoints (legacy roboflow/sports format)
  - 29 keypoints (Soccana_Keypoint, YOLOv11n-pose format)
Self-contained — no roboflow/sports dependency.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# ── Pitch dimensions in cm (shared) ────────────────────
PITCH_WIDTH_CM = 6800    # 68m
PITCH_LENGTH_CM = 10500  # 105m
PENALTY_BOX_WIDTH = 4032
PENALTY_BOX_LENGTH = 1650
GOAL_BOX_WIDTH = 1832
GOAL_BOX_LENGTH = 550
CENTRE_CIRCLE_RADIUS = 915
PENALTY_SPOT_DISTANCE = 1100

# Shorthand variables for formula readability
_W = PITCH_WIDTH_CM
_L = PITCH_LENGTH_CM
_PBW = PENALTY_BOX_WIDTH
_PBL = PENALTY_BOX_LENGTH
_GBW = GOAL_BOX_WIDTH
_GBL = GOAL_BOX_LENGTH
_CCR = CENTRE_CIRCLE_RADIUS
_PSD = PENALTY_SPOT_DISTANCE


@dataclass
class SoccerPitchConfiguration:
    """Standard soccer pitch (105×68m) with 32 keypoint vertices (legacy format)."""

    width: int = 6800       # [cm] = 68m
    length: int = 10500     # [cm] = 105m
    penalty_box_width: int = 4032
    penalty_box_length: int = 1650
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        """32 canonical pitch keypoints in cm."""
        return [
            (0, 0),                                                              # 1
            (0, (self.width - self.penalty_box_width) / 2),                      # 2
            (0, (self.width - self.goal_box_width) / 2),                         # 3
            (0, (self.width + self.goal_box_width) / 2),                         # 4
            (0, (self.width + self.penalty_box_width) / 2),                      # 5
            (0, self.width),                                                     # 6
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),      # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),      # 8
            (self.penalty_spot_distance, self.width / 2),                        # 9
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),# 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),   # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),   # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),# 13
            (self.length / 2, 0),                                                # 14
            (self.length / 2, self.width / 2 - self.centre_circle_radius),       # 15
            (self.length / 2, self.width / 2 + self.centre_circle_radius),       # 16
            (self.length / 2, self.width),                                       # 17
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 18
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),     # 19
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),     # 20
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 21
            (self.length - self.penalty_spot_distance, self.width / 2),         # 22
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),        # 23
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),        # 24
            (self.length, 0),                                                    # 25
            (self.length, (self.width - self.penalty_box_width) / 2),           # 26
            (self.length, (self.width - self.goal_box_width) / 2),              # 27
            (self.length, (self.width + self.goal_box_width) / 2),              # 28
            (self.length, (self.width + self.penalty_box_width) / 2),           # 29
            (self.length, self.width),                                           # 30
            (self.length / 2 - self.centre_circle_radius, self.width / 2),      # 31
            (self.length / 2 + self.centre_circle_radius, self.width / 2),      # 32
        ]

    # Edge pairs (1-indexed) for drawing pitch lines
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
        "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
        "14", "19",
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#00BFFF", "#00BFFF",
    ])


# ── Soccana Keypoint Configuration (29 keypoints) ───────
# NOTE: Exact keypoint order is determined by the YOLOv11n-pose model.
# If homography is inaccurate, verify against one frame's predictions
# and adjust vertex order accordingly.

@dataclass
class SoccanaKeypointConfig:
    """
    29 keypoint target coordinates for Soccana_Keypoint model.

    Keypoint groups (based on model README):
      0-3:   Field boundaries (4 corners)
      4-11:  Penalty areas / Big boxes (8 points)
      12-19: Goal areas / Small boxes (8 points)
      20-28: Center elements (9 points)

    All coordinates in centimeters. Pitch: 10500 × 6800 cm.
    """

    width: int = _W
    length: int = _L

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        """29 target pitch keypoints in cm (Soccana_Keypoint order)."""
        # NOTE: This is an APPROXIMATE mapping. The exact order depends on
        # the model's training data. Verify with one test frame and adjust.
        return [
            # ── Group 1: Field Boundaries (4) ──
            (0, 0),                                       #  0: top-left corner
            (_L, 0),                                      #  1: top-right corner
            (_L, _W),                                     #  2: bottom-right corner
            (0, _W),                                      #  3: bottom-left corner

            # ── Group 2: Penalty Areas — Left (4) ──
            (_PBL, (_W - _PBW) / 2),                      #  4: left penalty top
            (_PBL, (_W - _GBW) / 2),                      #  5: left penalty inner-top
            (_PBL, (_W + _GBW) / 2),                      #  6: left penalty inner-bottom
            (_PBL, (_W + _PBW) / 2),                      #  7: left penalty bottom

            # ── Group 2: Penalty Areas — Right (4) ──
            (_L - _PBL, (_W - _PBW) / 2),                 #  8: right penalty top
            (_L - _PBL, (_W - _GBW) / 2),                 #  9: right penalty inner-top
            (_L - _PBL, (_W + _GBW) / 2),                 # 10: right penalty inner-bottom
            (_L - _PBL, (_W + _PBW) / 2),                 # 11: right penalty bottom

            # ── Group 3: Goal Areas — Left (4) ──
            (_GBL, (_W - _GBW) / 2),                      # 12: left goal top
            (_GBL, (_W + _GBW) / 2),                      # 13: left goal bottom
            (0, (_W - _GBW) / 2),                         # 14: left goal-line top
            (0, (_W + _GBW) / 2),                         # 15: left goal-line bottom

            # ── Group 3: Goal Areas — Right (4) ──
            (_L - _GBL, (_W - _GBW) / 2),                 # 16: right goal top
            (_L - _GBL, (_W + _GBW) / 2),                 # 17: right goal bottom
            (_L, (_W - _GBW) / 2),                        # 18: right goal-line top
            (_L, (_W + _GBW) / 2),                        # 19: right goal-line bottom

            # ── Group 4: Center Elements (9) ──
            (_L / 2, 0),                                  # 20: center line top
            (_L / 2, _W),                                 # 21: center line bottom
            (_L / 2, _W / 2 - _CCR),                      # 22: center circle top
            (_L / 2, _W / 2 + _CCR),                      # 23: center circle bottom
            (_L / 2 - _CCR, _W / 2),                      # 24: center circle left
            (_L / 2 + _CCR, _W / 2),                      # 25: center circle right
            (_L / 2, _W / 2),                             # 26: field center
            (_PSD, _W / 2),                               # 27: left penalty spot
            (_L - _PSD, _W / 2),                          # 28: right penalty spot
        ]
