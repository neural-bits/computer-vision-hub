from dataclasses import dataclass

import numpy as np


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_int(self):
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def to_xywh(self):
        new_format = [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]
        new_format_asint = [int(v) for v in new_format]
        return new_format_asint

    def normalize_extend(self, percent):
        bbox_w = self.x2 - self.x1
        bbox_h = self.y2 - self.y1

        centroid_x = (self.x1 + self.x2) / 2
        centroid_y = (self.y1 + self.y2) / 2

        increase_w = bbox_w * (percent / 100)
        increase_h = bbox_h * (percent / 100)

        new_width = bbox_w + increase_w
        new_height = bbox_h + increase_h

        new_bbox = [
            centroid_x - (new_width / 2),
            centroid_y - (new_height / 2),
            centroid_x + (new_width / 2),
            centroid_y + (new_height / 2),
        ]

        extended_bbox = [
            int(max(new_bbox[0], 0)),
            int(max(new_bbox[1], 0)),
            int(max(new_bbox[2], 0)),
            int(max(new_bbox[3], 0)),
        ]

        return extended_bbox


@dataclass
class PersonDetection:
    frame_id: int
    track_id: int
    bbox: BoundingBox
    last_action: str = None
    crop: np.array = None

    @property
    def id(self):
        return self.track_id

    def set_last_action(self, action):
        self.last_action = action

    def crop_person(self, frame: np.array, scale):
        box = self.bbox.normalize_extend(scale)
        x1, y1, x2, y2 = self.bbox.as_int()
        crop = frame[y1:y2, x1:x2]
        self.crop = crop
