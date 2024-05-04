from dataclasses import dataclass, field
import numpy as np
import cv2

@dataclass
class AppFrame:
    frame: list
    detect: list = field(default=None)
    track: list = field(default=None)
    faces: list = field(default=None)
    filtered_faces: list = field(default=None)
    _frame_gray: list = field(default=None)
    _frame_std_dev: int = field(default=None)
    
    @property
    def frame_gray(self):
        return self._frame_gray
        
    @property
    def frame_std_dev(self):
        return self._frame_std_dev
        
    def __post_init__(self):
        self._frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self._frame_std_dev = np.std(self._frame_gray)

@dataclass
class AppScene:
    scene_orig: list = field(default=None)
    frame_start: int = field(default=None)
    frame_end: int = field(default=None)
    frames: list[AppFrame] = field(default_factory=list)
    detects: list = field(default_factory=list)


@dataclass
class AppVideoMeta:
    width: int
    height: int
    frame_count: int