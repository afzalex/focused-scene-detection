from dataclasses import dataclass, field
import numpy as np
import cv2

from ultralytics.engine.results import Results

# Forward declaration of AppScene
@dataclass
class AppScene:
    pass

@dataclass
class AppFrame:
    frame: list
    detect: Results
    faces: list = field(default_factory=list)
    app_scene: AppScene = field(default=None)
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

    _frames: list[AppFrame] = field(default_factory=list)
    @property
    def frames(self):
        return self._frames
    @frames.setter
    def frames(self, frames: list[AppFrame]):
        self._frames.clear()
        for frame in frames:
            self.add_frame(frame)
    def add_frame(self, frame: AppFrame):
        self._frames.append(frame)
        frame.app_scene = self
    

@dataclass
class AppVideoMeta:
    width: int
    height: int
    frame_count: int