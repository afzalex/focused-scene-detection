from abc import ABC, abstractmethod
from functools import wraps
import sys
import io
import cv2

from .utils import capture_output

class AppFacer(ABC):

    @abstractmethod
    def init(this):
        pass
    
    @abstractmethod
    def detect_faces(this, image):
        pass

    @abstractmethod
    def get_boundary(this, face):
        pass

class OpenCVAppFacer(AppFacer):
    def init(this):
        this.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(this, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = this.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

    def get_boundary(this, face):
        return face


from mtcnn import MTCNN
class MTCNNAppFacer(AppFacer):
    def init(this):
        this.detector = MTCNN()
        this.detect_faces = capture_output(this.detect_faces)
    
    def detect_faces(this, image):
        faces = this.detector.detect_faces(image)
        return faces

    def get_boundary(this, face):
        return face['box']


from deepface import DeepFace
class DeepFaceAppFacer(AppFacer):
    def init(this):
        pass
    
    def detect_faces(this, image):
        face_analysis = DeepFace.analyze(image, ['gender'], enforce_detection=False, silent=True)
        return face_analysis

    def get_boundary(this, face):
        region = face['region']
        return region['x'], region['y'], region['w'], region['h']

opencv = OpenCVAppFacer()
mtcnn = MTCNNAppFacer()
deepface = DeepFaceAppFacer()
