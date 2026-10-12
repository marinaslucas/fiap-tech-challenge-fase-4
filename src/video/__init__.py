"""Pipeline de video: deteccao YOLO e emocao facial multimodal."""

from src.video.azure_openai_vision import AzureOpenAIVisionEmotion
from src.video.detector import BleedingDetector, ensure_yolo_weights
from src.video.emotion import (
    FacialEmotionClassifierProtocol,
    FacialEmotionDetector,
    get_facial_emotion_classifier,
)
from src.video.pipeline import VideoPipeline
from src.video.scene_classifier import SceneType, classify_scene_type
from src.video.types import (
    BoundingBox,
    Detection,
    EmotionScore,
    VideoEvent,
)

__all__ = [
    "AzureOpenAIVisionEmotion",
    "BleedingDetector",
    "BoundingBox",
    "Detection",
    "EmotionScore",
    "FacialEmotionClassifierProtocol",
    "FacialEmotionDetector",
    "SceneType",
    "VideoEvent",
    "VideoPipeline",
    "classify_scene_type",
    "ensure_yolo_weights",
    "get_facial_emotion_classifier",
]
