"""Pipeline de audio: transcricao, features acusticas, emocao e analise textual."""

from src.audio.azure_language import AzureLanguageClient
from src.audio.emotion import VocalEmotionClassifier
from src.audio.features import extract_features
from src.audio.pipeline import AudioPipeline
from src.audio.transcriber import (
    AzureSpeechTranscriber,
    TranscriberProtocol,
    WhisperTranscriber,
    get_transcriber,
)
from src.audio.types import (
    AcousticFeatures,
    AudioAnalysis,
    EmotionScore,
    Segment,
    SentimentResult,
)

__all__ = [
    "AcousticFeatures",
    "AudioAnalysis",
    "AudioPipeline",
    "AzureLanguageClient",
    "AzureSpeechTranscriber",
    "EmotionScore",
    "Segment",
    "SentimentResult",
    "TranscriberProtocol",
    "VocalEmotionClassifier",
    "WhisperTranscriber",
    "extract_features",
    "get_transcriber",
]
