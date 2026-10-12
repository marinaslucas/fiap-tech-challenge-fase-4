"""Testes smoke do pipeline de audio.

Mocks isolam dependencias externas (`faster-whisper`, `transformers`,
`librosa`, Azure SDKs) para manter o teste rapido e offline.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from src.audio import (
    AcousticFeatures,
    AudioAnalysis,
    AudioPipeline,
    AzureLanguageClient,
    AzureSpeechTranscriber,
    EmotionScore,
    Segment,
    SentimentResult,
    VocalEmotionClassifier,
    WhisperTranscriber,
    extract_features,
    get_transcriber,
)

# ---------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_segment_aceita_timestamps_em_ms():
    seg = Segment(start_ms=0, end_ms=1500, text="ola")
    assert seg.end_ms - seg.start_ms == 1500


@pytest.mark.smoke
def test_acoustic_features_aceita_mfcc_de_13():
    feats = AcousticFeatures(
        duration_s=3.0,
        pitch_mean_hz=200.0,
        pitch_std_hz=10.0,
        energy_rms=0.3,
        zero_crossing_rate=0.1,
        jitter=0.01,
        shimmer=0.02,
        mfcc_means=[0.0] * 13,
    )
    assert len(feats.mfcc_means) == 13


@pytest.mark.smoke
def test_emotion_score_rejeita_confianca_invalida():
    with pytest.raises(ValidationError):
        EmotionScore(label="happy", confidence=1.5)


@pytest.mark.smoke
def test_audio_analysis_tem_defaults_sensatos():
    analise = AudioAnalysis()
    assert analise.transcription == ""
    assert analise.segments == []
    assert analise.acoustic_features is None
    assert analise.emotion is None
    assert analise.sentiment is None
    assert analise.key_phrases == []


# ---------------------------------------------------------------------
# Transcribers
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_whisper_transcriber_inicializa_lazy():
    transcriber = WhisperTranscriber(model_size="base", language="pt")
    assert transcriber.model_size == "base"
    assert transcriber._model is None
    assert transcriber._load_attempted is False


@pytest.mark.smoke
def test_whisper_transcriber_retorna_vazio_quando_indisponivel(monkeypatch, tmp_path):
    transcriber = WhisperTranscriber()
    monkeypatch.setattr(transcriber, "load", lambda: None)
    transcriber._load_attempted = True
    transcriber._available = False
    fake_audio = tmp_path / "fake.wav"
    fake_audio.touch()
    text, segments = transcriber.transcribe(fake_audio)
    assert text == ""
    assert segments == []


@pytest.mark.smoke
def test_azure_speech_transcriber_nao_configurado_retorna_vazio(tmp_path, monkeypatch):
    from pydantic import SecretStr

    from src.config.settings import settings as settings_obj

    monkeypatch.setattr(settings_obj, "azure_speech_key", SecretStr(""))
    monkeypatch.setattr(settings_obj, "azure_speech_region", "")
    transcriber = AzureSpeechTranscriber()
    assert transcriber.is_configured is False
    fake_audio = tmp_path / "fake.wav"
    fake_audio.touch()
    text, segments = transcriber.transcribe(fake_audio)
    assert text == ""
    assert segments == []


@pytest.mark.smoke
def test_get_transcriber_retorna_whisper_por_padrao(monkeypatch):
    monkeypatch.setattr("src.audio.transcriber.settings.use_cloud_transcription", False)
    transcriber = get_transcriber()
    assert isinstance(transcriber, WhisperTranscriber)


# ---------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_extract_features_retorna_acoustic_features(tmp_path):
    import soundfile as sf

    sr = 16_000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sinal = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    audio_path = tmp_path / "tone.wav"
    sf.write(str(audio_path), sinal, sr, subtype="PCM_16")

    feats = extract_features(audio_path, sample_rate=sr)
    assert isinstance(feats, AcousticFeatures)
    assert feats.duration_s == pytest.approx(duration, abs=0.05)
    assert feats.pitch_mean_hz > 0
    assert len(feats.mfcc_means) == 13


# ---------------------------------------------------------------------
# Emocao vocal
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_vocal_emotion_classifier_inicializa_lazy():
    classifier = VocalEmotionClassifier(model_name="superb/wav2vec2-base-superb-er")
    assert classifier._pipeline is None
    assert classifier._load_attempted is False


@pytest.mark.smoke
def test_vocal_emotion_classifier_fallback_quando_indisponivel(monkeypatch, tmp_path):
    classifier = VocalEmotionClassifier()
    monkeypatch.setattr(classifier, "load", lambda: None)
    classifier._load_attempted = True
    classifier._available = False
    fake = tmp_path / "fake.wav"
    fake.touch()
    assert classifier.classify(fake) is None


@pytest.mark.smoke
def test_vocal_emotion_classifier_parseia_pipeline_results(monkeypatch, tmp_path):
    classifier = VocalEmotionClassifier()
    fake_pipeline = MagicMock()
    fake_pipeline.return_value = [
        {"label": "happy", "score": 0.8},
        {"label": "neutral", "score": 0.15},
        {"label": "sad", "score": 0.05},
    ]
    classifier._pipeline = fake_pipeline
    classifier._available = True
    classifier._load_attempted = True
    fake = tmp_path / "fake.wav"
    fake.touch()
    score = classifier.classify(fake)
    assert score is not None
    assert score.label == "happy"
    assert score.confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------
# Azure Language
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_azure_language_nao_configurado_retorna_none_e_lista_vazia(monkeypatch):
    from pydantic import SecretStr

    from src.config.settings import settings as settings_obj

    monkeypatch.setattr(settings_obj, "azure_language_key", SecretStr(""))
    monkeypatch.setattr(settings_obj, "azure_language_endpoint", "")
    client = AzureLanguageClient()
    assert client.is_configured is False
    assert client.analyze_sentiment("texto qualquer") is None
    assert client.extract_key_phrases("texto qualquer") == []


@pytest.mark.smoke
def test_azure_language_ignora_texto_em_branco():
    client = AzureLanguageClient()
    assert client.analyze_sentiment("") is None
    assert client.extract_key_phrases("   ") == []


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_audio_pipeline_falha_quando_arquivo_nao_existe(tmp_path):
    pipeline = AudioPipeline(
        transcriber=MagicMock(),
        emotion_classifier=MagicMock(spec=VocalEmotionClassifier),
        language_client=MagicMock(spec=AzureLanguageClient),
    )
    with pytest.raises(FileNotFoundError):
        pipeline.process(tmp_path / "inexistente.wav")


@pytest.mark.smoke
def test_audio_pipeline_agrega_resultados_em_audio_analysis(tmp_path):
    import soundfile as sf

    sr = 16_000
    sinal = 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 1.0, sr, endpoint=False)).astype(
        np.float32
    )
    audio_path = tmp_path / "tone.wav"
    sf.write(str(audio_path), sinal, sr, subtype="PCM_16")

    transcriber = MagicMock()
    transcriber.transcribe.return_value = (
        "ola mundo",
        [Segment(start_ms=0, end_ms=1000, text="ola mundo")],
    )

    emotion = MagicMock(spec=VocalEmotionClassifier)
    emotion.classify.return_value = EmotionScore(
        label="neutral", confidence=0.6, scores={"neutral": 0.6}
    )

    language = MagicMock(spec=AzureLanguageClient)
    language.analyze_sentiment.return_value = SentimentResult(
        label="positive", confidence=0.9, scores={"positive": 0.9, "neutral": 0.1, "negative": 0.0}
    )
    language.extract_key_phrases.return_value = ["ola", "mundo"]

    pipeline = AudioPipeline(
        transcriber=transcriber,
        emotion_classifier=emotion,
        language_client=language,
    )
    analise = pipeline.process(audio_path)
    assert isinstance(analise, AudioAnalysis)
    assert analise.transcription == "ola mundo"
    assert analise.acoustic_features is not None
    assert analise.emotion is not None and analise.emotion.label == "neutral"
    assert analise.sentiment is not None and analise.sentiment.label == "positive"
    assert analise.key_phrases == ["ola", "mundo"]
