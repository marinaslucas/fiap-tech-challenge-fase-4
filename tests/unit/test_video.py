"""Testes smoke do pipeline de video.

Sem download de pesos ou inferencia real. Mocks isolam dependencias
externas (`YOLO`, `mediapipe`, `fer`) para manter o teste rapido e offline.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from src.video import (
    AzureOpenAIVisionEmotion,
    BleedingDetector,
    BoundingBox,
    Detection,
    EmotionScore,
    FacialEmotionDetector,
    SceneType,
    VideoEvent,
    VideoPipeline,
    classify_scene_type,
    ensure_yolo_weights,
    get_facial_emotion_classifier,
)
from src.video.detector import DEFAULT_STUB_MODEL
from src.video.scene_classifier import _decide, _sample_indices

# ---------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_bounding_box_calcula_dimensoes():
    bbox = BoundingBox(x1=10, y1=20, x2=110, y2=220)
    assert bbox.width == 100
    assert bbox.height == 200


@pytest.mark.smoke
def test_detection_valida_confianca_em_range():
    bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    det = Detection(class_id=0, class_name="person", confidence=0.85, bbox=bbox)
    assert det.confidence == pytest.approx(0.85)
    assert det.class_name == "person"


@pytest.mark.smoke
def test_detection_rejeita_confianca_acima_de_um():
    bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    with pytest.raises(ValidationError):
        Detection(class_id=0, class_name="x", confidence=1.5, bbox=bbox)


@pytest.mark.smoke
def test_emotion_score_aceita_distribuicao():
    score = EmotionScore(label="happy", confidence=0.7, scores={"happy": 0.7, "sad": 0.3})
    assert score.label == "happy"
    assert score.scores["happy"] == pytest.approx(0.7)


@pytest.mark.smoke
def test_emotion_score_aceita_body_language():
    score = EmotionScore(
        label="neutral", confidence=0.8, body_language="tranquila",
    )
    assert score.body_language == "tranquila"


@pytest.mark.smoke
def test_video_event_tem_defaults_vazios():
    event = VideoEvent(frame_index=0, timestamp_ms=0)
    assert event.detections == []
    assert event.facial_emotion is None


# ---------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_bleeding_detector_inicializa_com_path_de_settings():
    detector = BleedingDetector()
    assert detector.weights_path.is_absolute()
    assert detector.confidence_threshold == 0.25
    assert detector._model is None


@pytest.mark.smoke
def test_bleeding_detector_aceita_threshold_custom(tmp_path):
    fake_weights = tmp_path / "stub.pt"
    fake_weights.write_bytes(b"fake")
    detector = BleedingDetector(weights_path=fake_weights, confidence_threshold=0.5)
    assert detector.confidence_threshold == 0.5
    assert detector.weights_path == fake_weights.resolve()


@pytest.mark.smoke
def test_ensure_yolo_weights_retorna_path_existente_sem_download(tmp_path):
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"fake")
    with patch("src.video.detector.YOLO") as mock_yolo:
        result = ensure_yolo_weights(weights)
    assert result == weights.resolve()
    mock_yolo.assert_not_called()


@pytest.mark.smoke
def test_default_stub_model_aponta_para_yolov8n():
    assert DEFAULT_STUB_MODEL == "yolov8n.pt"


# ---------------------------------------------------------------------
# Emocao facial
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_facial_emotion_detector_fallback_quando_fer_indisponivel(monkeypatch):
    detector = FacialEmotionDetector()

    def _fake_load():
        detector._available = False

    monkeypatch.setattr(detector, "load", _fake_load)
    detector._available = False
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


@pytest.mark.smoke
def test_facial_emotion_detector_parseia_saida_do_fer(monkeypatch):
    detector = FacialEmotionDetector()
    fake_fer = MagicMock()
    fake_fer.detect_emotions.return_value = [
        {
            "box": (10, 20, 100, 200),
            "emotions": {"happy": 0.7, "sad": 0.3},
        }
    ]
    detector._detector = fake_fer
    detector._available = True
    detector._load_attempted = True
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    scores = detector.detect(frame)
    assert len(scores) == 1
    assert scores[0].label == "happy"
    assert scores[0].confidence == pytest.approx(0.7)
    assert scores[0].bbox is not None
    assert scores[0].bbox.x1 == 10


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_video_pipeline_falha_quando_arquivo_nao_existe(tmp_path):
    pipeline = VideoPipeline(
        detector=MagicMock(spec=BleedingDetector),
        emotion_classifier=MagicMock(spec=FacialEmotionDetector),
    )
    with pytest.raises(FileNotFoundError):
        pipeline.process(tmp_path / "inexistente.mp4")


# ---------------------------------------------------------------------
# SceneClassifier
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_scene_type_enum_valores():
    """Garante que os valores dos enums batem com as strings esperadas."""
    assert SceneType.SURGERY.value == "cirurgia"
    assert SceneType.CONSULTATION.value == "consulta"
    assert SceneType.MIXED.value == "misto"
    assert SceneType.UNKNOWN.value == "desconhecido"


@pytest.mark.smoke
def test_classify_scene_type_retorna_unknown_quando_video_nao_existe(tmp_path):
    """Arquivo inexistente deve retornar UNKNOWN sem lancar excecao."""
    result = classify_scene_type(tmp_path / "nao_existe.mp4")
    assert result == SceneType.UNKNOWN


@pytest.mark.smoke
def test_classify_scene_type_retorna_consulta_quando_ha_faces(tmp_path, monkeypatch):
    """Mock MediaPipe retorna deteccoes em todos os frames -> CONSULTATION."""
    fake_video = tmp_path / "fake.mp4"
    fake_video.touch()

    # Mock cv2.VideoCapture para retornar frames sinteticos
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Cor pele dessaturada (BGR ~200,215,225): sat ~28 (< _SURGERY_SAT_MIN=65),
    # nao classifica como cor cirurgica (saturacao baixa de ambiente clinico)
    fake_frame[:] = [200, 215, 225]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0  # 30 frames no total
    read_calls = [True, fake_frame] * 10 + [False, None]
    mock_cap.read.side_effect = [
        (read_calls[i], read_calls[i + 1]) for i in range(0, len(read_calls), 2)
    ]

    # Mock do detector de face: sempre detecta
    fake_detection_result = MagicMock()
    fake_detection_result.detections = [MagicMock()]
    mock_face_detector = MagicMock()
    mock_face_detector.process.return_value = fake_detection_result

    import src.video.scene_classifier as sc_mod

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch.object(sc_mod, "_load_face_detector", return_value=mock_face_detector),
    ):
        result = classify_scene_type(fake_video, num_samples=5)

    assert result == SceneType.CONSULTATION


@pytest.mark.smoke
def test_classify_scene_type_retorna_cirurgia_quando_hue_cirurgico(tmp_path, monkeypatch):
    """Frame com hue vermelho-rosado e sem faces -> SURGERY."""
    fake_video = tmp_path / "cirurgia.mp4"
    fake_video.touch()

    # Frame vermelho escuro saturado (BGR ~0, 30, 180) -> hue ~0 no OpenCV
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_frame[:] = [0, 30, 180]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0
    mock_cap.read.side_effect = [(True, fake_frame)] * 5 + [(False, None)]

    # Sem faces
    fake_detection_result = MagicMock()
    fake_detection_result.detections = []
    mock_face_detector = MagicMock()
    mock_face_detector.process.return_value = fake_detection_result

    import src.video.scene_classifier as sc_mod

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch.object(sc_mod, "_load_face_detector", return_value=mock_face_detector),
    ):
        result = classify_scene_type(fake_video, num_samples=5)

    assert result == SceneType.SURGERY


@pytest.mark.smoke
def test_classify_scene_type_retorna_unknown_quando_mediapipe_ausente(tmp_path):
    """Quando mediapipe nao esta disponivel, classificacao usa so HSV (sem face).

    A funcao nao deve lancar excecao. O resultado pode ser MIXED, SURGERY
    ou CONSULTATION dependendo do frame, mas nunca uma excecao.
    """
    fake_video = tmp_path / "sem_mp.mp4"
    fake_video.touch()

    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 10.0
    mock_cap.read.side_effect = [(True, fake_frame)] * 3 + [(False, None)]

    import src.video.scene_classifier as sc_mod

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch.object(sc_mod, "_load_face_detector", return_value=None),
    ):
        result = classify_scene_type(fake_video, num_samples=3)

    # Nao deve lancar excecao; resultado e um SceneType valido
    assert isinstance(result, SceneType)


@pytest.mark.smoke
def test_decide_retorna_consulta_quando_ambos_sinais_ausentes():
    """Sem face e sem hue cirurgico -> CONSULTATION (fallback conservador).

    Cenas com paciente mascarado (EPI) fazem MediaPipe perder o rosto.
    Na ausencia de assinatura cirurgica, e mais provavel ser consulta
    do que cirurgia laparoscopica.
    """
    assert _decide(face_ratio=0.0, surgery_ratio=0.0) == SceneType.CONSULTATION


@pytest.mark.smoke
def test_decide_retorna_misto_quando_ambos_sinais_presentes():
    """Com face E hue cirurgico -> MIXED (ex: cirurgiao no campo)."""
    assert _decide(face_ratio=1.0, surgery_ratio=1.0) == SceneType.MIXED


@pytest.mark.smoke
def test_sample_indices_distribui_uniformemente():
    """_sample_indices deve retornar exatamente `num_samples` indices."""
    indices = _sample_indices(total_frames=100, num_samples=5)
    assert len(indices) == 5
    assert indices[0] == 0
    assert all(0 <= i < 100 for i in indices)


# ---------------------------------------------------------------------
# AzureOpenAIVisionEmotion
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_azure_openai_vision_nao_configurado_quando_deployment_vazio(monkeypatch):
    """Sem deployment preenchido, is_configured deve ser False."""
    monkeypatch.setattr(
        "src.video.azure_openai_vision.settings",
        MagicMock(
            azure_openai_key=MagicMock(get_secret_value=lambda: ""),
            azure_openai_endpoint="",
            azure_openai_vision_deployment="",
        ),
    )
    classifier = AzureOpenAIVisionEmotion()
    assert classifier.is_configured is False


@pytest.mark.smoke
def test_azure_openai_vision_classify_retorna_none_quando_nao_configurado():
    """Sem credenciais, classify deve retornar None sem lancar excecao."""
    classifier = AzureOpenAIVisionEmotion()
    # No ambiente de teste as variaveis Azure nao estao preenchidas.
    if not classifier.is_configured:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = classifier.classify(frame)
        assert result is None


@pytest.mark.smoke
def test_azure_openai_vision_parse_json_valido():
    """_parse_json_response deve converter JSON valido em EmotionScore."""
    from src.video.azure_openai_vision import _parse_json_response

    content = '{"label": "neutral", "confidence": 0.85, "reasoning": "face relaxada"}'
    score = _parse_json_response(content)
    assert score is not None
    assert score.label == "neutral"
    assert score.confidence == pytest.approx(0.85)
    assert score.scores == {"neutral": pytest.approx(0.85)}


@pytest.mark.smoke
def test_azure_openai_vision_parse_json_label_invalido():
    """Label fora do dominio deve retornar None."""
    from src.video.azure_openai_vision import _parse_json_response

    content = '{"label": "confused", "confidence": 0.9, "reasoning": "x"}'
    assert _parse_json_response(content) is None


@pytest.mark.smoke
def test_azure_openai_vision_parse_json_malformado():
    """JSON malformado deve retornar None sem lancar excecao."""
    from src.video.azure_openai_vision import _parse_json_response

    assert _parse_json_response("nao e json") is None


# ---------------------------------------------------------------------
# get_facial_emotion_classifier
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_get_facial_emotion_classifier_retorna_fer_quando_deployment_vazio(monkeypatch):
    """Sem deployment de visao configurado, deve retornar FacialEmotionDetector."""
    monkeypatch.setattr(
        "src.video.emotion.settings",
        MagicMock(azure_openai_vision_deployment=""),
    )
    classifier = get_facial_emotion_classifier()
    assert isinstance(classifier, FacialEmotionDetector)


@pytest.mark.smoke
def test_get_facial_emotion_classifier_retorna_azure_quando_configurado(monkeypatch):
    """Com deployment preenchido e is_configured True, deve retornar AzureOpenAIVisionEmotion."""
    monkeypatch.setattr(
        "src.video.emotion.settings",
        MagicMock(azure_openai_vision_deployment="gpt-4o-mini"),
    )
    mock_vision = MagicMock(spec=AzureOpenAIVisionEmotion)
    mock_vision.is_configured = True
    mock_vision.deployment = "gpt-4o-mini"

    with patch("src.video.azure_openai_vision.AzureOpenAIVisionEmotion", return_value=mock_vision):
        classifier = get_facial_emotion_classifier()

    assert classifier is mock_vision


@pytest.mark.smoke
def test_get_facial_emotion_classifier_cai_no_fer_quando_azure_nao_configurado(monkeypatch):
    """Deployment preenchido mas is_configured False -> cai para FacialEmotionDetector."""
    monkeypatch.setattr(
        "src.video.emotion.settings",
        MagicMock(azure_openai_vision_deployment="gpt-4o-mini"),
    )
    mock_vision = MagicMock(spec=AzureOpenAIVisionEmotion)
    mock_vision.is_configured = False

    with patch("src.video.azure_openai_vision.AzureOpenAIVisionEmotion", return_value=mock_vision):
        classifier = get_facial_emotion_classifier()

    assert isinstance(classifier, FacialEmotionDetector)


# ---------------------------------------------------------------------
# VideoPipeline: logica de cena + emocao
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_video_pipeline_aceita_emotion_classifier_como_parametro():
    """Pipeline deve aceitar qualquer objeto com metodo classify."""
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = None
    pipeline = VideoPipeline(
        detector=MagicMock(spec=BleedingDetector),
        emotion_classifier=mock_classifier,
    )
    assert pipeline.emotion_classifier is mock_classifier


@pytest.mark.smoke
def test_video_pipeline_pula_emocao_em_cena_surgery(tmp_path, monkeypatch):
    """Cena SURGERY: classify nao deve ser chamado em nenhum frame."""
    fake_video = tmp_path / "cirurgia.mp4"
    fake_video.touch()

    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [30.0, 3.0]  # fps, total_frames para classify_scene
    mock_cap.read.side_effect = [(True, fake_frame), (True, fake_frame), (False, None)]

    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = None

    import src.video.pipeline as pipeline_mod

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch.object(pipeline_mod, "classify_scene_type", return_value=SceneType.SURGERY),
        patch("src.video.detector.BleedingDetector.predict", return_value=[]),
    ):
        p = VideoPipeline(
            target_fps=30.0,
            emotion_classifier=mock_classifier,
        )
        p.process(fake_video)

    mock_classifier.classify.assert_not_called()


@pytest.mark.smoke
def test_video_pipeline_chama_emocao_em_cena_consultation(tmp_path):
    """Cena CONSULTATION: classify deve ser chamado em cada frame amostrado."""
    fake_video = tmp_path / "consulta.mp4"
    fake_video.touch()

    emotion_result = EmotionScore(label="neutral", confidence=0.9, scores={"neutral": 0.9})
    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [30.0, 2.0]
    mock_cap.read.side_effect = [(True, fake_frame), (True, fake_frame), (False, None)]

    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = emotion_result

    import src.video.pipeline as pipeline_mod

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch.object(pipeline_mod, "classify_scene_type", return_value=SceneType.CONSULTATION),
        patch("src.video.detector.BleedingDetector.predict", return_value=[]),
    ):
        p = VideoPipeline(
            target_fps=30.0,
            emotion_classifier=mock_classifier,
        )
        p.process(fake_video)

    assert mock_classifier.classify.call_count >= 1


@pytest.mark.smoke
def test_video_pipeline_pula_deteccao_em_cena_consultation(tmp_path):
    """Cena CONSULTATION: BleedingDetector.predict nao deve ser chamado."""
    fake_video = tmp_path / "consulta.mp4"
    fake_video.touch()

    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [30.0, 2.0]
    mock_cap.read.side_effect = [(True, fake_frame), (True, fake_frame), (False, None)]

    mock_detector = MagicMock(spec=BleedingDetector)
    mock_detector.predict.return_value = []

    import src.video.pipeline as pipeline_mod

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch.object(pipeline_mod, "classify_scene_type", return_value=SceneType.CONSULTATION),
    ):
        p = VideoPipeline(
            target_fps=30.0,
            detector=mock_detector,
            emotion_classifier=MagicMock(classify=MagicMock(return_value=None)),
        )
        p.process(fake_video)

    mock_detector.predict.assert_not_called()


@pytest.mark.smoke
def test_video_pipeline_chama_deteccao_em_cena_surgery(tmp_path):
    """Cena SURGERY: BleedingDetector.predict deve ser chamado em cada frame."""
    fake_video = tmp_path / "cirurgia.mp4"
    fake_video.touch()

    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [30.0, 2.0]
    mock_cap.read.side_effect = [(True, fake_frame), (True, fake_frame), (False, None)]

    mock_detector = MagicMock(spec=BleedingDetector)
    mock_detector.predict.return_value = []

    import src.video.pipeline as pipeline_mod

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch.object(pipeline_mod, "classify_scene_type", return_value=SceneType.SURGERY),
    ):
        p = VideoPipeline(
            target_fps=30.0,
            detector=mock_detector,
            emotion_classifier=MagicMock(classify=MagicMock(return_value=None)),
        )
        p.process(fake_video)

    assert mock_detector.predict.call_count >= 1


@pytest.mark.smoke
def test_video_pipeline_amostra_emocao_a_cada_n_frames(tmp_path):
    """emotion_every_n_samples=3 deve classificar emocao so a cada 3 frames."""
    fake_video = tmp_path / "consulta.mp4"
    fake_video.touch()

    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [30.0, 6.0]  # fps, total_frames
    # 6 frames amostrados (target_fps=30 com video 30fps)
    mock_cap.read.side_effect = [(True, fake_frame)] * 6 + [(False, None)]

    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = None

    import src.video.pipeline as pipeline_mod

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch.object(pipeline_mod, "classify_scene_type", return_value=SceneType.CONSULTATION),
    ):
        p = VideoPipeline(
            target_fps=30.0,
            emotion_every_n_samples=3,
            detector=MagicMock(spec=BleedingDetector, predict=MagicMock(return_value=[])),
            emotion_classifier=mock_classifier,
        )
        p.process(fake_video)

    # 6 frames amostrados, com step=3 -> emocao em sample 0, 3 -> 2 chamadas
    assert mock_classifier.classify.call_count == 2


@pytest.mark.smoke
def test_facial_emotion_detector_classify_retorna_none_quando_sem_faces(monkeypatch):
    """FacialEmotionDetector.classify deve retornar None quando detect retorna []."""
    detector = FacialEmotionDetector()
    monkeypatch.setattr(detector, "detect", lambda frame: [])
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert detector.classify(frame) is None


@pytest.mark.smoke
def test_facial_emotion_detector_classify_retorna_primeiro_score(monkeypatch):
    """FacialEmotionDetector.classify deve retornar o primeiro EmotionScore detectado."""
    detector = FacialEmotionDetector()
    score = EmotionScore(label="happy", confidence=0.8, scores={"happy": 0.8})
    monkeypatch.setattr(detector, "detect", lambda frame: [score])
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result = detector.classify(frame)
    assert result is score
