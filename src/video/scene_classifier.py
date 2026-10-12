"""Classificador leve de tipo de cena para videos medicos.

Amostra alguns frames do video e decide, a partir de heuristica baseada em
deteccao de face (MediaPipe Face Detection) e assinatura de matiz HSV, se o
video e de:

- `cirurgia`  : campo cirurgico laparoscopico (poucos rostos, matiz vermelho-rosado escuro)
- `consulta`  : consulta clinica/ambulatorial (rostos presentes, paleta de pele/ambiente)
- `misto`     : mistura dos dois perfis
- `desconhecido`: falha ao ler frames ou dependencias indisponiveis

O classificador usa lazy import de `mediapipe`. Quando o pacote nao estiver
disponivel, retorna `SceneType.UNKNOWN` sem quebrar o pipeline.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Heuristica de cor (HSV) pra distinguir cirurgia de consulta
# -----------------------------------------------------------------------
# Investigacao empirica nos 4 videos de demo (out/2025):
#   cirurgia_rotina        sat=74.2 hue=41.3
#   cirurgia_sangramento   sat=102.9 hue=121.2
#   consulta_dermatologica sat=37.8 hue=47.5
#   consulta_clinica_geral sat=53.6 hue=67.1
# Saturacao discrimina limpo (cirurgias 74-103, consultas 38-54).
# Hue varia demais entre cirurgias (compress do MP4 muda o canal de cor)
# pra ser confiavel sozinho. Usamos saturacao como sinal primario.
_SURGERY_SAT_MIN: float = 65.0   # threshold entre consulta (max 54) e cirurgia (min 74)

# -----------------------------------------------------------------------
# Limiares de decisao
# -----------------------------------------------------------------------
_FACE_MAJORITY_RATIO: float = 0.5   # >= 50 % dos frames com face -> consulta candidata
_SURGERY_SAT_RATIO: float = 0.4     # >= 40 % dos frames saturados -> cirurgia candidata


class SceneType(StrEnum):
    """Tipo de cena identificada no video."""

    SURGERY = "cirurgia"
    CONSULTATION = "consulta"
    MIXED = "misto"
    UNKNOWN = "desconhecido"


def classify_scene_type(
    video_path: Path,
    num_samples: int = 5,
) -> SceneType:
    """Classifica o tipo de cena a partir de frames amostrados.

    Amostra `num_samples` frames distribuidos uniformemente pelo video. Para
    cada frame calcula:
    - `has_face`: via MediaPipe Face Detection (lazy import).
    - `hue_signature`: media de matiz e saturacao no espaco HSV.

    A decisao final combina a proporcao de frames com face e a assinatura
    de matiz para diferenciar consulta de cirurgia.

    Args:
        video_path: caminho para o arquivo de video.
        num_samples: quantidade de frames amostrados. Default 5.

    Returns:
        Um valor de `SceneType`. Retorna `UNKNOWN` quando nao e possivel
        ler frames ou quando `mediapipe` nao esta disponivel.
    """
    import cv2

    face_detector = _load_face_detector()

    video_path = Path(video_path).resolve()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("SceneClassifier: nao conseguiu abrir %s", video_path)
        return SceneType.UNKNOWN

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning(
                "SceneClassifier: video sem frames validos em %s", video_path
            )
            return SceneType.UNKNOWN

        indices = _sample_indices(total_frames, num_samples)
        face_hits: int = 0
        surgery_sat_hits: int = 0
        valid_frames: int = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            valid_frames += 1

            if _has_face(frame, face_detector):
                face_hits += 1

            if _is_surgery_color(frame):
                surgery_sat_hits += 1

    finally:
        cap.release()

    if valid_frames == 0:
        logger.warning("SceneClassifier: nenhum frame legivel em %s", video_path)
        return SceneType.UNKNOWN

    face_ratio = face_hits / valid_frames
    surgery_ratio = surgery_sat_hits / valid_frames

    logger.debug(
        "SceneClassifier %s: valid=%d face_ratio=%.2f surgery_sat_ratio=%.2f",
        video_path.name,
        valid_frames,
        face_ratio,
        surgery_ratio,
    )

    return _decide(face_ratio, surgery_ratio)


# -----------------------------------------------------------------------
# Helpers internos
# -----------------------------------------------------------------------


def _sample_indices(total_frames: int, num_samples: int) -> list[int]:
    """Retorna indices uniformemente distribuidos pelo video."""
    if num_samples <= 0:
        return []
    if num_samples >= total_frames:
        return list(range(total_frames))
    step = total_frames / num_samples
    return [int(i * step) for i in range(num_samples)]


def _load_face_detector():
    """Carrega MediaPipe Face Detection. Retorna None se indisponivel."""
    try:
        import mediapipe as mp

        face_detection = mp.solutions.face_detection
        detector = face_detection.FaceDetection(
            model_selection=0,  # modelo 0 = curta distancia, mais leve
            min_detection_confidence=0.3,  # mais permissivo pra detectar
            # face com mascara (so olhos+sobrancelha visiveis)
        )
        logger.debug("SceneClassifier: MediaPipe Face Detection carregado.")
        return detector
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "SceneClassifier: mediapipe indisponivel (%s). "
            "Heuristica de face desabilitada; classificacao usara apenas HSV.",
            exc,
        )
        return None


def _has_face(frame: np.ndarray, detector) -> bool:
    """Retorna True se o detector encontrar pelo menos uma face no frame.

    Se `detector` for None (mediapipe indisponivel), sempre retorna False.
    """
    if detector is None:
        return False
    rgb = frame[..., ::-1]  # BGR -> RGB
    result = detector.process(rgb)
    return bool(result.detections)


def _is_surgery_color(frame: np.ndarray) -> bool:
    """Retorna True se a assinatura HSV do frame e tipica de cirurgia laparoscopica.

    Criterio: saturacao media acima de `_SURGERY_SAT_MIN`. Campo cirurgico
    laparoscopico tem cores saturadas (tecido biologico iluminado por luz
    cirurgica) vs consulta clinica indoor onde paredes brancas e roupas
    neutras dessaturam a cena.

    Hue nao e usado: cirurgias com encoding MP4 diferentes podem ter shift
    no canal de cor (rotina=amarelo, sangramento=ciano apos compressao),
    tornando hue nao confiavel sozinho.
    """
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    return float(np.mean(sat)) >= _SURGERY_SAT_MIN


def _decide(face_ratio: float, surgery_ratio: float) -> SceneType:
    """Aplica as regras de decisao final.

    Regras (em ordem de prioridade):
    1. Maioria de frames com face E saturacao moderada (nao cirurgia) -> consulta.
    2. Poucos frames com face E hue tipico de cirurgia -> cirurgia.
    3. Faces + paleta cirurgica (ex: cirurgiao em campo) -> misto.
    4. Sem faces nem paleta cirurgica -> consulta (fallback). Cenas com
       paciente mascarado (EPI) fazem o detector perder o rosto, e na
       ausencia de assinatura cirurgica e mais provavel que seja consulta
       clinica do que cirurgia. Default mais conservador, evita falso
       positivo do YOLO laparoscopico em cenas de consulta.
    """
    has_faces = face_ratio >= _FACE_MAJORITY_RATIO
    is_surgery_color = surgery_ratio >= _SURGERY_SAT_RATIO

    if has_faces and not is_surgery_color:
        return SceneType.CONSULTATION
    if not has_faces and is_surgery_color:
        return SceneType.SURGERY
    if has_faces and is_surgery_color:
        return SceneType.MIXED
    # Sem faces detectadas: pelo sinal de cor decide.
    # Saturacao alta sem face -> cirurgia (laparoscopia sem rosto na cena).
    # Saturacao baixa sem face -> consulta (face nao detectada, ex: mascara).
    if is_surgery_color:
        return SceneType.SURGERY
    return SceneType.CONSULTATION
