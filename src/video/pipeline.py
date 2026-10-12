"""Pipeline de video completo.

Recebe um caminho de video, amostra frames a uma taxa configuravel
(`target_fps`) e roda em cada frame: detector YOLO e classificador
multimodal de emocao e linguagem corporal (via GPT-vision). A saida e
uma lista de `VideoEvent` com tudo agregado por frame.

Logica de gating por tipo de cena (2 pilares: detector e emocao):

| Cena         | YOLO (instrumentos) | Emocao + linguagem corporal |
|--------------|---------------------|-----------------------------|
| SURGERY      | rodando             | pulado                      |
| CONSULTATION | pulado              | rodando                     |
| MIXED        | rodando             | rodando                     |
| UNKNOWN      | rodando             | rodando                     |

Em CONSULTATION o YOLO custom (treinado em laparoscopia) gera falso
positivo em objetos clinicos comuns como agulhas, seringas, otoscopios,
classificando como Grasper/L-hook. Em SURGERY o pilar humano nao tem
sinal porque o campo cirurgico nao expoe a paciente.

Amostragem de emocao reduzida por padrao (`emotion_every_n_samples=3`):
classificacao via GPT-vision tem latencia ~3s/frame. Como paciente em
consulta clinica raramente muda emocao em <3s, amostrar 1 a cada 3
frames reduz tempo total em 3x sem perder resolucao temporal relevante.
Frames sem analise continuam no resultado com `facial_emotion=None`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import cv2

from src.video.detector import BleedingDetector
from src.video.emotion import FacialEmotionClassifierProtocol, get_facial_emotion_classifier
from src.video.scene_classifier import SceneType, classify_scene_type
from src.video.types import VideoEvent

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


class VideoPipeline:
    """Orquestra todos os modulos de analise de video frame a frame."""

    def __init__(
        self,
        target_fps: float = 1.0,
        emotion_every_n_samples: int = 3,
        detector: BleedingDetector | None = None,
        emotion_classifier: FacialEmotionClassifierProtocol | None = None,
    ) -> None:
        """Configura o pipeline.

        Args:
            target_fps: quantos frames por segundo amostrar do video.
                Default 1 fps mantem custo baixo para videos longos.
            emotion_every_n_samples: roda classificacao a cada N frames
                amostrados (default 3). Reduz latencia GPT-vision em 3x
                sem perder resolucao real em consultas clinicas. Use 1
                pra rodar em todos os frames amostrados.
            detector: instancia opcional de `BleedingDetector`.
            emotion_classifier: instancia opcional de classificador. Default
                usa `get_facial_emotion_classifier()` que seleciona GPT-4o
                vision (quando configurado) ou FER local.
        """
        self.target_fps: float = target_fps
        self.emotion_every_n_samples: int = max(1, emotion_every_n_samples)
        self.detector: BleedingDetector = detector or BleedingDetector()
        self.emotion_classifier: FacialEmotionClassifierProtocol = (
            emotion_classifier or get_facial_emotion_classifier()
        )
        # Preenchido apos cada chamada a `process()`.
        self.last_scene_type: SceneType = SceneType.UNKNOWN

    def process(
        self,
        video_path: Path,
        progress: ProgressCallback | None = None,
    ) -> list[VideoEvent]:
        """Processa um video e retorna a lista de `VideoEvent`.

        Args:
            video_path: caminho do arquivo de video.
            progress: callback opcional `(frac, desc)` com `frac` em [0, 1]
                e `desc` string curta. Util pra integrar com `gr.Progress`
                ou outros indicadores de UI. Default `None` (sem reportes).

        Returns:
            Lista de eventos, um por frame amostrado.

        Raises:
            FileNotFoundError: se o video nao existir.
            RuntimeError: se o OpenCV nao conseguir abrir o arquivo.
        """
        video_path = video_path.resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video nao encontrado: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Falha ao abrir video com OpenCV: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            sampling_step = max(1, int(round(video_fps / self.target_fps)))
            expected_samples = max(1, total_video_frames // sampling_step)
            logger.info(
                "Processando %s: video_fps=%.2f, target_fps=%.2f, step=%d, samples~%d",
                video_path,
                video_fps,
                self.target_fps,
                sampling_step,
                expected_samples,
            )

            if progress is not None:
                progress(0.02, "Detectando tipo de cena...")

            self.last_scene_type = classify_scene_type(video_path)
            logger.info("Tipo de cena identificado: %s", self.last_scene_type.value)

            # Emocao + linguagem corporal so fazem sentido em cenas com
            # paciente visivel. Em cirurgia laparoscopica o campo cirurgico
            # nao expoe faces nem corpos, entao pular evita custo e ruido.
            run_emotion = self.last_scene_type != SceneType.SURGERY
            if not run_emotion:
                logger.info(
                    "Cena SURGERY: classificacao de emocao/linguagem corporal ignorada."
                )

            # Deteccao YOLO so faz sentido em cenas cirurgicas. Em consulta,
            # o modelo (treinado em laparoscopia) gera falso positivo em
            # objetos comuns de exame fisico (agulhas, seringas, otoscopios)
            # classificando como Grasper/L-hook.
            run_detection = self.last_scene_type != SceneType.CONSULTATION
            if not run_detection:
                logger.info(
                    "Cena CONSULTATION: deteccao de instrumentos ignorada "
                    "(evita falso positivo do YOLO laparoscopico)."
                )

            if progress is not None:
                progress(
                    0.05,
                    f"Cena: {self.last_scene_type.value}. Iniciando processamento...",
                )

            events: list[VideoEvent] = []
            frame_idx = 0
            sample_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sampling_step != 0:
                    frame_idx += 1
                    continue

                timestamp_ms = int(round(frame_idx * 1000 / video_fps))
                detections = self.detector.predict(frame) if run_detection else []

                # Emocao roda so a cada N amostras (default 3) pra reduzir
                # latencia de chamada GPT-vision sem perder resolucao real
                should_classify_emotion = (
                    run_emotion and sample_idx % self.emotion_every_n_samples == 0
                )
                facial_emotion = (
                    self.emotion_classifier.classify(frame)
                    if should_classify_emotion
                    else None
                )

                events.append(
                    VideoEvent(
                        frame_index=frame_idx,
                        timestamp_ms=timestamp_ms,
                        detections=detections,
                        facial_emotion=facial_emotion,
                    )
                )

                if progress is not None and sample_idx % 2 == 0:
                    frac = 0.05 + 0.92 * min(
                        1.0, (sample_idx + 1) / max(1, expected_samples)
                    )
                    progress(
                        frac,
                        f"Frame {sample_idx + 1}/{expected_samples}",
                    )

                frame_idx += 1
                sample_idx += 1

            if progress is not None:
                progress(1.0, "Concluido.")

            logger.info("Processamento concluido: %d eventos", len(events))
            return events
        finally:
            cap.release()
