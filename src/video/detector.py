"""Detector de objetos baseado em YOLO (model-agnostic).

A classe `BleedingDetector` (nome historico, agora generico) opera com
qualquer arquivo `.pt` compativel com YOLOv8. Por padrao usa o stub
`yolov8n.pt` (classes COCO, sem relevancia clinica direta) para destravar o
pipeline. Quando pesos custom de instrumentos cirurgicos (treinados em
CholecSeg8k) estao disponiveis em `YOLO_WEIGHTS_PATH`, o detector os
carrega automaticamente.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.config.settings import settings
from src.video.types import BoundingBox, Detection

logger = logging.getLogger(__name__)

# Stub baixado automaticamente pela ultralytics se nenhum peso custom existir.
DEFAULT_STUB_MODEL = "yolov8n.pt"


def ensure_yolo_weights(target_path: Path) -> Path:
    """Garante que o arquivo de pesos YOLO exista em `target_path`.

    Se o arquivo nao existir, baixa o stub `yolov8n.pt` via ultralytics e
    move para o local destino. Para usar pesos custom, basta substituir o
    arquivo no mesmo path que o detector continua funcionando.

    Args:
        target_path: caminho desejado para os pesos `.pt`.

    Returns:
        Caminho absoluto dos pesos garantidos no disco.

    Raises:
        FileNotFoundError: se o download do stub falhar.
    """
    target_path = target_path.resolve()
    if target_path.exists():
        logger.debug("Pesos YOLO ja presentes em %s", target_path)
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Pesos %s nao encontrados; baixando stub %s via ultralytics",
        target_path,
        DEFAULT_STUB_MODEL,
    )
    # ultralytics baixa o arquivo no cwd quando o nome nao e um path absoluto.
    YOLO(DEFAULT_STUB_MODEL)
    cached_path = Path(DEFAULT_STUB_MODEL).resolve()
    if not cached_path.exists():
        raise FileNotFoundError(
            f"Falha ao baixar {DEFAULT_STUB_MODEL}. Verifique a conexao com a internet."
        )
    shutil.move(str(cached_path), str(target_path))
    logger.info("Pesos do stub copiados para %s", target_path)
    return target_path


class BleedingDetector:
    """Detector de objetos por frame, baseado em YOLOv8.

    Mantem o modelo carregado em memoria entre chamadas. Lazy-load: o modelo
    so e carregado na primeira chamada a `predict()` ou via `load()` explicito.
    """

    def __init__(
        self,
        weights_path: Path | None = None,
        confidence_threshold: float = 0.25,
    ) -> None:
        """Configura o detector.

        Args:
            weights_path: caminho do arquivo `.pt`. Se `None`, usa
                `settings.yolo_weights_absolute()`.
            confidence_threshold: confianca minima para retornar uma deteccao.
        """
        self.weights_path: Path = (
            weights_path.resolve() if weights_path else settings.yolo_weights_absolute()
        )
        self.confidence_threshold: float = confidence_threshold
        self._model: YOLO | None = None

    def load(self) -> None:
        """Carrega o modelo em memoria, baixando o stub se necessario."""
        ensure_yolo_weights(self.weights_path)
        self._model = YOLO(str(self.weights_path))
        logger.info("Modelo YOLO carregado de %s", self.weights_path)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Detecta objetos em um unico frame.

        Args:
            frame: imagem em formato BGR (HxWx3) compativel com OpenCV.

        Returns:
            Lista de `Detection` com confianca >= `confidence_threshold`.
        """
        if self._model is None:
            self.load()
        assert self._model is not None  # narrowing para o type checker
        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
        )
        return list(self._parse_results(results))

    def _parse_results(self, results) -> Iterable[Detection]:
        """Converte saida do ultralytics em sequencia de `Detection`."""
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for cls_id, conf, xyxy in zip(
                boxes.cls.tolist(),
                boxes.conf.tolist(),
                boxes.xyxy.tolist(),
                strict=True,
            ):
                yield Detection(
                    class_id=int(cls_id),
                    class_name=names[int(cls_id)],
                    confidence=float(conf),
                    bbox=BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    ),
                )
