"""Sistema de alertas a partir do `AnomalyResult`.

Estrutura um `Alert` auditavel e dispara no canal local (logger + console).
Canais externos (e-mail, SMS, webhook) podem ser adicionados registrando
novos `dispatcher`s em `dispatch_alert` sem alterar o orquestrador.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from src.anomaly.types import AnomalyResult, RiskLevel, Trigger

logger = logging.getLogger(__name__)


class Alert(BaseModel):
    """Estrutura serializavel de um alerta gerado a partir da anomalia."""

    case_id: str
    level: RiskLevel
    title: str
    summary: str
    triggers: list[Trigger] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def headline(self) -> str:
        """Texto curto em uma linha para logs e notificacoes."""
        return f"[{self.level.upper()}] {self.title}"


def build_alert(*, case_id: str, anomaly: AnomalyResult) -> Alert | None:
    """Constroi um `Alert` a partir do `AnomalyResult`.

    Args:
        case_id: identificador do caso (vindo do orquestrador).
        anomaly: resultado da analise de anomalia.

    Returns:
        `Alert` quando o nivel for `moderate` ou `critical`; `None` para
        `normal` (nao gera alerta).
    """
    if anomaly.level == "normal":
        return None
    title = _title_for_level(anomaly.level, anomaly.triggers)
    summary = anomaly.explanation or "Anomalia detectada."
    return Alert(
        case_id=case_id,
        level=anomaly.level,
        title=title,
        summary=summary,
        triggers=anomaly.triggers,
    )


def _title_for_level(level: RiskLevel, triggers: list[Trigger]) -> str:
    """Gera um titulo curto: usa o trigger mais grave quando disponivel."""
    if not triggers:
        return f"Anomalia {level} detectada"
    critical = [t for t in triggers if t.level == "critical"]
    if critical:
        return critical[0].message
    return triggers[0].message


Dispatcher = Callable[[Alert], None]


def _log_dispatcher(alert: Alert) -> None:
    """Dispatcher padrao: escreve no logger nivel apropriado."""
    if alert.level == "critical":
        logger.error("ALERTA CRITICO %s: %s", alert.case_id, alert.headline())
    else:
        logger.warning("ALERTA %s %s: %s", alert.level, alert.case_id, alert.headline())


def dispatch_alert(alert: Alert | None, dispatchers: list[Dispatcher] | None = None) -> None:
    """Envia o alerta para todos os dispatchers configurados.

    Args:
        alert: alerta a despachar; `None` e ignorado.
        dispatchers: lista de funcoes que aceitam um `Alert`. Quando vazio,
            usa apenas o dispatcher de log padrao.
    """
    if alert is None:
        return
    dispatcher_list: list[Dispatcher] = dispatchers or [_log_dispatcher]
    for dispatcher in dispatcher_list:
        try:
            dispatcher(alert)
        except Exception as exc:  # noqa: BLE001 - canais externos podem variar
            logger.warning(
                "Dispatcher %s falhou para alerta %s: %s",
                dispatcher.__name__,
                alert.case_id,
                exc,
            )
