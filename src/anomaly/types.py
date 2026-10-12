"""Tipos compartilhados do modulo de anomalia.

Modelos Pydantic v2 produzidos pelas regras clinicas (`rules.py`) e pelo
classificador estatistico (`statistical.py`), agregados em `AnomalyResult`
pelo `classifier.py`. Manter aqui evita imports circulares e mantem
o contrato do orquestrador estavel.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Niveis de risco em ordem crescente de gravidade. Mantido como Literal para
# tipagem estatica e validacao Pydantic; a ordenacao numerica vive em
# `RISK_LEVEL_ORDER` (modulo rules) para evitar magic strings.
RiskLevel = Literal["normal", "moderate", "critical"]


class Trigger(BaseModel):
    """Sinal disparado por uma regra clinica ou modelo estatistico.

    Cada trigger e atomico: representa um unico motivo de preocupacao,
    rastreavel ate a regra que o gerou. O classificador final agrega
    multiplos triggers em um `AnomalyResult` consolidado.
    """

    rule_id: str
    level: RiskLevel
    message: str
    source: Literal["video", "audio", "text", "fusion"]
    evidence: dict = Field(default_factory=dict)


class AnomalyResult(BaseModel):
    """Resultado final da analise de anomalia para um caso.

    Saida consumida pelo gerador de relatorio e pelo sistema de alerta.
    O `level` reflete o trigger mais grave; `triggers` preserva o detalhe
    para auditoria; `explanation` e texto em PT-BR pronto para humanos.
    """

    level: RiskLevel
    triggers: list[Trigger] = Field(default_factory=list)
    explanation: str = ""
    recommended_actions: list[str] = Field(default_factory=list)
