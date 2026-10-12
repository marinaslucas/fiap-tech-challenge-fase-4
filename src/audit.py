"""Log de auditoria em SQLite.

Registra cada caso processado pelo orquestrador com timestamp,
modalidades, nivel de risco, triggers e ids relacionados. Schema pensado
para consulta direta via SQL e para a aba "Auditoria" da UI.

Default `data/processed/audit.sqlite` na raiz. Diretorio e schema sao
criados sob demanda na primeira escrita.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from src.anomaly.types import AnomalyResult
from src.config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH: Path = Path("data/processed/audit.sqlite")

SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    modalities TEXT NOT NULL,
    triggers_json TEXT NOT NULL,
    explanation TEXT,
    report_md TEXT,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_case_id ON audit_log(case_id);
CREATE INDEX IF NOT EXISTS idx_audit_risk_level ON audit_log(risk_level);
"""


class AuditLogger:
    """Logger SQLite simples para registrar e listar casos."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Configura o caminho do banco e garante o schema.

        Args:
            db_path: caminho do arquivo SQLite. Se relativo, e resolvido
                contra a raiz do projeto. Default em `DEFAULT_DB_PATH`.
        """
        path = db_path or DEFAULT_DB_PATH
        self.db_path: Path = path if path.is_absolute() else settings.project_root / path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Cria diretorio pai e schema se ainda nao existirem."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)

    def log_case(
        self,
        *,
        case_id: str,
        anomaly: AnomalyResult,
        modalities: list[str],
        report_md: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Insere um registro e retorna o `id` gerado.

        Args:
            case_id: identificador externo do caso.
            anomaly: resultado consolidado.
            modalities: lista das modalidades processadas
                (ex.: `["video", "audio"]`).
            report_md: texto markdown do relatorio gerado.
            metadata: dict serializavel com paciente, modelos usados, etc.

        Returns:
            `id` da linha inserida (`audit_log.id`).
        """
        created_at = datetime.now(UTC).isoformat()
        triggers_payload = [t.model_dump() for t in anomaly.triggers]
        row = (
            case_id,
            created_at,
            anomaly.level,
            json.dumps(modalities, ensure_ascii=False),
            json.dumps(triggers_payload, ensure_ascii=False),
            anomaly.explanation,
            report_md,
            json.dumps(metadata or {}, ensure_ascii=False, default=str),
        )
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO audit_log (
                    case_id, created_at, risk_level, modalities,
                    triggers_json, explanation, report_md, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            conn.commit()
            audit_id = int(cursor.lastrowid or 0)
        logger.info("Audit %s registrado (id=%d, risco=%s)", case_id, audit_id, anomaly.level)
        return audit_id

    def list_cases(self, limit: int = 100) -> list[dict]:
        """Lista os ultimos casos registrados, mais recentes primeiro.

        Args:
            limit: numero maximo de registros a retornar.

        Returns:
            Lista de dicts com as colunas principais (sem markdown).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, case_id, created_at, risk_level, modalities, explanation
                FROM audit_log ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_case(self, audit_id: int) -> dict | None:
        """Retorna um registro completo pelo `id` (ou `None` se nao existir)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM audit_log WHERE id = ?", (audit_id,)).fetchone()
        return dict(row) if row else None
