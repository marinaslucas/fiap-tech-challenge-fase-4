"""Tipos compartilhados do pipeline RAG.

`Chunk` representa um trecho indexado no vector store, e `RetrievalResult`
agrega a resposta do `Retriever` (chunks + scores).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Trecho de documento indexado no vector store.

    `chunk_id` deve ser estavel entre execucoes para permitir upsert sem
    duplicacao (`source-page-index` cumpre o papel).
    """

    text: str
    source: str
    section: str | None = None
    page: int | None = None
    chunk_id: str

    def metadata(self) -> dict[str, str | int]:
        """Devolve metadata pronta para o vector store (sem `text` ou `chunk_id`)."""
        meta: dict[str, str | int] = {"source": self.source}
        if self.section is not None:
            meta["section"] = self.section
        if self.page is not None:
            meta["page"] = self.page
        return meta


class RetrievalResult(BaseModel):
    """Resultado da busca: chunks recuperados e scores associados (mesma ordem)."""

    chunks: list[Chunk] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
