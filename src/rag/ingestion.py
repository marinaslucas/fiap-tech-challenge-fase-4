"""Ingestao de documentos para o RAG.

Pipeline: arquivo (PDF ou TXT) -> texto por pagina -> chunks com metadata.
Chunking e feito por separadores naturais (paragrafos, frases) com janela
deslizante e overlap configuravel para preservar contexto entre trechos.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.rag.types import Chunk

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Extrai texto de um PDF, pagina a pagina.

    Args:
        path: caminho do arquivo `.pdf`.

    Returns:
        Lista de tuplas `(numero_pagina_1_based, texto)`. Paginas vazias sao
        omitidas para evitar lixo no indice.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i, text))
    logger.info("PDF %s carregado: %d paginas com texto", path, len(pages))
    return pages


def load_text(path: Path) -> list[tuple[int, str]]:
    """Carrega um arquivo `.txt` ou `.md` como pagina unica.

    Util para fallback quando o PDF nao esta disponivel.

    Returns:
        Lista com 1 entrada `[(1, conteudo)]` ou vazia se o arquivo for vazio.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [(1, text)]


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Quebra texto em chunks por separadores naturais com overlap.

    Estrategia:
        1. Divide em paragrafos (`\\n\\n`)
        2. Junta paragrafos consecutivos ate atingir `chunk_size`
        3. Mantem overlap entre chunks consecutivos para preservar contexto

    Args:
        text: texto bruto.
        chunk_size: tamanho alvo (em caracteres) de cada chunk.
        overlap: quantos caracteres de cauda do chunk anterior repetir.

    Returns:
        Lista de strings, cada uma com ate `chunk_size + overlap` caracteres.
    """
    if not text.strip():
        return []
    # Normaliza quebras e divide por paragrafo.
    text = re.sub(r"\r\n", "\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if not buffer:
            buffer = paragraph
            continue
        if len(buffer) + 2 + len(paragraph) <= chunk_size:
            buffer = f"{buffer}\n\n{paragraph}"
        else:
            chunks.append(buffer)
            tail = buffer[-overlap:] if overlap > 0 else ""
            buffer = f"{tail}\n\n{paragraph}".lstrip("\n")
    if buffer:
        chunks.append(buffer)
    return chunks


def ingest_document(
    path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Carrega um documento e devolve chunks com metadata.

    Suporta `.pdf`, `.txt` e `.md`. O `chunk_id` segue o padrao
    `<source>-p<page>-c<index>` para ser estavel entre execucoes.

    Args:
        path: caminho do arquivo.
        chunk_size: tamanho alvo dos chunks em caracteres.
        overlap: overlap em caracteres entre chunks consecutivos.

    Returns:
        Lista de `Chunk` prontos para indexacao.

    Raises:
        FileNotFoundError: se o arquivo nao existir.
        ValueError: se a extensao nao for suportada.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Documento nao encontrado: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = load_pdf(path)
    elif suffix in {".txt", ".md"}:
        pages = load_text(path)
    else:
        raise ValueError(f"Extensao nao suportada: {suffix}")

    source = path.stem
    chunks: list[Chunk] = []
    for page_num, page_text in pages:
        for idx, body in enumerate(chunk_text(page_text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    text=body,
                    source=source,
                    page=page_num,
                    chunk_id=f"{source}-p{page_num}-c{idx}",
                )
            )
    logger.info("Documento %s ingerido: %d chunks", path.name, len(chunks))
    return chunks
