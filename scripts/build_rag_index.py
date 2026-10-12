"""Constroi o indice Chroma a partir dos PDFs em `data/raw/`.

Procura PDFs com prefixo `manual_ms_*` ou `febrasgo_*` (reais ou sinteticos)
e faz upsert no vector store. Idempotente: por padrao aborta cedo se o
indice ja existe no disco, evitando re-embedar tudo (bge-m3 leva horas
em CPU). Use `--force` pra reindexar do zero.

Uso:
    python scripts/build_rag_index.py            # no-op se ja existe
    python scripts/build_rag_index.py --force    # reindexa
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.ingestion import ingest_document  # noqa: E402
from src.rag.vector_store import VectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_rag_index")

INPUT_DIR = Path("data/raw")
PATTERNS = ("manual_ms_*.pdf", "febrasgo_*.pdf", "*.pdf")
MIN_SQLITE_BYTES = 64 * 1024


def _coletar_documentos() -> list[Path]:
    """Lista PDFs em `data/raw/` priorizando prefixos esperados."""
    encontrados: list[Path] = []
    for pattern in PATTERNS:
        for p in INPUT_DIR.glob(pattern):
            if p not in encontrados:
                encontrados.append(p)
    return encontrados


def _indice_ja_persistido(persist_dir: Path) -> bool:
    """Detecta se o indice Chroma ja esta materializado no disco.

    Checa o `chroma.sqlite3` com tamanho minimo e ao menos um subdiretorio
    HNSW gerado. Esses dois sinais juntos descartam o caso de diretorio
    recem-criado vazio e o caso de pointer LFS nao baixado.
    """
    sqlite = persist_dir / "chroma.sqlite3"
    if not sqlite.exists() or sqlite.stat().st_size < MIN_SQLITE_BYTES:
        return False
    return any(p.is_dir() for p in persist_dir.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="reindexa do zero mesmo que ja exista indice persistido",
    )
    args = parser.parse_args()

    store = VectorStore()

    if not args.force and _indice_ja_persistido(store.persist_dir):
        logger.info(
            "Indice ja existe em %s. Nada a fazer (use --force pra reindexar).",
            store.persist_dir,
        )
        return

    docs = _coletar_documentos()
    if not docs:
        logger.warning(
            "Nenhum PDF em %s. Rode scripts/gen_synthetic_pdfs.py para gerar exemplos.",
            INPUT_DIR,
        )
        sys.exit(1)

    store.load()
    if not store._available:
        logger.error("VectorStore nao disponivel. Verifique chromadb e sentence-transformers.")
        sys.exit(2)

    total_chunks = 0
    for doc in docs:
        logger.info("Processando %s", doc.name)
        chunks = ingest_document(doc)
        total_chunks += store.add(chunks)
    logger.info(
        "Indice atualizado: %d chunks (total na colecao: %d)",
        total_chunks,
        store.count(),
    )


if __name__ == "__main__":
    main()
