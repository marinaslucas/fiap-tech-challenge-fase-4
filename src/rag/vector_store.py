"""Vector store baseado em Chroma com embeddings via sentence-transformers.

Persiste o indice em `settings.rag_index_path`. Por padrao usa `BAAI/bge-m3`
(multilingue, ~2GB no primeiro download). O modelo e carregado lazy via
`load()` ou na primeira chamada de `add` / `query`. Em caso de falha de
import, marca-se indisponivel e devolve `RetrievalResult` vazio.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config.settings import settings
from src.rag.types import Chunk, RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDER = "BAAI/bge-m3"
DEFAULT_COLLECTION = "diretrizes_clinicas"


def _detect_device() -> str:
    """Detecta melhor device disponivel pra inferencia do embedder.

    Ordem de preferencia: cuda > mps > cpu. Cai pra cpu se torch nao
    estiver disponivel.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class VectorStore:
    """Wrapper persistente em torno do Chroma com embedder configuravel."""

    def __init__(
        self,
        persist_dir: Path | None = None,
        embedder_name: str = DEFAULT_EMBEDDER,
        collection_name: str = DEFAULT_COLLECTION,
        device: str | None = None,
    ) -> None:
        """Configura o store sem ainda carregar Chroma ou o embedder.

        Args:
            persist_dir: diretorio do indice Chroma (default `settings.rag_index_path`).
            embedder_name: modelo sentence-transformers (default `BAAI/bge-m3`).
            collection_name: nome da colecao Chroma.
            device: `cuda` / `mps` / `cpu`. Quando `None`, auto-detect
                (preferencia cuda > mps > cpu).
        """
        self.persist_dir: Path = (persist_dir or settings.rag_index_absolute()).resolve()
        self.embedder_name: str = embedder_name
        self.collection_name: str = collection_name
        self.device: str = device if device is not None else _detect_device()
        self._client = None
        self._collection = None
        self._available: bool = False
        self._load_attempted: bool = False

    def load(self) -> None:
        """Inicializa o cliente Chroma e a colecao (cria se nao existir)."""
        self._load_attempted = True
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedder_name,
                device=self.device,
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedder,
            )
            self._available = True
            logger.info(
                "VectorStore pronto: persist=%s, embedder=%s, device=%s, colecao=%s",
                self.persist_dir,
                self.embedder_name,
                self.device,
                self.collection_name,
            )
        except (ImportError, RuntimeError) as exc:
            logger.warning("VectorStore indisponivel (%s). Operacoes retornarao vazio.", exc)
            self._available = False

    def add(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Indexa chunks (upsert por `chunk_id`).

        Args:
            chunks: lista a inserir/atualizar.
            batch_size: quantos chunks enviar por upsert. Embedding de bge-m3
                requer memoria proporcional ao tamanho do batch; lotes maiores
                que 64 podem estourar GiB de RAM/VRAM em PDFs longos.

        Returns:
            Numero de chunks indexados (0 se store indisponivel ou lista vazia).
        """
        if not chunks:
            return 0
        if not self._load_attempted:
            self.load()
        if not self._available or self._collection is None:
            return 0
        total = len(chunks)
        for offset in range(0, total, batch_size):
            batch = chunks[offset : offset + batch_size]
            self._collection.upsert(
                documents=[c.text for c in batch],
                metadatas=[c.metadata() for c in batch],
                ids=[c.chunk_id for c in batch],
            )
        logger.info("Indexados %d chunks na colecao %s", total, self.collection_name)
        return total

    def query(
        self,
        text: str,
        top_k: int = 3,
        where: dict | None = None,
    ) -> RetrievalResult:
        """Busca chunks mais similares a `text`.

        Args:
            text: query em linguagem natural.
            top_k: quantos resultados retornar.
            where: filtro opcional sobre metadata (Chroma where clause).

        Returns:
            `RetrievalResult` com chunks ordenados por relevancia.
        """
        if not self._load_attempted:
            self.load()
        if not self._available or self._collection is None:
            return RetrievalResult()

        result = self._collection.query(
            query_texts=[text],
            n_results=top_k,
            where=where,
        )
        return _to_retrieval_result(result)

    def count(self) -> int:
        """Numero de itens indexados (0 se store indisponivel)."""
        if not self._load_attempted:
            self.load()
        if not self._available or self._collection is None:
            return 0
        return int(self._collection.count())


def _to_retrieval_result(raw: dict) -> RetrievalResult:
    """Converte saida bruta do Chroma em `RetrievalResult` tipado."""
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    ids = (raw.get("ids") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    chunks: list[Chunk] = []
    for doc, meta, chunk_id in zip(docs, metas, ids, strict=True):
        meta = meta or {}
        chunks.append(
            Chunk(
                text=doc,
                source=str(meta.get("source", "")),
                section=meta.get("section"),
                page=meta.get("page"),
                chunk_id=chunk_id,
            )
        )
    # Chroma retorna distancia (menor = mais similar). Convertemos para score
    # 1 - distancia para ficar "maior = melhor" e mais intuitivo no log.
    scores = [float(1.0 - d) for d in distances]
    return RetrievalResult(chunks=chunks, scores=scores)
