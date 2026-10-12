"""Demo do retriever RAG com 5 queries clinicas.

Roda 5 perguntas representativas e imprime os top-2 chunks recuperados
com seus scores. Util pra julgar qualidade da recuperacao apos
`build_rag_index.py`.

Uso:
    python scripts/demo_rag.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.retriever import Retriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("demo_rag")

QUERIES: list[str] = [
    "Quando suspeitar de pre-eclampsia em gestante com cefaleia?",
    "Quais sinais de alerta no pre-natal exigem encaminhamento imediato?",
    "Conduta no pronto atendimento de pre-eclampsia grave",
    "Suplementacao de acido folico durante a gestacao",
    "Sintomas de sindrome HELLP em gestante hipertensa",
]


def main() -> None:
    retriever = Retriever()
    for i, query in enumerate(QUERIES, start=1):
        result = retriever.search(query, top_k=2)
        print(f"\n[{i}/{len(QUERIES)}] Query: {query}")
        if not result.chunks:
            print("  (sem resultados)")
            continue
        for chunk, score in zip(result.chunks, result.scores, strict=True):
            preview = chunk.text.replace("\n", " ")[:140]
            print(f"  - score={score:.3f} source={chunk.source} page={chunk.page}")
            print(f"    {preview}...")


if __name__ == "__main__":
    main()
