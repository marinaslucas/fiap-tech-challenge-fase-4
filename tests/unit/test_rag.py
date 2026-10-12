"""Testes smoke do pipeline RAG.

Mocks de chromadb e sentence-transformers evitam download de modelos
em CI. Os testes de chunking/ingestion usam arquivos `.txt` para
manter o teste rapido (sem PDF parsing).
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.rag import (
    Chunk,
    RetrievalResult,
    Retriever,
    VectorStore,
    chunk_text,
    ingest_document,
)

# ---------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_chunk_metadata_omite_campos_none():
    chunk = Chunk(text="ola", source="manual", chunk_id="manual-p1-c0")
    meta = chunk.metadata()
    assert meta == {"source": "manual"}


@pytest.mark.smoke
def test_chunk_metadata_inclui_section_e_page():
    chunk = Chunk(
        text="ola",
        source="manual",
        section="cap1",
        page=3,
        chunk_id="manual-p3-c0",
    )
    assert chunk.metadata() == {"source": "manual", "section": "cap1", "page": 3}


@pytest.mark.smoke
def test_retrieval_result_default_vazio():
    result = RetrievalResult()
    assert result.chunks == []
    assert result.scores == []


@pytest.mark.smoke
def test_chunk_exige_id_estavel():
    with pytest.raises(ValidationError):
        Chunk(text="ola", source="x")  # falta chunk_id


# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_chunk_text_devolve_lista_vazia_para_string_vazia():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


@pytest.mark.smoke
def test_chunk_text_mantem_paragrafo_curto_inteiro():
    texto = "Paragrafo unico curto."
    chunks = chunk_text(texto, chunk_size=800, overlap=50)
    assert chunks == [texto]


@pytest.mark.smoke
def test_chunk_text_quebra_quando_excede_tamanho():
    p1 = "A" * 400
    p2 = "B" * 400
    p3 = "C" * 400
    texto = "\n\n".join([p1, p2, p3])
    chunks = chunk_text(texto, chunk_size=600, overlap=50)
    # Deve gerar mais de 1 chunk porque 3 paragrafos de 400 cada nao cabem em 600.
    assert len(chunks) >= 2


# ---------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_ingest_document_falha_para_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_document(tmp_path / "nao_existe.pdf")


@pytest.mark.smoke
def test_ingest_document_falha_para_extensao_nao_suportada(tmp_path):
    arquivo = tmp_path / "x.docx"
    arquivo.write_bytes(b"x")
    with pytest.raises(ValueError, match="Extensao"):
        ingest_document(arquivo)


@pytest.mark.smoke
def test_ingest_document_aceita_txt_e_gera_chunks(tmp_path):
    arquivo = tmp_path / "guia.txt"
    arquivo.write_text(
        "Pre-natal: realizar 6 consultas no minimo.\n\n"
        "Sinais de alerta: sangramento, cefaleia intensa, edema progressivo.",
        encoding="utf-8",
    )
    chunks = ingest_document(arquivo, chunk_size=200)
    assert len(chunks) >= 1
    assert chunks[0].source == "guia"
    assert chunks[0].chunk_id.startswith("guia-p1-c")


# ---------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_vector_store_inicializa_lazy(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    assert store._collection is None
    assert store._load_attempted is False


@pytest.mark.smoke
def test_vector_store_add_retorna_zero_quando_indisponivel(tmp_path, monkeypatch):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    monkeypatch.setattr(store, "load", lambda: None)
    store._load_attempted = True
    store._available = False
    chunks = [Chunk(text="x", source="s", chunk_id="s-p1-c0")]
    assert store.add(chunks) == 0


@pytest.mark.smoke
def test_vector_store_query_retorna_vazio_quando_indisponivel(tmp_path, monkeypatch):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    monkeypatch.setattr(store, "load", lambda: None)
    store._load_attempted = True
    store._available = False
    result = store.query("query qualquer")
    assert result.chunks == []
    assert result.scores == []


@pytest.mark.smoke
def test_vector_store_query_chama_chroma_e_parseia(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [["doc-1", "doc-2"]],
        "metadatas": [[{"source": "manual", "page": 1}, {"source": "febrasgo", "page": 3}]],
        "ids": [["manual-p1-c0", "febrasgo-p3-c0"]],
        "distances": [[0.1, 0.4]],
    }
    store._collection = fake_collection
    store._available = True
    store._load_attempted = True
    result = store.query("sangramento", top_k=2)
    assert len(result.chunks) == 2
    assert result.chunks[0].source == "manual"
    assert result.chunks[0].page == 1
    assert result.scores[0] == pytest.approx(0.9)


# ---------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_retriever_propaga_query_sem_filtros():
    fake_store = MagicMock(spec=VectorStore)
    fake_store.query.return_value = RetrievalResult()
    retriever = Retriever(store=fake_store)
    retriever.search("teste", top_k=5)
    fake_store.query.assert_called_once_with(text="teste", top_k=5, where=None)


@pytest.mark.smoke
def test_retriever_envia_filtro_por_source():
    fake_store = MagicMock(spec=VectorStore)
    fake_store.query.return_value = RetrievalResult()
    retriever = Retriever(store=fake_store)
    retriever.search("teste", source="febrasgo")
    fake_store.query.assert_called_once()
    _, kwargs = fake_store.query.call_args
    assert kwargs["where"] == {"source": "febrasgo"}


@pytest.mark.smoke
def test_retriever_combina_source_e_section_com_and():
    fake_store = MagicMock(spec=VectorStore)
    fake_store.query.return_value = RetrievalResult()
    retriever = Retriever(store=fake_store)
    retriever.search("teste", source="manual", section="cap1")
    _, kwargs = fake_store.query.call_args
    assert kwargs["where"] == {"$and": [{"source": "manual"}, {"section": "cap1"}]}


@pytest.mark.smoke
def test_retriever_multi_search_deduplica_chunks_iguais_em_queries_diferentes():
    """Mesmo chunk retornado em 2 queries deve aparecer 1 vez no resultado final."""
    fake_store = MagicMock(spec=VectorStore)
    chunk_a = Chunk(text="postura", source="manual", chunk_id="manual::p1::c0")
    chunk_b = Chunk(text="ansiedade", source="cab34", chunk_id="cab34::p1::c0")
    # Query 1 retorna A e B; Query 2 retorna B e A. Dedupe por chunk_id.
    fake_store.query.side_effect = [
        RetrievalResult(chunks=[chunk_a, chunk_b], scores=[0.9, 0.7]),
        RetrievalResult(chunks=[chunk_b, chunk_a], scores=[0.85, 0.6]),
    ]
    retriever = Retriever(store=fake_store)
    result = retriever.multi_search(
        ["query clinica", "query mental"], top_k_per_query=2, total_k=4,
    )
    assert len(result.chunks) == 2
    ids = [c.chunk_id for c in result.chunks]
    assert "manual::p1::c0" in ids
    assert "cab34::p1::c0" in ids


@pytest.mark.smoke
def test_retriever_multi_search_ignora_queries_vazias():
    """Queries vazias ou whitespace devem ser ignoradas sem chamar o store."""
    fake_store = MagicMock(spec=VectorStore)
    fake_store.query.return_value = RetrievalResult(chunks=[], scores=[])
    retriever = Retriever(store=fake_store)
    result = retriever.multi_search(["", "  ", ""], top_k_per_query=3, total_k=6)
    assert result.chunks == []
    fake_store.query.assert_not_called()


@pytest.mark.smoke
def test_retriever_multi_search_interleaving_round_robin():
    """Round-robin: 1o chunk da query 1, 1o da query 2, 2o da query 1..."""
    fake_store = MagicMock(spec=VectorStore)
    chunks_q1 = [
        Chunk(text="a", source="s1", chunk_id="s1::p1::c0"),
        Chunk(text="b", source="s1", chunk_id="s1::p1::c1"),
    ]
    chunks_q2 = [
        Chunk(text="c", source="s2", chunk_id="s2::p1::c0"),
        Chunk(text="d", source="s2", chunk_id="s2::p1::c1"),
    ]
    fake_store.query.side_effect = [
        RetrievalResult(chunks=chunks_q1, scores=[0.9, 0.8]),
        RetrievalResult(chunks=chunks_q2, scores=[0.85, 0.75]),
    ]
    retriever = Retriever(store=fake_store)
    result = retriever.multi_search(["q1", "q2"], top_k_per_query=2, total_k=4)
    ids = [c.chunk_id for c in result.chunks]
    # Round-robin: q1[0], q2[0], q1[1], q2[1]
    assert ids == ["s1::p1::c0", "s2::p1::c0", "s1::p1::c1", "s2::p1::c1"]
