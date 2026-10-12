"""Centraliza setup do projeto: modelos, PDFs, indice RAG e exemplos UI.

Roda todos os downloads upfront, idempotentemente: se ja esta em disco/cache,
pula. Cobre 4 grupos:

1. **Modelos**: YOLOv8n stub, Whisper small, wav2vec2 (emocao), bge-m3 (RAG).
   Todos os modelos sao baixados lazy pelos proprios pipelines na primeira
   chamada. O warmup so disparara o download com antecedencia para evitar
   pausa na primeira demo. Stub YOLO pode ser substituido pelo `.pt` custom
   treinado no notebook do Colab (ver `notebooks/train_yolo_colab.ipynb`).
2. **PDFs**: 8 diretrizes brasileiras oficiais pro RAG (~18 MB total).
3. **Index RAG**: reconstroi Chroma a partir dos PDFs.
4. **Exemplos UI**: 4 casos reais pre-carregados pro Gradio. Os MP4s ja
   vem versionados no repo em `data/examples/`; este passo so garante que
   audios (Azure Speech TTS PT-BR) e contextos (GPT-4.1-mini) estejam
   gerados. Orquestrado por `scripts/seed_real_examples.py`.

O dataset CholecSeg8k (~3 GB) nao e baixado localmente: o treino do YOLO
acontece exclusivamente no notebook do Colab, que baixa direto do HF.

Uso:
    python scripts/warmup.py                  # tudo (~3 GB modelos + 18 MB PDFs)
    python scripts/warmup.py --models         # so modelos
    python scripts/warmup.py --pdfs           # so PDFs
    python scripts/warmup.py --examples       # so exemplos da UI
    python scripts/warmup.py --skip-rag       # pula bge-m3 + index (compat)
    python scripts/warmup.py --force          # re-baixa tudo, ignora cache
    python scripts/warmup.py --quiet          # reduz logging

Tamanhos aproximados:
    YOLOv8n:       6 MB        wav2vec2:     360 MB
    Whisper small: 500 MB      bge-m3:       2 GB
    8 PDFs:        ~18 MB
    Total:         ~3 GB
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.emotion import VocalEmotionClassifier  # noqa: E402
from src.audio.transcriber import WhisperTranscriber  # noqa: E402
from src.config.settings import settings  # noqa: E402
from src.rag.vector_store import VectorStore  # noqa: E402
from src.video.detector import ensure_yolo_weights  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("warmup")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CHROMA_DIR = DATA_PROCESSED / "chroma"
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"
# Casos reais gerados por seed_real_examples.py. Cada caso tem video.mp4 e
# context.txt; consultas tambem tem audio.wav extraido do video.
# Mapeia caso -> categoria (subpasta em data/examples/).
EXAMPLE_CASES: dict[str, str] = {
    "consulta_clinica_geral": "consultas",
    "rastreio_mama":          "consultas",
    "dermatologica":          "consultas",
    "prenatal_acolhimento":   "consultas",
    "rotina":                 "cirurgias",
    "sangramento":            "cirurgias",
}


# ---------------------------------------------------------------------
# Step dataclass
# ---------------------------------------------------------------------


@dataclass
class Step:
    """Passo de warmup: verifica se ja foi feito; se nao, executa."""

    titulo: str
    group: str
    size_label: str
    check: Callable[[], bool]
    download: Callable[[], None]
    cleanup_on_force: Callable[[], None] | None = None
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# Grupo 1: Modelos
# ---------------------------------------------------------------------


def check_yolo() -> bool:
    return settings.yolo_weights_absolute().exists()


def download_yolo() -> None:
    target = settings.yolo_weights_absolute()
    ensure_yolo_weights(target)


def check_whisper() -> bool:
    """Heuristica: tenta carregar; se _available=True, esta em cache."""
    try:
        transcriber = WhisperTranscriber()
        transcriber.load()
        return transcriber._available  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return False


def download_whisper() -> None:
    transcriber = WhisperTranscriber()
    transcriber.load()
    if not transcriber._available:  # noqa: SLF001
        raise RuntimeError("Whisper falhou ao carregar")


def check_wav2vec2() -> bool:
    try:
        clf = VocalEmotionClassifier()
        clf.load()
        return clf._available  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return False


def download_wav2vec2() -> None:
    clf = VocalEmotionClassifier()
    clf.load()
    if not clf._available:  # noqa: SLF001
        raise RuntimeError("wav2vec2 falhou ao carregar")


def check_bge_m3() -> bool:
    try:
        store = VectorStore()
        store.load()
        return store._available  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return False


def download_bge_m3() -> None:
    store = VectorStore()
    store.load()
    if not store._available:  # noqa: SLF001
        raise RuntimeError("VectorStore (bge-m3 + Chroma) falhou ao carregar")


# ---------------------------------------------------------------------
# Grupo 2: PDFs do RAG
# ---------------------------------------------------------------------

PDFS: dict[str, dict[str, str]] = {
    "manual_ms_prenatal.pdf": {
        "url": "https://bvsms.saude.gov.br/bvs/publicacoes/cadernos_atencao_basica_32_prenatal.pdf",
        "size": "~20 MB",
        "label": "MS - Caderno Atencao Basica 32 (Pre-natal Baixo Risco)",
    },
    "febrasgo_preeclampsia.pdf": {
        "url": "https://www.febrasgo.org.br/images/pec/Protocolos-assistenciais/2020-Pr-Eclmpsia.pdf",
        "size": "~2.5 MB",
        "label": "FEBRASGO - Pre-Eclampsia 2020",
    },
    "ms_gestacao_alto_risco.pdf": {
        "url": "https://bvsms.saude.gov.br/bvs/publicacoes/manual_gestacao_alto_risco.pdf",
        "size": "~4.8 MB",
        "label": "MS - Manual Gestacao Alto Risco 2022",
    },
    "inca_cancer_mama.pdf": {
        "url": "https://www.inca.gov.br/sites/ufu.sti.inca.local/files/media/document/diretrizes_deteccao_precoce_cancer_mama_brasil.pdf",
        "size": "~2.5 MB",
        "label": "INCA - Deteccao Precoce Cancer de Mama",
    },
    "inca_cancer_colo_utero.pdf": {
        "url": "https://www.inca.gov.br/sites/ufu.sti.inca.local/files/media/document/diretrizesparaorastreamentodocancerdocolodoutero_2016_corrigido.pdf",
        "size": "~2.8 MB",
        "label": "INCA - Rastreamento Cancer do Colo do Utero",
    },
    "ms_pcdt_ist_violencia.pdf": {
        "url": "https://bvsms.saude.gov.br/bvs/publicacoes/protocolo_clinico_diretrizes_terapeutica_atencao_integral_pessoas_infeccoes_sexualmente_transmissiveis.pdf",
        "size": "~4.2 MB",
        "label": "MS - PCDT IST + Violencia Sexual 2022",
    },
    "ms_parto_normal.pdf": {
        "url": "https://bvsms.saude.gov.br/bvs/publicacoes/diretrizes_nacionais_assistencia_parto_normal.pdf",
        "size": "~0.6 MB",
        "label": "MS - Diretrizes Nacionais Assistencia ao Parto Normal",
    },
    "cab26_saude_sexual_reprodutiva.pdf": {
        "url": "https://bvsms.saude.gov.br/bvs/publicacoes/saude_sexual_saude_reprodutiva.pdf",
        "size": "~3.2 MB",
        "label": "MS - Caderno Atencao Basica 26 (Saude Sexual e Reprodutiva)",
    },
    "cab34_saude_mental.pdf": {
        "url": "https://bvsms.saude.gov.br/bvs/publicacoes/cadernos_atencao_basica_34_saude_mental.pdf",
        "size": "~8.2 MB",
        "label": "MS - Caderno Atencao Basica 34 (Saude Mental)",
    },
}


def make_pdf_check(filename: str) -> Callable[[], bool]:
    def _check() -> bool:
        return (DATA_RAW / filename).exists() and (DATA_RAW / filename).stat().st_size > 1024

    return _check


def make_pdf_download(filename: str, url: str) -> Callable[[], None]:
    def _download() -> None:
        target = DATA_RAW / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target)
        if target.stat().st_size < 1024:
            raise RuntimeError(f"PDF baixado mas tamanho suspeito (<1KB): {filename}")

    return _download


# ---------------------------------------------------------------------
# Grupo 4: Index RAG
# ---------------------------------------------------------------------


def check_rag_index() -> bool:
    """Verifica se Chroma ja foi populado (existe pelo menos 1 sqlite)."""
    if not CHROMA_DIR.exists():
        return False
    return any(CHROMA_DIR.rglob("*.sqlite*"))


def rebuild_rag_index() -> None:
    """Reconstroi o indice Chroma rodando scripts/build_rag_index.py."""
    script = PROJECT_ROOT / "scripts" / "build_rag_index.py"
    if not script.exists():
        raise RuntimeError(f"Script nao encontrado: {script}")
    subprocess.run([sys.executable, str(script)], check=True)


def cleanup_rag_index() -> None:
    """Limpa Chroma antes de reconstruir (modo --force)."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)


# ---------------------------------------------------------------------
# Grupo 5: Exemplos da UI
# ---------------------------------------------------------------------


def check_real_examples() -> bool:
    """Verifica que o manifest existe e cada caso tem audio.wav + context.txt."""
    manifest = EXAMPLES_DIR / "manifest.json"
    if not manifest.exists():
        return False
    for caso, categoria in EXAMPLE_CASES.items():
        pasta = EXAMPLES_DIR / categoria / caso
        # Casos de cirurgia nao tem audio.wav (sem voz do paciente)
        if categoria == "consultas" and not (pasta / "audio.wav").exists():
            return False
        if not (pasta / "video.mp4").exists():
            return False
        ctx = pasta / "context.txt"
        if not ctx.exists() or ctx.stat().st_size == 0:
            return False
    return True


def generate_real_examples() -> None:
    """Gera 4 casos reais via scripts/seed_real_examples.py.

    Orquestra os 3 sub-scripts: video.mp4 (ffmpeg sobre CholecSeg8k),
    audio.wav (Azure Speech TTS PT-BR) e context.txt (GPT-4.1-mini).
    Requer ffmpeg no PATH e as chaves AZURE_SPEECH_* / AZURE_OPENAI_*
    no .env.
    """
    script = PROJECT_ROOT / "scripts" / "seed_real_examples.py"
    if not script.exists():
        raise RuntimeError(f"Script nao encontrado: {script}")
    subprocess.run([sys.executable, str(script)], check=True)


# ---------------------------------------------------------------------
# Montagem dos steps
# ---------------------------------------------------------------------


def build_steps() -> list[Step]:
    steps: list[Step] = []

    # Grupo 1: Modelos
    steps.append(
        Step(
            titulo="YOLOv8n stub (Ultralytics)",
            group="modelos",
            size_label="~6 MB",
            check=check_yolo,
            download=download_yolo,
        )
    )
    steps.append(
        Step(
            titulo="Whisper small (transcricao)",
            group="modelos",
            size_label="~500 MB",
            check=check_whisper,
            download=download_whisper,
        )
    )
    steps.append(
        Step(
            titulo="wav2vec2 (emocao vocal)",
            group="modelos",
            size_label="~360 MB",
            check=check_wav2vec2,
            download=download_wav2vec2,
        )
    )
    steps.append(
        Step(
            titulo="bge-m3 embeddings (RAG)",
            group="modelos",
            size_label="~2 GB",
            check=check_bge_m3,
            download=download_bge_m3,
            tags=["rag"],
        )
    )

    # Grupo 2: PDFs
    for filename, info in PDFS.items():
        steps.append(
            Step(
                titulo=info["label"],
                group="pdfs",
                size_label=info["size"],
                check=make_pdf_check(filename),
                download=make_pdf_download(filename, info["url"]),
            )
        )

    # Grupo 3: Index RAG
    steps.append(
        Step(
            titulo="Index Chroma (8 PDFs)",
            group="rag",
            size_label="processamento",
            check=check_rag_index,
            download=rebuild_rag_index,
            cleanup_on_force=cleanup_rag_index,
            tags=["rag"],
        )
    )

    # Grupo 4: Exemplos da UI
    steps.append(
        Step(
            titulo="Casos reais pre-carregados (4 cenarios: videos + TTS + contextos)",
            group="examples",
            size_label="~4.5 MB",
            check=check_real_examples,
            download=generate_real_examples,
        )
    )

    return steps


# ---------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------


def run_step(step: Step, force: bool) -> tuple[bool, float, bool]:
    """Executa um step. Retorna (sucesso, tempo, foi_pulado)."""
    print(f"\n==> [{step.group}] {step.titulo} ({step.size_label})")
    inicio = time.perf_counter()

    if not force and step.check():
        elapsed = time.perf_counter() - inicio
        print(f"    [pulado, ja existe] ({elapsed:.1f}s)")
        return True, elapsed, True

    if force and step.cleanup_on_force is not None:
        try:
            step.cleanup_on_force()
        except Exception as exc:  # noqa: BLE001
            print(f"    [aviso cleanup: {exc}]")

    try:
        step.download()
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - inicio
        print(f"    [FALHA apos {elapsed:.1f}s] {exc}")
        return False, elapsed, False

    elapsed = time.perf_counter() - inicio
    print(f"    [ok em {elapsed:.1f}s]")
    return True, elapsed, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", action="store_true", help="So roda modelos.")
    parser.add_argument("--pdfs", action="store_true", help="So roda PDFs.")
    parser.add_argument("--examples", action="store_true", help="So roda exemplos UI.")
    parser.add_argument("--skip-rag", action="store_true", help="Pula bge-m3 + index Chroma.")
    parser.add_argument("--force", action="store_true", help="Re-baixa tudo, ignora cache.")
    parser.add_argument("--quiet", action="store_true", help="Reduz logging dos modelos.")
    args = parser.parse_args(argv)

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    print("medica-ia-multimodal :: warmup")
    print("=" * 64)

    all_steps = build_steps()

    # Filtros de grupo
    filters_used = any([args.models, args.pdfs, args.examples])
    if filters_used:
        groups_to_run = set()
        if args.models:
            groups_to_run.add("modelos")
        if args.pdfs:
            groups_to_run.add("pdfs")
        if args.examples:
            groups_to_run.add("examples")
        steps = [s for s in all_steps if s.group in groups_to_run]
    else:
        steps = list(all_steps)

    # Skip-rag
    if args.skip_rag:
        steps = [s for s in steps if "rag" not in s.tags and s.group != "rag"]
        print("(--skip-rag: bge-m3 e index Chroma serao pulados)")

    if not steps:
        print("\nNenhum step selecionado.")
        return 0

    inicio_total = time.perf_counter()
    resultados: list[tuple[Step, bool, float, bool]] = []
    for step in steps:
        ok, elapsed, skipped = run_step(step, args.force)
        resultados.append((step, ok, elapsed, skipped))

    total = time.perf_counter() - inicio_total
    print("\n" + "=" * 64)
    print(f"Resumo ({total:.1f}s total):")

    by_group: dict[str, list[tuple[Step, bool, float, bool]]] = {}
    for r in resultados:
        by_group.setdefault(r[0].group, []).append(r)

    for group_name, group_results in by_group.items():
        print(f"\n  [{group_name}]")
        for step, ok, elapsed, skipped in group_results:
            status = "pulado" if skipped else ("ok" if ok else "FALHA")
            print(f"    [{status}] {step.titulo} ({elapsed:.1f}s)")

    falhas = sum(1 for _, ok, _, _ in resultados if not ok)
    if falhas:
        print(f"\n{falhas} passo(s) falharam. O app ainda roda com os fallbacks disponiveis.")
        return 1
    print("\nSetup completo. Pode rodar `python app.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
