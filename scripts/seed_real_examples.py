"""Orquestra os scripts de seed e regenera o `manifest.json` real.

Executa em sequencia:
    1. `gen_tts_scripts.py`  (WAVs Azure Speech TTS)
    2. `gen_contexts.py`     (TXTs GPT-4.1-mini)

Os MP4s ja vem versionados em `data/examples/<caso>/video.mp4` (commitados
no repo). Em seguida sobrescreve `data/examples/manifest.json` com as 4
entradas dos casos demo (normal, moderado, critico cirurgia, critico
consulta) lendo o conteudo de cada `context.txt`. Casos sem `context.txt`
valido sao pulados do manifest (mas listados no stdout para investigacao).

Limpa pastas obsoletas do esquema antigo (`data/examples/caso_critico/`)
para evitar lixo.

Uso:
    python scripts/seed_real_examples.py [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("seed_real_examples")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Mapeia caso -> categoria (subpasta em data/examples/). Casos de
# consulta com paciente ficam em `consultas/`; videos cirurgicos em
# `cirurgias/`. Nomes descrevem o conteudo do video (nao a severidade
# esperada, que e atribuida pelo pipeline em tempo de execucao).
CATEGORIA_POR_CASO: dict[str, str] = {
    "consulta_clinica_geral": "consultas",
    "rastreio_mama":          "consultas",
    "dermatologica":          "consultas",
    "prenatal_acolhimento":   "consultas",
    "rotina":                 "cirurgias",
    "sangramento":            "cirurgias",
}

# Esquema do manifest. Paths sao relativos a raiz do projeto.
CASOS: list[dict] = [
    {
        "key": "consulta_clinica_geral",
        "nome": "Consulta clinica geral - queixa toracica",
        "descricao": (
            "Primeira consulta clinica de avaliacao geral, paciente "
            "refere dor toracica e alteracao de pressao arterial."
        ),
        "video_path": "data/examples/consultas/consulta_clinica_geral/video.mp4",
        "audio_path": "data/examples/consultas/consulta_clinica_geral/audio.wav",
        "patient_metadata": {"id": "exemplo-consulta-clinica-geral"},
        "nivel_esperado": "normal",
    },
    {
        "key": "rastreio_mama",
        "nome": "Rastreio de cancer de mama (depoimento)",
        "descricao": (
            "Depoimento real de paciente sobre experiencia com "
            "rastreio e diagnostico mamario. Audio-only: video "
            "apresenta agulhas/instrumentos de exame que geram "
            "falso positivo no YOLO."
        ),
        "video_path": None,
        "audio_path": "data/examples/consultas/rastreio_mama/audio.wav",
        "patient_metadata": {"id": "exemplo-rastreio-mama"},
        "nivel_esperado": "moderate",
    },
    {
        "key": "dermatologica",
        "nome": "Consulta dermatologica com ansiedade",
        "descricao": (
            "Consulta dermatologica com queixa de manchas faciais "
            "e componente emocional acentuado."
        ),
        "video_path": "data/examples/consultas/dermatologica/video.mp4",
        "audio_path": "data/examples/consultas/dermatologica/audio.wav",
        "patient_metadata": {"id": "exemplo-dermatologica"},
        "nivel_esperado": "moderate",
    },
    {
        "key": "prenatal_acolhimento",
        "nome": "Pre-natal - acolhimento emocional",
        "descricao": (
            "Primeira consulta gestacional, trecho de acolhimento "
            "ao resultado positivo. Paciente expressa ansiedade e "
            "duvidas sobre como contar ao parceiro."
        ),
        "video_path": "data/examples/consultas/prenatal_acolhimento/video.mp4",
        "audio_path": "data/examples/consultas/prenatal_acolhimento/audio.wav",
        "patient_metadata": {"id": "exemplo-prenatal-acolhimento"},
        "nivel_esperado": "moderate",
    },
    {
        "key": "rotina",
        "nome": "Cirurgia laparoscopica de rotina",
        "descricao": (
            "Procedimento laparoscopico em andamento sem "
            "intercorrencia visivel."
        ),
        "video_path": "data/examples/cirurgias/rotina/video.mp4",
        "audio_path": None,
        "patient_metadata": {"id": "exemplo-cirurgia-rotina"},
        "nivel_esperado": "normal",
    },
    {
        "key": "sangramento",
        "nome": "Sangramento intraoperatorio",
        "descricao": (
            "Procedimento laparoscopico com sangramento "
            "intraoperatorio em foco operatorio."
        ),
        "video_path": "data/examples/cirurgias/sangramento/video.mp4",
        "audio_path": None,
        "patient_metadata": {"id": "exemplo-sangramento"},
        "nivel_esperado": "critical",
    },
]

# Pastas do esquema antigo que devem ser removidas para nao confundir o app.
PASTAS_OBSOLETAS = [
    "caso_critico", "caso_normal", "caso_moderado",
    "caso_critico_cirurgia", "caso_critico_consulta",
    "consulta_normal", "consulta_moderado", "consulta_critica",
    "cirurgia_normal", "cirurgia_sangramento",
]


def _rodar(script: str, extra: list[str]) -> int:
    """Executa `python scripts/<script>` como subprocess e retorna o exit code."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *extra]
    logger.info("--> %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _pasta_caso(caso_key: str) -> Path:
    """Resolve a pasta absoluta do caso aplicando a categoria correta."""
    categoria = CATEGORIA_POR_CASO[caso_key]
    return EXAMPLES_DIR / categoria / caso_key


def _ler_contexto(caso_key: str) -> str | None:
    """Le `data/examples/<categoria>/<caso>/context.txt` se existir e nao for vazio."""
    path = _pasta_caso(caso_key) / "context.txt"
    if not path.exists():
        return None
    texto = path.read_text(encoding="utf-8").strip()
    return texto or None


def _limpar_obsoletas() -> None:
    """Remove pastas do esquema antigo que nao fazem parte do novo manifest."""
    for nome in PASTAS_OBSOLETAS:
        pasta = EXAMPLES_DIR / nome
        if pasta.exists():
            shutil.rmtree(pasta)
            logger.info("[clean] removida pasta obsoleta %s", pasta.relative_to(PROJECT_ROOT))


def _gerar_manifest() -> tuple[int, int]:
    """Monta `manifest.json` com os casos que tem `context.txt` valido."""
    casos_saida: list[dict] = []
    pulados = 0
    for caso in CASOS:
        key = caso["key"]
        texto = _ler_contexto(key)
        if texto is None:
            logger.warning("[skip-manifest] %s sem context.txt valido", key)
            pulados += 1
            continue
        entrada = {
            "nome": caso["nome"],
            "descricao": caso["descricao"],
            "video_path": caso["video_path"],
            "audio_path": caso["audio_path"],
            "context_text": texto,
            "patient_metadata": caso["patient_metadata"],
            "nivel_esperado": caso["nivel_esperado"],
        }
        casos_saida.append(entrada)

    manifest_path = EXAMPLES_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps({"casos": casos_saida}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Manifest com %d caso(s) escrito em %s",
                len(casos_saida), manifest_path.relative_to(PROJECT_ROOT))
    return len(casos_saida), pulados


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Le os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="Forca regerar todos os artefatos (passa --force aos sub-scripts).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Roda os 3 scripts, limpa obsoletos e regenera o manifest."""
    args = _parse_args(argv)
    extra = ["--force"] if args.force else []

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    rc_tts = _rodar("gen_tts_scripts.py", extra)
    rc_ctx = _rodar("gen_contexts.py", extra)

    _limpar_obsoletas()
    incluidos, pulados = _gerar_manifest()

    print(
        f"Seed real concluido: {incluidos} caso(s) no manifest, {pulados} pulado(s). "
        f"Exit codes: tts={rc_tts}, contextos={rc_ctx}."
    )
    if incluidos == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
