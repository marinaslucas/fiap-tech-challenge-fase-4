"""Gera contextos clinicos PT-BR para os 4 casos demo via GPT-4.1-mini.

Usa `src.llm.azure_openai.AzureOpenAIClient` (endpoint Foundry v1).
Cada contexto sai com 5 a 8 linhas curtas cobrindo idade, semanas de
gestacao ou puerperio, queixa principal, sinais vitais e historico breve.

O nivel esperado (normal, moderate, critical) e injetado no prompt para
manter a coerencia clinica. O modelo recebe instrucao explicita para NAO
inventar nomes proprios nem datas absolutas.

Salva em `data/examples/<caso>/context.txt`. Idempotente: pula casos com
arquivo nao vazio; use `--force` para regerar.

Se a chamada ao LLM falhar (chave invalida, timeout, RAI block), o caso
afetado e pulado e os demais continuam.

Uso:
    python scripts/gen_contexts.py [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.azure_openai import AzureOpenAIClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gen_contexts")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"

# Mapeia caso -> categoria (subpasta em data/examples/). Casos de
# consulta com paciente ficam em `consultas/`; videos cirurgicos em
# `cirurgias/`.
CATEGORIA_POR_CASO: dict[str, str] = {
    "consulta_clinica_geral": "consultas",
    "rastreio_mama":          "consultas",
    "dermatologica":          "consultas",
    "prenatal_acolhimento":   "consultas",
    "rotina":                 "cirurgias",
    "sangramento":            "cirurgias",
}


def _pasta_caso(caso: str) -> Path:
    """Resolve a pasta absoluta do caso aplicando a categoria correta."""
    return EXAMPLES_DIR / CATEGORIA_POR_CASO[caso] / caso

# Cada caso define o perfil clinico que o modelo deve respeitar, alinhado
# ao conteudo real do video correspondente.
CASES: dict[str, dict[str, str]] = {
    "consulta_clinica_geral": {
        "nivel": "normal",
        "perfil": (
            "Primeira consulta clinica de avaliacao geral. Paciente adulta "
            "refere dor toracica iniciada ha algumas horas, com sensacao de "
            "mal estar e alteracao de pressao arterial em aferimento "
            "domiciliar. Anamnese inicial em curso, sinais vitais sendo "
            "coletados. Conversa fluente, postura geral tranquila apesar do "
            "desconforto."
        ),
    },
    "rastreio_mama": {
        "nivel": "moderate",
        "perfil": (
            "Paciente do sexo feminino relata em depoimento sua experiencia "
            "com rastreio e diagnostico de alteracao mamaria. Tom emocional "
            "presente. Importancia de retorno e acompanhamento medico."
        ),
    },
    "dermatologica": {
        "nivel": "moderate",
        "perfil": (
            "Consulta dermatologica de saude da mulher. Paciente com queixa "
            "de manchas faciais hiperpigmentadas, em tratamento topico, "
            "componente emocional de apreensao ao discutir as lesoes."
        ),
    },
    "prenatal_acolhimento": {
        "nivel": "moderate",
        "perfil": (
            "Primeira consulta gestacional em atencao basica, conduzida pela "
            "equipe de enfermagem. Trecho de acolhimento ao resultado positivo: "
            "gestacao nao planejada, paciente jovem expressa ansiedade situacional "
            "e duvidas sobre como comunicar a noticia ao parceiro e a familia. "
            "Enfermeiro acolhe, oferece suporte e introduz o fluxo de "
            "acompanhamento multiprofissional."
        ),
    },
    "rotina": {
        "nivel": "normal",
        "perfil": (
            "Procedimento laparoscopico ginecologico em andamento, sem "
            "intercorrencias visiveis. Visualizacao da cavidade abdominal "
            "e instrumentos cirurgicos em etapa rotineira."
        ),
    },
    "sangramento": {
        "nivel": "critical",
        "perfil": (
            "Procedimento laparoscopico ginecologico com sangramento "
            "intraoperatorio em foco operatorio. Necessario monitoramento "
            "continuo e provavel intervencao hemostatica imediata."
        ),
    },
}

SYSTEM_PROMPT = (
    "Voce e um redator clinico que escreve resumos de prontuario "
    "em portugues do Brasil para apresentacoes academicas. Use linguagem "
    "tecnica concisa. NUNCA invente nomes proprios, datas absolutas, hospitais "
    "ou identificadores. Nao use emoji. Nao use travessao (em-dash ou en-dash). "
    "Use ponto, virgula, dois-pontos ou ponto-e-virgula."
)

USER_TEMPLATE = (
    "Escreva um contexto clinico realista para o caso abaixo, em 5 a 8 linhas "
    "curtas (uma frase por linha). Cubra: idade, queixa principal, sinais "
    "vitais relevantes, historico breve e conduta esperada. "
    "Mantenha tom compativel com nivel de risco '{nivel}'. Nao inclua "
    "cabecalhos, listas ou markdown: apenas texto corrido em linhas separadas.\n\n"
    "Perfil base: {perfil}"
)


def _gerar_contexto(client: AzureOpenAIClient, caso: str, config: dict[str, str]) -> str | None:
    """Chama o LLM e retorna o texto, ou None em falha."""
    prompt = USER_TEMPLATE.format(nivel=config["nivel"], perfil=config["perfil"])
    resposta = client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=400,
    )
    if resposta is None or not resposta.strip():
        logger.error("[fail] %s: LLM retornou vazio", caso)
        return None
    return resposta.strip()


def _salvar(caso: str, texto: str) -> Path:
    """Persiste o contexto em `data/examples/<caso>/context.txt`."""
    destino = _pasta_caso(caso) / "context.txt"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto + "\n", encoding="utf-8")
    return destino


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Le os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="Regenera context.txt mesmo se ja existir.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Gera os 4 contextos clinicos."""
    args = _parse_args(argv)
    client = AzureOpenAIClient()
    if not client.is_configured:
        logger.error(
            "Azure OpenAI nao configurado. Defina AZURE_OPENAI_KEY, "
            "AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_DEPLOYMENT em .env."
        )
        return 1

    sucesso = 0
    falha = 0
    for caso, config in CASES.items():
        destino = _pasta_caso(caso) / "context.txt"
        if destino.exists() and destino.stat().st_size > 0 and not args.force:
            logger.info("[skip] %s ja existe (%d bytes)",
                        destino.relative_to(PROJECT_ROOT), destino.stat().st_size)
            sucesso += 1
            continue
        try:
            texto = _gerar_contexto(client, caso, config)
        except Exception as exc:  # noqa: BLE001 - SDK lanca varias subclasses
            logger.error("[fail] %s: excecao %s", caso, exc)
            texto = None
        if texto is None:
            falha += 1
            continue
        _salvar(caso, texto)
        tamanho_bytes = (_pasta_caso(caso) / "context.txt").stat().st_size
        logger.info("[OK] %s/context.txt (%d bytes)", caso, tamanho_bytes)
        sucesso += 1

    print(f"Contextos gerados: {sucesso} OK, {falha} falha(s).")
    return 0 if falha == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
