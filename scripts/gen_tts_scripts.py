"""Sintetiza 4 audios PT-BR via Azure Speech TTS para os casos demo.

Usa a voz neural `pt-BR-FranciscaNeural` com SSML por cenario para variar
prosodia (rate, pitch, style). Cada audio e salvo como
`data/examples/<caso>/audio.wav` em formato PCM 16 kHz mono.

Idempotente: pula casos cujo WAV ja existe. Use `--force` para regerar.

Credenciais: `settings.azure_speech_key` e `settings.azure_speech_region`.
Sem chave, o script falha cedo com mensagem clara (nao gera nada silencioso).

Se a sintese de UM caso falhar (ex: RAI block, estilo invalido na regiao),
o script reporta e continua os demais.

Uso:
    python scripts/gen_tts_scripts.py [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gen_tts_scripts")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"
VOICE = "pt-BR-FranciscaNeural"

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

# Cada caso define o SSML que sera enviado ao TTS. Estilos da voz Francisca
# variam por regiao; mantemos um conjunto pequeno que costuma estar liberado
# em brazilsouth. Se algum estilo for negado pelo RAI, basta remover o
# atributo `style` do SSML correspondente.
# SSML por caso de consulta (fallback se nao usar audio extraido do
# video real). Cirurgias nao tem voz da paciente, entao nao tem entry
# aqui (sao geradas apenas com video.mp4 sem audio).
CASES: dict[str, dict[str, str]] = {
    "consulta_clinica_geral": {
        "style": "calm",
        "rate": "0%",
        "pitch": "0%",
        "text": (
            "Doutora, comecei a sentir uma dor no peito ha algumas horas, "
            "junto com mal estar. Tambem percebi a pressao alterada em "
            "casa. Vim aqui pra entender o que esta acontecendo."
        ),
    },
    "rastreio_mama": {
        "style": "sad",
        "rate": "-10%",
        "pitch": "-5%",
        "text": (
            "Eu nunca imaginei que iria passar por isso. Quando vi o "
            "resultado, foi um choque muito grande. Hoje eu sigo o "
            "tratamento e quero alertar outras mulheres."
        ),
    },
    "dermatologica": {
        "style": "empathetic",
        "rate": "-5%",
        "pitch": "0%",
        "text": (
            "Doutora, estou preocupada com essas manchas no rosto. Uso "
            "os cremes que voce passou mas elas nao saem. Tem outros "
            "exames que eu posso fazer?"
        ),
    },
}


def _montar_ssml(config: dict[str, str]) -> str:
    """Monta o SSML com expressao por estilo + prosodia (rate/pitch)."""
    style = config["style"]
    rate = config["rate"]
    pitch = config["pitch"]
    text = config["text"]
    return (
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="https://www.w3.org/2001/mstts" '
        'xml:lang="pt-BR">'
        f'<voice name="{VOICE}">'
        f'<mstts:express-as style="{style}" styledegree="2">'
        f'<prosody rate="{rate}" pitch="{pitch}">{text}</prosody>'
        '</mstts:express-as>'
        '</voice>'
        '</speak>'
    )


def _sintetizar(caso: str, config: dict[str, str], *, force: bool, key: str, region: str) -> bool:
    """Sintetiza um caso via Azure Speech TTS. Retorna True em sucesso."""
    import azure.cognitiveservices.speech as speechsdk

    destino = _pasta_caso(caso) / "audio.wav"
    if destino.exists() and not force:
        logger.info("[skip] %s ja existe (%d KB)",
                    destino.relative_to(PROJECT_ROOT),
                    destino.stat().st_size // 1024)
        return True

    destino.parent.mkdir(parents=True, exist_ok=True)
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )
    audio_config = speechsdk.audio.AudioOutputConfig(filename=str(destino))
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    ssml = _montar_ssml(config)
    resultado = synthesizer.speak_ssml_async(ssml).get()

    if resultado.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        tamanho_kb = destino.stat().st_size // 1024
        logger.info("[OK] %s/audio.wav (%d KB, voz=%s, estilo=%s)",
                    caso, tamanho_kb, VOICE, config["style"])
        return True

    if resultado.reason == speechsdk.ResultReason.Canceled:
        details = resultado.cancellation_details
        logger.error("[fail] %s: cancelado (%s) %s", caso, details.reason, details.error_details)
    else:
        logger.error("[fail] %s: reason=%s", caso, resultado.reason)
    # Remove arquivo parcial se o Azure escreveu algo invalido.
    if destino.exists() and destino.stat().st_size == 0:
        destino.unlink()
    return False


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Le os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Regenera WAV mesmo se ja existir.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Sintetiza os 4 audios."""
    args = _parse_args(argv)
    key = settings.azure_speech_key.get_secret_value()
    region = settings.azure_speech_region
    if not key or not region:
        logger.error(
            "Azure Speech nao configurado. Defina AZURE_SPEECH_KEY e "
            "AZURE_SPEECH_REGION em .env."
        )
        return 1

    sucesso = 0
    falha = 0
    for caso, config in CASES.items():
        try:
            ok = _sintetizar(caso, config, force=args.force, key=key, region=region)
        except Exception as exc:  # noqa: BLE001 - SDK lanca varias subclasses
            logger.error("[fail] %s: excecao %s", caso, exc)
            ok = False
        if ok:
            sucesso += 1
        else:
            falha += 1

    print(f"Audios sintetizados: {sucesso} OK, {falha} falha(s).")
    return 0 if falha == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
