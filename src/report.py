"""Gerador de relatorio clinico em Markdown.

O relatorio combina:
- nivel de risco e triggers do `AnomalyResult`
- transcricao e sentimento do `AudioAnalysis`
- numero de eventos do pipeline de video
- diretrizes recuperadas (RAG) como contexto clinico

Quando o `AzureOpenAIClient` esta configurado, o LLM produz um texto
naturalizado em PT-BR a partir de um prompt estruturado. Sem credenciais,
um fallback deterministico monta um markdown com as mesmas secoes,
preservando a auditabilidade e mantendo o pipeline operacional.
"""

from __future__ import annotations

import logging

from src.anomaly.types import AnomalyResult
from src.audio.types import AudioAnalysis
from src.llm.azure_openai import AzureOpenAIClient
from src.rag.types import Chunk
from src.video.scene_classifier import SceneType
from src.video.types import VideoEvent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_PT_BR: str = (
    "Voce e um assistente clinico especializado em saude integral da mulher "
    "(ginecologia, obstetricia, rastreio oncologico, saude mental, situacoes "
    "de violencia e atencao primaria) que produz relatorios estruturados em "
    "PT-BR a partir de analise multimodal (video, audio, contexto textual e "
    "diretrizes clinicas indexadas).\n\n"
    "Diretrizes obrigatorias:\n"
    "1. Use linguagem objetiva e tecnica. Nao invente dados que nao estejam "
    "explicitamente no contexto fornecido.\n"
    "2. Cite todos os triggers detectados; nao omita nenhum.\n"
    "3. Identifique o cenario provavel a partir do contexto: consulta de "
    "rotina, triagem (rastreio oncologico, violencia, etc.), emergencia "
    "obstetrica ou cirurgia ginecologica. Adapte o tom do relatorio. "
    "Quando o caso NAO indicar gestacao, puerperio ou pos-parto, NAO "
    "mencione 'saude mental perinatal' nem encaminhamentos especificos pra "
    "esse subtema: use 'avaliacao em saude mental na atencao primaria' ou "
    "'encaminhamento para psicologia/psiquiatria' conforme o contexto.\n"
    "4. Avalie inconsistencias entre modalidades APENAS quando elas existirem "
    "de fato. Quando o paciente verbaliza estar bem mas o tom vocal e monotono, "
    "a expressao facial indica distress, ou as features acusticas mostram "
    "baixa energia, isso pode indicar quadros sutis como ansiedade encoberta, "
    "minimizacao de sintomas ou (em contextos gestacionais) depressao "
    "perinatal; destaque essa inconsistencia como achado clinico. Quando as "
    "modalidades CONVERGEM (ex.: paciente verbaliza angustia e a voz tambem "
    "demonstra angustia), nao invente inconsistencia: declare a coerencia "
    "entre os sinais como um achado que reforca a interpretacao clinica.\n"
    "5. Cite APENAS diretrizes que aparecem efetivamente no contexto RAG "
    "fornecido. NUNCA invente nomes de protocolos, manuais, diretrizes ou "
    "documentos que nao tenham trecho concreto no contexto — mesmo em casos "
    "criticos onde pareca natural mencionar 'protocolo X'. Quando houver "
    "diretrizes no contexto, cite a fonte (nome do documento) e incorpore "
    "as recomendacoes aplicaveis. Quando o contexto RAG vier vazio ou nenhum "
    "chunk for relevante, informe explicitamente: 'Nenhuma das diretrizes "
    "indexadas aborda diretamente este tema; recomenda-se seguir protocolos "
    "institucionais e padrao da especialidade para avaliacao clinica direta.' "
    "Se o caso NAO for gestacional/obstetrico/puerperal, NAO cite diretrizes "
    "de pre-natal (manual_ms_prenatal), gestacao de alto risco "
    "(ms_gestacao_alto_risco) ou parto (ms_parto_normal) como aplicaveis — "
    "esses documentos podem aparecer nos chunks recuperados mas sao "
    "irrelevantes pra pacientes nao gestantes. Cite apenas se a evidencia for "
    "genuinamente transferivel (ex.: trecho sobre saude mental geral que "
    "aparece num manual obstetrico).\n"
    "6. Limitacao conhecida: classificadores de emocao vocal e facial podem "
    "ter vies em vozes/rostos fora do dominio de treino. Pondere o sinal "
    "junto com outros pilares antes de afirmar um afeto patologico.\n"
    "7. Quando o campo 'Tipo de cena identificada' estiver presente no contexto, "
    "use-o para adaptar o tom do relatorio: 'consulta' indica orientacao clinica "
    "ambulatorial; 'cirurgia' indica relato cirurgico tecnico; 'misto' indica "
    "cenario hibrido (ex: cirurgiao em consulta pos-operatoria); 'desconhecido' "
    "indica que a classificacao nao foi possivel e o tom deve ser neutro."
)


def generate_report(
    *,
    anomaly: AnomalyResult,
    audio_analysis: AudioAnalysis | None = None,
    video_events: list[VideoEvent] | None = None,
    rag_chunks: list[Chunk] | None = None,
    patient_metadata: dict | None = None,
    scene_type: SceneType | None = None,
    llm_client: AzureOpenAIClient | None = None,
) -> str:
    """Gera um relatorio clinico em markdown.

    Args:
        anomaly: resultado final da analise de anomalia.
        audio_analysis: saida do pipeline de audio (opcional).
        video_events: lista de eventos do pipeline de video (opcional).
        rag_chunks: trechos das diretrizes recuperados pelo RAG (opcional).
        patient_metadata: dict com metadata do paciente (opcional).
        scene_type: tipo de cena identificado pelo classificador de cena (opcional).
            Quando fornecido, e adicionado ao prompt e orienta o tom do relatorio.
        llm_client: cliente Azure OpenAI; quando ausente, sera instanciado.

    Returns:
        String em markdown pronta para exibicao na UI.
    """
    client = llm_client or AzureOpenAIClient()
    llm_text: str | None = None

    if client.is_configured:
        prompt = _build_user_prompt(
            anomaly=anomaly,
            audio_analysis=audio_analysis,
            video_events=video_events,
            rag_chunks=rag_chunks,
            patient_metadata=patient_metadata,
            scene_type=scene_type,
        )
        llm_text = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT_PT_BR},
                {"role": "user", "content": prompt},
            ]
        )
        if llm_text:
            logger.info("Relatorio gerado via LLM (%d chars).", len(llm_text))
            return llm_text
        logger.warning("LLM retornou vazio; usando fallback deterministico.")

    return _fallback_report(
        anomaly=anomaly,
        audio_analysis=audio_analysis,
        video_events=video_events,
        rag_chunks=rag_chunks,
        patient_metadata=patient_metadata,
    )


def _build_user_prompt(
    *,
    anomaly: AnomalyResult,
    audio_analysis: AudioAnalysis | None,
    video_events: list[VideoEvent] | None,
    rag_chunks: list[Chunk] | None,
    patient_metadata: dict | None,
    scene_type: SceneType | None = None,
) -> str:
    """Monta o prompt do usuario com todos os fatos estruturados."""
    lines: list[str] = []
    if patient_metadata:
        lines.append(f"Paciente: {patient_metadata}")
    if scene_type is not None:
        lines.append(f"Tipo de cena identificada: {scene_type.value}")
    lines.append(f"Nivel de risco final: {anomaly.level}")
    if anomaly.triggers:
        lines.append("Triggers detectados:")
        for trigger in anomaly.triggers:
            lines.append(
                f"- [{trigger.level}] ({trigger.source}/{trigger.rule_id}) {trigger.message}"
            )
    else:
        lines.append("Nenhum trigger clinico disparado.")

    if video_events is not None:
        lines.append(f"Frames analisados: {len(video_events)}.")

    if audio_analysis is not None:
        if audio_analysis.transcription:
            lines.append(f'Transcricao: "{audio_analysis.transcription.strip()}"')
        if audio_analysis.emotion is not None:
            lines.append(
                f"Emocao vocal predominante: {audio_analysis.emotion.label}"
                f" (confianca {audio_analysis.emotion.confidence:.2f})."
            )
        if audio_analysis.sentiment is not None:
            lines.append(
                f"Sentimento textual: {audio_analysis.sentiment.label}"
                f" (confianca {audio_analysis.sentiment.confidence:.2f})."
            )
        if audio_analysis.key_phrases:
            lines.append("Frases-chave: " + ", ".join(audio_analysis.key_phrases))

    if rag_chunks:
        lines.append("Diretrizes clinicas relevantes:")
        for chunk in rag_chunks[:5]:
            snippet = chunk.text.strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:277] + "..."
            lines.append(f"- ({chunk.source}) {snippet}")

    lines.append(
        "\nGere um relatorio em markdown com 4 secoes:\n\n"
        "## Resumo\n"
        "Sintese clinica em 2-3 frases, mencionando o nivel de risco e o "
        "cenario provavel (rotina, triagem, emergencia ou cirurgia).\n\n"
        "## Achados\n"
        "Liste e interprete os triggers detectados. Quando houver "
        "inconsistencia entre modalidades (texto vs voz vs face), destaque "
        "explicitamente como achado clinico relevante.\n\n"
        "## Diretrizes Aplicaveis\n"
        "Cite a fonte (documento) e o trecho aplicavel para cada chunk "
        "relevante. Se nenhum chunk for relevante, informe que o tema "
        "pode estar fora das diretrizes indexadas e sugira avaliacao "
        "clinica direta.\n\n"
        "## Recomendacoes\n"
        "Acoes clinicas concretas (monitoramento, encaminhamento, exames "
        "complementares, suporte psicologico). Ordenar por urgencia."
    )
    return "\n".join(lines)


def _fallback_report(
    *,
    anomaly: AnomalyResult,
    audio_analysis: AudioAnalysis | None,
    video_events: list[VideoEvent] | None,
    rag_chunks: list[Chunk] | None,
    patient_metadata: dict | None,
) -> str:
    """Monta um relatorio determinístico em markdown sem LLM."""
    parts: list[str] = ["# Relatorio Clinico Automatico", ""]
    parts.append(f"**Nivel de risco:** `{anomaly.level}`")
    if patient_metadata:
        parts.append(f"**Paciente:** {patient_metadata}")
    parts.append("")

    parts.append("## Resumo")
    parts.append(anomaly.explanation or "Sem observacoes adicionais.")
    parts.append("")

    parts.append("## Achados")
    if anomaly.triggers:
        for trigger in anomaly.triggers:
            parts.append(
                f"- **[{trigger.level}]** ({trigger.source}/{trigger.rule_id}) {trigger.message}"
            )
    else:
        parts.append("- Nenhum sinal clinico relevante detectado.")
    parts.append("")

    parts.append("## Modalidades Analisadas")
    if video_events is not None:
        parts.append(f"- Video: {len(video_events)} frames analisados.")
    if audio_analysis is not None:
        if audio_analysis.transcription:
            parts.append(f'- Audio: transcricao "{audio_analysis.transcription.strip()}"')
        if audio_analysis.emotion is not None:
            parts.append(
                f"- Emocao vocal: {audio_analysis.emotion.label}"
                f" (confianca {audio_analysis.emotion.confidence:.2f})"
            )
        if audio_analysis.sentiment is not None:
            parts.append(
                f"- Sentimento textual: {audio_analysis.sentiment.label}"
                f" (confianca {audio_analysis.sentiment.confidence:.2f})"
            )
    if video_events is None and audio_analysis is None:
        parts.append("- Nenhuma modalidade fornecida.")
    parts.append("")

    if rag_chunks:
        parts.append("## Diretrizes Aplicaveis")
        for chunk in rag_chunks[:5]:
            snippet = chunk.text.strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:277] + "..."
            parts.append(f"- _{chunk.source}_: {snippet}")
        parts.append("")

    parts.append("## Recomendacoes")
    if anomaly.recommended_actions:
        for action in anomaly.recommended_actions:
            parts.append(f"- {action}")
    else:
        parts.append("- Manter monitoramento padrao.")

    return "\n".join(parts).rstrip() + "\n"
