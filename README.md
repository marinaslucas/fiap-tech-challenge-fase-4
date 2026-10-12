---
title: Medica IA Multimodal
emoji: 👩‍⚕️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Análise multimodal aplicada à saúde da mulher
---

# medica-ia-multimodal

Sistema de monitoramento multimodal aplicado à saúde da mulher. Tech Challenge Fase 4 da pós-graduação Tech IADT.

Combina análise de vídeo (YOLOv8 detector + GPT-4o vision pra emoção e linguagem corporal), áudio (faster-whisper + librosa + wav2vec2), RAG sobre diretrizes clínicas brasileiras (Chroma + bge-m3) e LLM (Azure OpenAI) para detectar anomalias e gerar relatórios clínicos com alertas categorizados.

## O Que Faz

Recebe vídeo, áudio e/ou contexto textual sobre uma paciente. Processa cada modalidade em pipelines especializados, funde os resultados, consulta diretrizes clínicas via RAG e devolve:

- **Nível de risco** categorizado (normal, moderado, crítico)
- **Relatório clínico** em markdown com achados e recomendações
- **Alerta estruturado** quando moderado ou crítico
- **Auditoria** persistente em SQLite

UI Gradio com 4 abas: **Vídeo**, **Áudio**, **Multimodal** e **Auditoria**. Sem credenciais Azure, o sistema usa fallback determinístico e ainda demonstra o fluxo completo.

## Como Rodar

Pré-requisitos: Python ≥ 3.12, ~6 GB livres em disco (modelos + dataset).

```bash
# 1. Clonar e entrar no diretório (use Git LFS pra baixar o índice + PDFs)
git lfs install
git clone <repo-url> medica-ia-multimodal && cd medica-ia-multimodal

# 2. Criar venv e instalar dependências
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configurar variáveis de ambiente (opcional; o app roda offline com fallback)
cp .env.example .env
# Editar .env preenchendo as chaves Azure que você tiver

# 4. Subir a UI Gradio
python app.py
# Abrir http://127.0.0.1:7860
```

O índice RAG (Chroma + bge-m3 sobre 9 PDFs de diretrizes clínicas MS/INCA/FEBRASGO) já vem pronto em `data/processed/chroma/`, versionado via Git LFS. Os PDFs originais ficam em `data/raw/`. Modelos (YOLO stub, Whisper, wav2vec2, bge-m3) são baixados lazy pelos próprios pipelines na primeira chamada.

### Regenerar o índice RAG (opcional)

Só é necessário se você adicionar/remover PDFs em `data/raw/`. Por padrão, se o índice já existe em `data/processed/chroma/`, o script sai imediatamente sem trabalho (não duplica chunks, nem re-embeda, nem baixa modelo). Use `--force` pra reindexar do zero, que aí sim demora ~3 min em CPU rápido (e bem mais em máquinas lentas, já que a primeira execução baixa o `bge-m3`, ~2 GB).

```bash
python scripts/build_rag_index.py          # no-op se ja existe
python scripts/build_rag_index.py --force  # reindexa
```

Opcional: `python scripts/warmup.py` faz setup completo idempotente (re-baixa PDFs se sumirem, prepara dataset, gera índice RAG, baixa modelos). Aceita filtros `--models`, `--datasets`, `--pdfs`, `--examples`, `--skip-rag`. Detalhes em `python scripts/warmup.py --help`.

Os exemplos da aba **Multimodal** são gerados por `scripts/seed_real_examples.py` (orquestra `gen_tts_scripts.py` + `gen_contexts.py`): os MP4s já vêm versionados no repo em `data/examples/`, os áudios vêm de Azure Speech TTS PT-BR e os contextos clínicos saem do GPT-4.1-mini. Para regerar áudios e contextos:

```bash
python scripts/seed_real_examples.py --force   # áudios + contextos
python scripts/gen_tts_scripts.py --force      # só os WAVs
python scripts/gen_contexts.py --force         # só os contextos
```

Requer as variáveis `AZURE_SPEECH_*` e `AZURE_OPENAI_*` configuradas no `.env`.

## Integrações Azure

Quando as chaves estão preenchidas no `.env`, o sistema usa os serviços gerenciados abaixo; sem elas, cada pilar cai num fallback local equivalente.

| Serviço | Função | Onde é usado | Fallback offline |
|---|---|---|---|
| **Azure OpenAI** (GPT-4.1-mini, AI Foundry) | Gera o relatório clínico final em markdown | `src/llm/azure_openai.py`, `src/report.py` | Relatório determinístico em markdown construído a partir das triggers |
| **Azure Speech** | Transcrição de áudio + TTS para gerar voz PT-BR | `src/audio/transcriber.py` (toggle `USE_CLOUD_TRANSCRIPTION`) | `faster-whisper` local |
| **Azure Language** | Análise de sentimento + key phrases na transcrição | `src/audio/azure_language.py` | Pular esse pilar (não há substituto local equivalente) |
| **Azure OpenAI Vision** (GPT-4o multimodal) | Estado emocional via linguagem corporal + face em vídeo | `src/video/azure_openai_vision.py` (código pronto, deployment Azure pendente) | `FER` local (Py 3.12) |

A aba **Configurações** da UI mostra imediatamente quais serviços estão ativos (cloud) ou em fallback (local).

## Docker

```bash
docker compose up
```

## Testes

```bash
pytest -m smoke        # rápidos, com mocks
pytest -m integration  # end-to-end com pipelines mockados
ruff check src/ ui/ tests/ scripts/ app.py
```

## Treino do YOLO custom

O detector de vídeo usa pesos próprios treinados sobre CholecSeg8k (ADR-012). O treino é feito exclusivamente no Google Colab (GPU T4 gratuita): abrir [notebooks/train_yolo_colab.ipynb](notebooks/train_yolo_colab.ipynb), rodar do começo ao fim. O notebook baixa o dataset do Hugging Face, converte máscaras em bounding boxes, treina o YOLOv8n e exporta `best.pt` para download. O dataset não é mantido localmente no repo.

## Documentação

- [docs/overview.md](docs/overview.md): visão geral, escopo, métricas de sucesso
- [docs/arquitetura/arquitetura.md](docs/arquitetura/arquitetura.md): módulos, fluxo, interfaces
- [docs/arquitetura/decisoes_tecnicas.md](docs/arquitetura/decisoes_tecnicas.md): ADRs com justificativa
- [docs/arquitetura/modelos_e_datasets.md](docs/arquitetura/modelos_e_datasets.md): modelos e fontes de dados
- [docs/arquitetura/padroes_codigo.md](docs/arquitetura/padroes_codigo.md): convenções de código

## Equipe

Joalisson Borges, Luis Gustavo Santini, Marina Souza Lucas, Diego Santos.

## Licença

MIT. Ver [LICENSE](LICENSE).
