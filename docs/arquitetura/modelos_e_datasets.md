# Modelos e Datasets

Lista consolidada de modelos pré-treinados, datasets e outras fontes de dados usadas no projeto. Atualizar conforme novos itens forem adquiridos.

## Modelos

### Visão

| Modelo | Onde | Uso | Observação |
|---|---|---|---|
| YOLOv8n base | Ultralytics (download automático) | Stub para destravar o pipeline durante o desenvolvimento | Detecta classes COCO; sem relevância clínica direta. ADR-011 |
| YOLOv8n custom | Treino próprio no Colab sobre CholecSeg8k | Modelo final para detecção de instrumentos cirúrgicos (Grasper, L-hook Electrocautery) | Substitui o stub via `YOLO_WEIGHTS_PATH` no `.env`. ADR-012 |
| FER (Facial Expression Recognition) | github/justinshenk/fer | Emoção facial | Pré-treinado, CPU |

### Áudio

| Modelo | Onde | Uso | Observação |
|---|---|---|---|
| faster-whisper small | github/SYSTRAN/faster-whisper | Transcrição local (fallback) | Multilíngue, suporta PT-BR. `compute_type=int8` em CPU |
| Azure Speech (continuous) | Azure Cognitive Services | Transcrição cloud (toggle `USE_CLOUD_TRANSCRIPTION`) | Reconhecimento contínuo (`start_continuous_recognition_async`) com handlers `recognized`/`session_stopped`/`canceled`; captura áudio inteiro acima de 60s com timestamps por segmento. ADR-016 |
| wav2vec2 emotion | superb/wav2vec2-base | Classificação de emoção (fallback local) | Pré-treinado em RAVDESS; enviesado para angry/sad em PT-BR (ADR cita Azure OpenAI Audio como alternativa) |
| Azure OpenAI GPT-4o-audio | Azure AI Foundry (`gpt-audio-mini`) | Emoção vocal multimodal substituindo wav2vec2 | Deployment próprio. Recebe audio + prompt textual e retorna JSON com emoção classificada. Substitui wav2vec2 enviesado para angry/sad em PT-BR |

### Texto e LLM

| Modelo | Onde | Uso | Observação |
|---|---|---|---|
| BAAI/bge-m3 | Hugging Face | Embeddings RAG multilíngue | Roda em CPU, ~1GB |
| Azure OpenAI GPT-4.1-mini | Azure AI Foundry | Sumarização e geração de relatório | Deployment próprio no hub Foundry |

### TTS (geração de áudio sintético)

| Modelo | Onde | Uso | Observação |
|---|---|---|---|
| Azure Speech TTS | Azure Cognitive Services | Vozes em PT-BR para áudios sintéticos | Vozes Francisca, Antonio, etc |

## Datasets

### Visão

| Dataset | Fonte | Uso | Observação |
|---|---|---|---|
| **CholecSeg8k** | Hugging Face (`minwoosun/CholecSeg8k`), Hong et al., arXiv 2012.12463, CC BY-NC-SA 4.0 | Treino YOLO custom para instrumentos cirúrgicos | 8080 frames, 3.1 GB; classes alvo `Grasper` + `L-hook Electrocautery`; splits 5656/1616/808. ADR-012 |

### Datasets de demonstração

Conjunto de mídias reais usadas exclusivamente na demo e no vídeo de entrega (sem treino), cobrindo cenários do enunciado:

| Mídia | Fonte | Cenário coberto | Observação |
|---|---|---|---|
| CholecSeg8k test split | Hugging Face | Cirurgia laparoscópica real (validação visual do YOLO custom) | 808 frames separados do treino |
| AVOS Open Surgery | research.bidmc.org/surgical-informatics/avos | Cirurgia aberta (referência comparativa) | Vídeos publicamente disponíveis para pesquisa |
| Geeky Medics (YouTube CC) | YouTube, canal Geeky Medics, licença CC | Consulta clínica simulada (entrevista médico-paciente) | Mídia em EN; usar trechos curtos com legenda |
| Wikimedia Pelvic Floor | Wikimedia Commons | Fisioterapia pélvica / educação | CC BY-SA |
| WHO RESPECT Women | World Health Organization | Conteúdo sobre violência contra a mulher (sensível) | Uso ilustrativo apenas |
| RAVDESS | Universidade Ryerson | Atrizes em estados afetivos (referência cruzada) | Inglês, complementar ao TTS PT-BR |

### Áudio (validação em fala espontânea)

| Fonte | Onde | Uso | Observação |
|---|---|---|---|
| **Azure TTS PT-BR Neural** | Azure Cognitive Services | Geração de áudios scriptados em PT-BR com rótulo gold | Vozes Francisca, Antonio, Brenda + estilos `sad`, `empathetic`, `terrified`. ADR-013 |
| **CORAA-SER** | Hugging Face (`alefiury/CORAA-SER`) | Validação cruzada do classificador wav2vec2 em fala espontânea PT-BR | Garante que o classificador não overfita ao timbre sintético do TTS. ADR-013 |
| RAVDESS (legado) | Universidade Ryerson | Benchmark internacional de emoção | Inglês, comparativo apenas |

### Texto e RAG (knowledge base)

9 PDFs públicos indexados no Chroma. Cada documento cobre cenário clínico distinto:

| Documento | Slug | Fonte | Cenário coberto |
|---|---|---|---|
| Manual MS Pré-natal de baixo e alto risco | `manual_ms_prenatal` | Ministério da Saúde | Pré-natal de rotina, fatores de risco |
| Diretriz FEBRASGO Pré-eclâmpsia | `febrasgo_preeclampsia` | FEBRASGO | Hipertensão na gravidez, sinais de gravidade |
| Manual MS Gestação de Alto Risco | `ms_gestacao_alto_risco` | Ministério da Saúde | Condutas em gestação de alto risco |
| INCA, Detecção Precoce do Câncer de Mama | `inca_cancer_mama` | INCA | Rastreio mamográfico, sinais clínicos |
| INCA, Detecção Precoce do Câncer do Colo do Útero | `inca_cancer_colo_utero` | INCA | Citologia, HPV, condutas |
| MS, PCDT IST e Atenção a Vítimas de Violência | `ms_pcdt_ist_violencia` | Ministério da Saúde | IST, violência sexual, profilaxia |
| MS, Diretrizes de Atenção ao Parto Normal | `ms_parto_normal` | Ministério da Saúde | Parto fisiológico, intervenções |
| Caderno AB nº 26, Saúde Sexual e Reprodutiva | `cab26_saude_sexual_reprodutiva` | Ministério da Saúde | Contracepção, planejamento reprodutivo |
| Caderno AB nº 34, Saúde Mental | `cab34_saude_mental` | Ministério da Saúde | Transtornos mentais comuns na APS, manejo e encaminhamento |

## Convenções

### Diretório `models/`

- Modelos pequenos (menos de 100MB) commitados no Git
- Modelos maiores via Git LFS ou release Git
- Cada modelo com arquivo `_README.md` adjacente listando: origem, licença, métricas

### Diretório `data/`

- `data/raw/`: dados brutos baixados, gitignored
- `data/processed/`: dados pré-processados, gitignored exceto índice Chroma
- `data/synthetic/`: áudios e exemplos sintéticos, amostras pequenas commitadas
- `data/examples/`: vídeos e áudios usados na demo, sempre commitados (pequenos)

### Atualizações

- Ao baixar novo modelo, adicionar linha à tabela acima
- Ao usar novo dataset, registrar fonte e licença
- Manter este arquivo como inventário consolidado

## Licenças e Considerações Legais

- CholecSeg8k: licença CC BY-NC-SA 4.0. Uso académico/demo permitido, citar Hong et al. (arXiv 2012.12463)
- CORAA-SER: verificar termos da Hugging Face (`alefiury/CORAA-SER`) antes de redistribuir
- PDFs do MS, FEBRASGO e INCA são públicos para uso educacional
- Áudios gerados por TTS Azure têm uso comercial permitido
- Áudios reais de pacientes não são usados (privacidade e LGPD)
