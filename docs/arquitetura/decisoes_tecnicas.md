# Decisões Técnicas

Registro de decisões importantes no formato ADR leve (Architecture Decision Record). Cada decisão tem contexto, opções consideradas, decisão tomada e consequências.

Atualizar conforme novas decisões surgem durante a implementação.

---

## ADR-001: Não reaproveitar código da Fase 3

**Contexto:** A Fase 3 entregou um assistente médico geral com Qwen3.5-4B fine-tunado, LangGraph + ReAct, MIMIC-IV. A Fase 4 muda o domínio para saúde da mulher e adiciona análise multimodal.

**Opções:**

A. Estender o pipeline LangGraph da Fase 3 adicionando módulos multimodais
B. Construir solução nova focada em análise multimodal
C. Híbrida: usar LLM da Fase 3 e construir o resto novo

**Decisão:** Opção B.

**Justificativa:** o núcleo técnico da Fase 4 é diferente da Fase 3. Reaproveitar adiciona mais ônus (Ollama no Colab, fine-tuning genérico, LangGraph com 6 nós que não fazem mais sentido) do que ganho.

**Consequências:** notebook menor, arquitetura limpa, onboarding rápido. Perde narrativa de continuidade que será mencionada no relatório técnico mas não no código.

---

## ADR-002: UI em Gradio Blocks ao invés de React + FastAPI

**Contexto:** A entrega exige um vídeo demo de 15 minutos mostrando o sistema funcionando. UI ajuda a apresentar.

**Opções:**

A. Gradio (Interface ou Blocks)
B. Streamlit
C. React + FastAPI separados

**Decisão:** Gradio Blocks.

**Justificativa:** trade-off tempo vs qualidade. Gradio Blocks permite layout custom, tema próprio, componentes ML especializados (Video, Audio, Plot) e roda em 1-2 dias. React + FastAPI levaria 5-10 dias para resultado equivalente.

**Consequências:** UI fica menos "produto real" mas suficiente para a demo. Backend e frontend no mesmo processo, deploy unificado.

---

## ADR-003: Hospedagem em Hugging Face Spaces

**Contexto:** precisa de URL público para a demo e para incluir no README.

**Opções:**

A. Hugging Face Spaces (free)
B. Azure Container Apps (pago)
C. Cloud Run (pago)
D. Apenas local com Docker

**Decisão:** Hugging Face Spaces.

**Justificativa:** zero custo, deploy via git push, suporta Gradio nativo, link público estável. Hibernação ao ocioso é aceitável para demo.

**Consequências:** uso limitado a 16GB RAM e 2 vCPU. Não tem GPU mas a inferência foi planejada para CPU.

---

## ADR-004: Provider de cloud é Azure (não AWS)

**Contexto:** o enunciado cita Azure Cognitive Services, mas a grade da fase ensina AWS Textract + Comprehend.

**Opções:**

A. Azure puro
B. AWS puro
C. Híbrido Azure + AWS

**Decisão:** Azure puro.

**Justificativa:** o vídeo demo da entrega lista explicitamente "Integração dos serviços Azure". Tratar como obrigatório é mais seguro do que apostar na interpretação literal do enunciado.

**Consequências:** AWS Textract e Comprehend Medical ficam fora do escopo, não cobrindo a parte da grade que ensina esses serviços. O relatório técnico justifica a escolha.

---

## ADR-005: Provider de LLM é Azure OpenAI

**Contexto:** projeto precisa de LLM para sumarização clínica e geração de relatórios.

**Opções:**

A. OpenAI direto
B. Azure OpenAI
C. Modelos open source local

**Decisão:** Azure OpenAI.

**Justificativa:** mantém tudo no ecossistema Azure, reforça o requisito de integração Azure do desafio. Mesmos modelos GPT-4.1-mini e Whisper, com SLA empresarial.

**Consequências:** precisa provisionar deployment de modelo no Azure OpenAI (via AI Foundry, caminho de provisionamento adotado no 2026-05-14, sem espera de aprovação manual). Custo controlado por limites no painel.

---

## ADR-006: YOLO base pré-treinado primeiro, custom depois

**Contexto:** treinar YOLO custom desde o início bloqueia o desenvolvimento das outras frentes (UI, fusão, RAG).

**Opções:**

A. Treinar YOLO custom antes de tudo o mais
B. Usar YOLOv8 base do ultralytics como stub
C. Usar YOLO pré-treinado clínico do Roboflow Universe

**Decisão:** Opção C primeiro, depois treino custom em etapa posterior.

**Justificativa:** padrão "stub the model" destrava o desenvolvimento. Detector é model-agnostic, troca de pesos não afeta arquitetura. Roboflow Universe oferece modelos clínicos prontos.

**Consequências:** o pipeline inicial começa rápido; a etapa de treino custom entrega o modelo final.

---

## ADR-007: YOLO target é sangramento anômalo

> **SUPERSEDED por [ADR-012](#adr-012-yolo-custom-muda-alvo-para-instrumentos-cirúrgicos-cholecseg8k).** Registro mantido por histórico de decisão.

**Contexto:** o enunciado lista 4 alvos possíveis para o YOLOv8 customizado.

**Opções:**

A. Instrumentos cirúrgicos ginecológicos
B. Áreas críticas (útero, ovários, mamas)
C. Sangramento anômalo durante procedimentos
D. Objetos suspeitos de automutilação

**Decisão:** Opção C, sangramento anômalo.

**Justificativa:** alinhamento direto com saúde materna, maior chance de obter dado rotulado público, relevância clínica imediata, baixo risco ético comparado a opção D.

**Consequências:** dataset precisa ser obtido via Roboflow ou anotação manual em pequena escala.

---

## ADR-008: Vector store Chroma para RAG

**Contexto:** RAG precisa de vector store funcional para diretrizes clínicas.

**Opções:**

A. Chroma persistente
B. FAISS
C. pgvector
D. Pinecone

**Decisão:** Chroma persistente.

**Justificativa:** zero setup, persiste em arquivo, integra direto com LangChain, suporta filtros por metadata. Suficiente para algumas dezenas de PDFs.

**Consequências:** se o volume crescer muito, migrar para FAISS ou pgvector. Não é o caso aqui.

---

## ADR-009: Idioma do código em EN, comentários e docs em PT-BR

**Contexto:** projeto tem nomenclatura técnica padrão e domínio clínico brasileiro.

**Opções:**

A. Tudo em PT-BR
B. Tudo em EN
C. Híbrido: nomes em EN, docs e comentários em PT-BR

**Decisão:** Opção C.

**Justificativa:** nomes em EN seguem padrão da indústria e libs externas. Comentários e docs em PT-BR são mais naturais para o domínio clínico brasileiro e para o avaliador.

**Consequências:** algumas mensagens de log e variáveis terão mistura. Manter consistência interna por módulo.

---

## ADR-010: Audit log em SQLite

**Contexto:** sistema precisa registrar decisões e ações para auditoria.

**Opções:**

A. SQLite local
B. Postgres em container
C. Logs em arquivos JSON

**Decisão:** SQLite local.

**Justificativa:** zero setup, query SQL, persiste, tamanho irrisório. Para auditoria de eventos de sessão é perfeito.

**Consequências:** não escala para múltiplos usuários simultâneos, mas o caso de uso atual é demonstração.

---

## ADR-011: Stub `yolov8n.pt` inicial; Roboflow Universe avaliado para etapa de treino custom

**Contexto:** ADR-006 escolheu Roboflow Universe como fonte do modelo pré-treinado para destravar o pipeline inicial. Na prática, garimpar um modelo público de "bleeding" no Universe com pesos baixáveis, license permissiva e mAP confiável tem custo de tempo alto, e o Universe ainda bloqueia automação externa. Isso bloqueia o pipeline inicial inteiro por algo que é desnecessário no curto prazo: o detector é model-agnostic e o treino custom está planejado para etapa posterior de qualquer jeito.

**Opções:**

A. Continuar buscando modelo Roboflow ideal antes de destravar o pipeline
B. Stub com `yolov8n.pt` base (Ultralytics) imediatamente, trocar pelos pesos custom na etapa de treino
C. Stub com `yolov8n.pt` agora E pesos Roboflow assim que aparecerem como melhoria opcional

**Decisão:** Opção B.

**Justificativa:** o treino custom está planejado para etapa posterior (com dataset rotulado vindo de Roboflow Universe ou outras fontes). Não há ganho real em ter modelo "intermediário Roboflow" entre `yolov8n` base e o custom final. Roboflow Universe segue como **fonte de dataset** (não de modelo) na etapa de treino.

**Consequências:**
- O pipeline inicial desbloqueia imediatamente, sem dependência externa.
- O detector aceita qualquer `.pt` via `YOLO_WEIGHTS_PATH` (env var), mantendo a model-agnosticidade prometida no ADR-006.
- ADR-006 letra C continua válido para a etapa de treino custom: avaliar Roboflow Universe como dataset; alternativas (Kaggle, HuggingFace, PhysioNet) ficam abertas.
- Stub atual detecta classes COCO (não `bleeding`). O pipeline roda end-to-end mesmo assim; a relevância clínica vem do treino custom.

---

## ADR-012: YOLO custom muda alvo para instrumentos cirúrgicos (CholecSeg8k)

**Supersede:** ADR-007.

**Contexto:** ADR-007 escolheu "sangramento anômalo" como alvo do YOLO custom. Ao executar o provisionamento de dataset, uma rodada de pesquisa empírica em fontes públicas (Roboflow Universe, Kaggle, Hugging Face, PhysioNet, repositórios académicos) revelou que nenhuma das opções viáveis sustenta o alvo "sangramento" no domínio ginecológico:

- **WCEBleedGen (Kaggle/Wireless Capsule Endoscopy):** sangramento sim, mas em gastroenterologia (cápsula endoscópica). Domínio não-ginecológico, framing forçado.
- **BUSI (Breast Ultrasound Images):** ultrassom mamário diagnóstico, sem sangramento. Substituir "sangramento" por "lesão mamária" seria mudança implícita, sem aderência ao enunciado.
- **Dresden Surgical Anatomy Dataset:** alvo cirúrgico ginecológico real, mas 19 GB com licença restritiva e pipeline de download inviável para reprodutibilidade em Colab/HF Spaces.
- **CholecSeg8k (Hong et al., arXiv 2012.12463, CC BY-NC-SA 4.0):** 8080 frames anotados de colecistectomia laparoscópica, 3.1 GB, disponível anônimo via Hugging Face Datasets. Contém **Grasper** e **L-hook Electrocautery** anotados, exatamente os instrumentos usados em cirurgia ginecológica laparoscópica.

**Opções:**

A. Manter "sangramento" mesmo sem dataset ginecológico real (forçar com WCEBleedGen)
B. Pivotar para "lesão mamária" usando BUSI (US diagnóstico)
C. Pivotar para "instrumentos cirúrgicos" usando Dresden (19 GB inviável)
D. Pivotar para "instrumentos cirúrgicos" usando CholecSeg8k (3.1 GB, HF, classes adequadas)

**Decisão:** Opção D. YOLO custom passa a detectar **instrumentos cirúrgicos** (classes: `Grasper`, `L-hook Electrocautery`) treinado em CholecSeg8k.

**Justificativa:**
- Aderência LITERAL ao alvo 1 do enunciado ("Instrumentos cirúrgicos ginecológicos"). Foi a única opção do enunciado que sustenta dado público real.
- Reprodutibilidade: 3.1 GB cabe no fluxo Colab + Hugging Face, sem login proprietário ou EULA bloqueante.
- Técnica laparoscópica idêntica entre colecistectomia (treino) e cirurgia ginecológica (aplicação): mesmo trocarte, mesma pinça Grasper, mesmo L-hook eletrocautério. Transferência de domínio justificada clinicamente.
- Licença CC BY-NC-SA 4.0 compatível com uso académico/demonstração.

**Consequências:**
- Dataset baixado em `data/raw/cholecseg8k/`, convertido para formato YOLO em `data/processed/cholecseg8k_yolo/` (8080 frames, 2 classes, splits 5656/1616/808 train/val/test).
- Refactor em código: `LESION_MAMA_*` → `SURGICAL_INSTRUMENT_*` em `src/anomaly/rules.py`, `src/anomaly/statistical.py` e respectivos testes.
- **Semântica nova:** detecção de instrumento é estado NORMAL de cirurgia laparoscópica. Trigger correspondente eleva nível para `moderate` apenas (sinaliza presença de procedimento invasivo em curso); nunca para `critical` isoladamente. Combinação com outros triggers (vocais, textuais) pode escalar.
- ADR-007 superseded. Histórico de pivots preservado neste registro: WCEBleedGen → BUSI → Dresden → CholecSeg8k.

---

## ADR-013: Estratégia de áudio (TTS Azure scriptado + validação CORAA-SER)

**Contexto:** o pipeline de áudio precisa de exemplos clínicos em PT-BR para demonstrar emoção vocal, transcrição e classificação de afeto. Não existe corpus público brasileiro de consultas ginecológicas/obstétricas reais com rótulos de emoção (pacientes reais implicam LGPD e comité de ética, inviável no escopo do challenge). Corpora estrangeiros (RAVDESS, IEMOCAP) são em inglês e atuados, e o pipeline Whisper + wav2vec2 precisa de PT-BR para ser convincente.

**Opções:**

A. Apenas TTS Azure scriptado (controlado mas artificial)
B. Apenas CORAA-SER (PT-BR espontâneo mas sem cenários clínicos específicos)
C. Híbrida: TTS Azure como "rótulo gold" da demo + CORAA-SER como validação de generalização

**Decisão:** Opção C.

**Justificativa:**
- TTS Azure (vozes Francisca, Antonio, Brenda, etc., com estilos `sad`, `empathetic`, `terrified`) produz áudios PT-BR controlados, com rótulo de emoção determinístico e cenário clínico scripted. Serve como gold standard para a demo.
- CORAA-SER (`alefiury/CORAA-SER` no Hugging Face) fornece fala espontânea PT-BR rotulada em emoção. Usado para validar que o classificador wav2vec2 não overfita ao timbre sintético do TTS.
- Estratégia híbrida mantém demo previsível (TTS) e responsabilidade técnica (validação em fala real).

**Consequências:**
- O projeto gera áudios TTS Azure em `data/synthetic/audio/` cobrindo cenários: normal, ansiedade, depressão, sofrimento agudo.
- `tests/integration/test_audio_validation.py` (a criar) roda subset de CORAA-SER e reporta acurácia agregada do classificador.
- TTS Azure exige `AZURE_SPEECH_KEY` (já provisionado). CORAA-SER baixado on-demand pelo `datasets` da HF.

---

## ADR-014: 3ª funcionalidade (sinais vitais) marcada como Adiado

**Contexto:** o enunciado pede no mínimo 2 funcionalidades dentre as 4 listadas. O projeto já cobre 3 funcionalidades atuais: análise de vídeo, processamento de áudio e integração com Azure Cognitive Services. A 4ª opção do enunciado é "Monitoramento de sinais vitais" (integração com dados de cardiotocografia, pressão arterial, ECG, etc.).

**Opções:**

A. Implementar sinais vitais como 4ª modalidade (cardiotocografia ou PA via CSV/JSON sintético)
B. Adiar formalmente, marcar como `⏸️ Adiado` no roadmap

**Decisão:** Opção B.

**Justificativa:**
- Estimativa de esforco adicional significativo (modelagem de serie temporal + nova aba + novos triggers + integracao no orquestrador) para ganho marginal frente ao requisito de >=2 funcionalidades ja cumprido.
- Quebra coerência multimodal: vídeo + áudio + texto formam uma análise unificada (humana, qualitativa); série temporal de PA/CTG é outro paradigma (numérico, monitoramento contínuo) e exigiria UX própria.
- Aderência ao padrão "stub-first + fallback local": registrar como adiado é melhor do que entregar versão parcial.

**Consequências:**
- Roadmap exibe `⏸️ Adiado` para sinais vitais; mencionado em "Próximos Passos para Evoluir" em `arquitetura.md`.
- Relatório técnico e vídeo demo explicitam que 3 funcionalidades já superam o mínimo do enunciado.
- Caso outro grupo do trio ou avaliador peça, a 4ª funcionalidade fica pré-arquitetada para implementação futura sem refactor.

---

## ADR-015: Remocao do pilar YOLOv8 Pose em favor de GPT-vision multimodal

**Contexto:** O pilar de pose corporal usava YOLOv8 Pose (Ultralytics, ~6 MB) pra extrair 17 keypoints COCO de cenas de consulta e classificar postura geometricamente (`ereta`, `ereta_tensa`, `inclinada`, `retraida`, `indefinido`). Substituiu o MediaPipe Pose anterior por causa de duas limitacoes (single-person e incompatibilidade com Py 3.14). Apos validacao com inputs reais, identificamos que:

1. A heuristica "maior bbox" nao identifica a paciente corretamente em cenas com medico + paciente. A bbox alternava entre as duas pessoas em frames consecutivos, gerando classificacao postural pouco estavel.
2. Nos cinco videos de demonstracao, a categoria predominante sempre saiu "ereta", sem diferenciar casos clinicamente relevantes. A categoria `ereta_tensa` (heuristica composta s1 + s2) nunca disparou.
3. O GPT-vision (Azure OpenAI multimodal) ja analisava postura, gestos e expressao facial como evidencia interna pra inferir emocao, com a vantagem de identificar a paciente pelo contexto semantico em vez da heuristica geometrica.
4. O pilar agregava mais como demonstracao tecnica de CV classica do que como sinal analitico real.

**Opcoes consideradas:**
A. Manter YOLOv8 Pose com agregacao multi-pose (avaliar todas as pessoas detectadas)
B. Adicionar heuristica posicional (paciente sempre a direita) sobre a bbox principal
C. Remover YOLOv8 Pose e ensinar o GPT-vision a emitir tambem a categoria de linguagem corporal predominante

**Decisao:** opcao C. O detector YOLOv8 custom em CholecSeg8k (instrumentos cirurgicos) ja cobre o requisito do enunciado de "analise de videos clinicos com YOLOv8". O pilar humano fica centralizado no GPT-vision multimodal, que agora retorna dois sinais por frame analisado: `label` (emocao) e `body_language` (uma de: `tranquila`, `tensa`, `retraida`, `agitada`, `indefinida`).

**Consequencias:**
- Pipeline com dois pilares de video: YOLO custom (instrumentos em cirurgia) + GPT-vision (emocao e linguagem corporal em consulta), sob gating por tipo de cena.
- KPI "Postura" na aba Video substituido por "Linguagem corporal".
- `src/video/pose.py` removido. `PostureCategory`, `PoseLandmark`, `PoseEstimator`, `classify_posture` deixam de existir. Schema `VideoEvent` perde o campo `pose_landmarks`.
- `EmotionScore` ganha campo opcional `body_language: str | None`. Backends que nao oferecem o sinal (FER local) deixam `None`.
- Codigo do pilar removido foi preservado em workspace pessoal (`medica-ia-workspace/execucao/yolo_pose_backup.md`) pra eventual restauracao futura.
- `scene_classifier.py` mantem uso de `mediapipe.solutions.face_detection` (face apenas, sem pose), com lazy import e fallback silencioso (sem erro) quando mediapipe nao esta disponivel.

---

## ADR-016: Azure Speech com reconhecimento continuo no transcritor cloud

**Contexto:** A primeira implementacao do `AzureSpeechTranscriber` (`src/audio/transcriber.py`) usava `recognize_once_async`, que processa o audio ate a primeira pausa significativa e encerra. Em audios reais de consulta clinica (acima de 60 segundos, com pausas naturais entre falas) isso devolvia apenas a primeira frase, ignorando o restante do conteudo. O pipeline a jusante recebia transcricao incompleta, e a analise de sentimento/key phrases via Azure Language nao tinha texto suficiente pra produzir sinal util.

**Opcoes consideradas:**

A. Manter `recognize_once_async` e segmentar o audio em chunks de poucos segundos antes de enviar (juncao manual no cliente).
B. Migrar para `start_continuous_recognition_async` com handlers de evento (`recognized`, `session_stopped`, `canceled`) acumulando segmentos.
C. Trocar inteiramente pra batch transcription da Azure Speech (API REST assincrona com polling).

**Decisao:** Opcao B. O cliente cloud passa a usar reconhecimento continuo, acumulando cada frase finalizada em um `Segment` com timestamps em milissegundos.

**Justificativa:**
- Cobre audios longos (acima de 60s) sem chunking manual, que introduziria erros nas bordas e dificultaria a reconstrucao de timestamps absolutos.
- Mantem a mesma interface `transcribe(path) -> (text, segments)` que o `WhisperTranscriber` local. O pipeline a jusante nao percebe a mudanca.
- Os eventos `recognized` ja entregam `offset` e `duration` em ticks de 100 nanosegundos, convertidos para milissegundos no cliente, alimentando diretamente o schema `Segment` que o Whisper local ja produz.
- Evita complexidade operacional da batch transcription (polling de status, gerenciamento de blob storage de saida), que e mais adequada pra cargas em massa.

**Consequencias:**
- `AzureSpeechTranscriber.transcribe` agora abre um `threading.Event` que e setado pelo handler `session_stopped` (fim normal) ou `canceled` (erro), aguardando o fim da execucao continua antes de retornar.
- Erros do servico sao logados via handler `canceled` com `error_code` e `error_details`, sem propagar excecao: o pipeline cai pra texto vazio e segue (mesmo comportamento de falha de cliente nao configurado).
- O fallback local (`faster-whisper small`) continua intacto e e selecionado quando `USE_CLOUD_TRANSCRIPTION=false` ou quando o cliente Azure nao esta configurado.

---

## ADR-017: RAG multi-query por eixo tematico (substitui query unica)

**Contexto:** O `_retrieve_context` no orchestrator concatenava contexto clinico, transcricao de audio e mensagens de trigger numa unica query passada ao retriever. Em casos com sinais emocionais sobre quadro nao gestacional (ex.: paciente dermatologica com angustia, depressao fora do pos-parto, violencia domestica), o ranking do `bge-m3` era dominado pelo PDF mais volumoso da colecao (geralmente `manual_ms_prenatal`, que tem uma secao psicologica grande). O CAB 34 (saude mental geral), recem adicionado, nao aparecia entre os top-4 chunks, e o LLM acabava recomendando "saude mental perinatal" pra pacientes nao gestantes.

**Opcoes consideradas:**

A. Aumentar `rag_top_k` de 4 pra 6 e ajustar prompt do LLM pra ignorar pre-natal em paciente nao gestante.

B. Concatenar palavras-chave clinicas extra na query unica quando triggers sinalizam tema especifico (boost de termos).

C. Dividir a recuperacao em multiplas queries focadas por eixo tematico (clinico, saude mental, violencia, reprodutivo, rastreio), fundir resultados via interleaving round-robin com dedup por chunk_id.

**Decisao:** Opcao C, em conjunto com aumento moderado de `rag_top_k` para 6.

**Justificativa:**
- Generica: cada eixo identificado pelas keywords ativa uma query especifica com sufixo clinico que orienta o retriever pro PDF correto. O caso dermatologica + ansiedade ativa `saude_mental`, o caso violencia domestica ativa `violencia`, o caso pre-natal ativa `reprodutivo`, e assim por diante. Nao depende de regras casuisticas por especialidade.
- Cirurgica: queries separadas evitam que sinais clinicos competam com sinais emocionais no mesmo vetor. O `bge-m3` retorna chunks mais relevantes do PDF correto para cada eixo.
- Escalavel: adicionar novo PDF (ex.: climaterio) requer apenas adicionar um eixo novo no dicionario `RAG_AXES`, sem mudancas em codigo de runtime.
- Honesta: se nenhum eixo casa com o caso, so a query base roda (mesmo comportamento de antes), sem ruido.
- Multi-query e padrao em sistemas RAG modernos para queries complexas (LangChain MultiQueryRetriever) e cobre o cenario "uma query nao representa todos os angulos do caso".

**Implementacao:**

- `src/rag/retriever.py:multi_search(queries, top_k_per_query, total_k, min_score)`: roda cada query independente, dedupe por `chunk_id`, interleaving round-robin (rodada 1 pega 1o chunk de cada query, rodada 2 pega 2o, etc) ate completar `total_k`. Score final por chunk e o melhor observado entre as queries em que ele apareceu.
- `src/orchestrator.py:RAG_AXES`: dicionario com 4 eixos (`saude_mental`, `violencia`, `reprodutivo`, `rastreio`), cada um com keywords (gatilhos a buscar no haystack lowercase) e sufixo (texto clinico que orienta a query focada).
- `_retrieve_context`: monta query base (contexto + transcricao + triggers) e, pra cada eixo detectado no haystack, adiciona query focada com `contexto[:200] + sufixo do eixo`. Chama `multi_search` com `top_k_per_query=3, total_k=6`.

**Consequencias:**
- Custo extra de embedding: 1 a 5 embeddings locais por caso (1 query base + ate 4 eixos detectados). Tudo local via `bge-m3`, sem custo Azure adicional.
- Latencia adicional: ~50ms por embedding extra (~200ms no pior caso). Imperceptivel no fluxo do orchestrator que ja gasta segundos em LLM.
- `DEFAULT_RAG_TOP_K` aumentado de 4 para 6 chunks no contexto final do LLM (mais tokens consumidos no prompt, ~50% a mais), mas em troca de cobertura mais ampla do tema.
- Tests novos cobrem dedup, interleaving e queries vazias. Mocks de testes existentes precisam stubbar `multi_search.return_value` alem de `search.return_value`.

---

## ADR-018: Remocao do Azure Face Service em favor de Azure OpenAI vision multimodal

**Contexto:** A configuracao inicial do projeto previa o Azure Face Service como provider cloud opcional para emocao facial, ativado pelo toggle `USE_CLOUD_EMOTION` em conjunto com `AZURE_FACE_KEY` / `AZURE_FACE_ENDPOINT`. Na pratica, o servico nunca chegou a ser exercitado em runtime: nao houve provisionamento da chave nem implementacao de cliente especifico. Apos a introducao do `AzureOpenAIVisionEmotion` (`src/video/azure_openai_vision.py`), que usa GPT-vision multimodal para inferir simultaneamente emocao e linguagem corporal numa unica chamada e sem o vies de FER-2013, a presenca do Azure Face no codigo passou a ser ruido: campos de settings sem uso, entradas na aba de Configuracoes que nunca ficariam verdes, e documentacao sugerindo um caminho que ja foi superado.

**Opcoes consideradas:**

A. Manter os campos como contrato historico, marcando-os como deprecated com aviso na UI.

B. Remover por completo (settings, env vars, entrada na aba de Configuracoes, mencoes em docs) e tratar o GPT-vision como unico caminho cloud para emocao facial.

C. Manter o toggle `USE_CLOUD_EMOTION` como flag generica desacoplada do Face, controlando apenas o roteamento cloud/local independente do provider.

**Decisao:** opcao B. Os campos `azure_face_key`, `azure_face_endpoint` e o toggle `use_cloud_emotion` foram removidos do `Settings` (`src/config/settings.py`), do `.env.example` e da aba de Configuracoes da UI (`ui/tabs/tab_config.py`). A funcao `emotion_provider_label()` (`ui/components.py`) passa a checar somente o caminho GPT-vision, com fallback explicito para FER local.

**Justificativa:**

- O Azure Face Service exige aprovacao via processo de Responsible AI da Microsoft para emocao, com prazo e overhead que nunca couberam no escopo do projeto.
- O GPT-vision multimodal cobre o mesmo requisito (emocao facial em video clinico) e entrega o sinal adicional `body_language`, simplificando o pipeline.
- Manter codigo morto em settings e UI confunde quem le o repositorio e gera atrito desnecessario na aba de Configuracoes, onde apareceria sempre como "Inativo" sem caminho de ativacao real.

**Consequencias:**

- `Settings` perde dois campos secretos e um boolean toggle. Testes que validavam o default de `use_cloud_emotion` foram simplificados (`tests/unit/test_config.py`).
- `.env.example` mais enxuto. Quem ja tinha `AZURE_FACE_KEY` / `AZURE_FACE_ENDPOINT` no `.env` local pode remover manualmente; Pydantic Settings ignora chaves extras (`extra="ignore"`).
- Aba de Configuracoes deixa de listar Azure Face entre os servicos Azure. A linha "Emocao facial" agora reflete apenas dois estados: GPT-vision ativo (cloud) ou FER local (fallback).
- `FacialEmotionDetector` continua intacto como fallback offline, conforme padrao "stub-first + fallback local" adotado em todo o projeto.

---

## ADR-019: Remocao do Azure Video Indexer Service (nao exercitado)

**Contexto:** A arquitetura inicial previa o Azure Video Indexer (VI) como provider cloud para enriquecer cada caso com cenas, transcricao embutida, deteccao de objetos e analise visual semantica, ativado pelas variaveis `AZURE_VIDEO_INDEXER_KEY` e `AZURE_VIDEO_INDEXER_ACCOUNT_ID`. Na pratica, o cliente (`src/video/azure_video.py`) sempre permaneceu como esqueleto que retornava `None` em `analyze()` por falta de provisionamento: nem o fluxo de obtencao de access token, nem o upload + polling de Insights chegaram a ser implementados. O `VideoEvent` carregava o campo `azure_metadata: dict | None`, fixo em `None` em todo caminho de execucao real. Em paralelo, os pilares ja implementados passaram a cobrir o mesmo conjunto de sinais que o VI ofereceria:

- **Transcricao:** Whisper local (`faster-whisper`) e Azure Speech com reconhecimento continuo (ADR-016) ja produzem texto e segmentos com timestamps a partir do audio extraido do video.
- **Analise de cenas:** `src/video/scene_classifier.py` ja identifica o tipo de cena (`SURGERY`, `CONSULTATION`, `MIXED`, `UNKNOWN`) e direciona o pipeline por gating, dispensando o classificador de cenas do VI.
- **Deteccao de objetos:** o YOLO custom em CholecSeg8k (ADR-012) cobre instrumentos cirurgicos com aderencia clinica superior a deteccao generica do VI.
- **Analise visual semantica:** o GPT-vision multimodal (`src/video/azure_openai_vision.py`, ADR-018) ja entrega emocao + linguagem corporal por frame com identificacao da paciente por contexto.

**Opcoes consideradas:**

A. Manter o esqueleto e o campo `azure_metadata`, marcando-os como deprecated com aviso, ate que algum provisionamento futuro motive a implementacao completa.

B. Remover por completo o modulo `azure_video.py`, o campo `azure_metadata` em `VideoEvent` e `AudioAnalysis`, os tres settings (`azure_video_indexer_key`, `azure_video_indexer_account_id` e `azure_video_indexer_region`, quando aplicaveis), as entradas correspondentes em `.env.example` e na aba de Configuracoes, e quaisquer mencoes em docs e relatorio.

C. Substituir o cliente VI por uma implementacao funcional via batch transcription + polling, integrando-o ao orquestrador como passo assincrono.

**Decisao:** opcao B. O Azure Video Indexer e removido do projeto. Pipelines sincronos cobrem o que o VI ofereceria sem introduzir custos cloud adicionais, complexidade de polling assincrono ou dependencia de provisionamento bloqueada por aprovacao da Microsoft.

**Justificativa:**

- O VI nunca foi exercitado em runtime; manter codigo morto em `src/`, `tests/` e docs gera atrito de leitura e sinaliza compromissos arquiteturais inexistentes.
- Os quatro insights do VI (cenas, transcricao, objetos, semantica visual) ja sao cobertos por modulos proprios mais aderentes ao dominio clinico (YOLO custom em laparoscopia, scene classifier por HSV + face detection, Whisper/Azure Speech, GPT-vision).
- Eliminar tres variaveis de ambiente simplifica o deploy em Hugging Face Spaces e o roteiro de configuracao para quem rodar o projeto pela primeira vez.
- O pipeline atual e sincrono e roda em segundos por caso; o VI exigiria polling de Insights com latencia tipica de minutos, incompativel com a UX da aba multimodal.

**Consequencias:**

- `src/video/azure_video.py` removido. `src/video/__init__.py` deixa de exportar `AzureVideoIndexerClient`.
- `VideoEvent` perde o campo `azure_metadata`. `AudioAnalysis` perde o campo equivalente, que era vestigial da mesma decisao.
- `VideoPipeline.__init__` deixa de aceitar `azure_client`. A chamada `self.azure_client.analyze(video_path)` e removida do loop de processamento.
- `Settings` perde os campos `azure_video_indexer_key` e `azure_video_indexer_account_id`. `.env.example` perde o bloco correspondente.
- Aba de Configuracoes deixa de listar "Azure Video Indexer" entre os servicos Azure.
- Testes em `tests/unit/test_video.py`, `tests/unit/test_audio.py`, `tests/unit/test_anomaly.py` e `tests/integration/test_orchestrator.py` deixam de mockar o cliente VI e de assertar o campo `azure_metadata`.
- Quem ja tinha as chaves no `.env` local pode remover manualmente; Pydantic Settings ignora chaves extras (`extra="ignore"`).
- Documentacao (`arquitetura.md`, `modelos_e_datasets.md`, `overview.md`, `relatorio_tecnico.md`) atualizada para remover mencoes ao VI dos diagramas, schemas e tabela de stack cloud.

---

## Como Adicionar Nova ADR

1. Próximo número sequencial (ADR-011, etc.)
2. Título descritivo da decisão
3. Contexto explicando o problema
4. Opções consideradas (mínimo 2)
5. Decisão tomada com justificativa
6. Consequências (positivas e negativas)
7. Linkar de outros docs quando relevante
