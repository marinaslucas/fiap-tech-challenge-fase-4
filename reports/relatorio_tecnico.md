# Relatório Técnico: medica-ia-multimodal

Tech Challenge Fase 4, pós-graduação Tech IADT.

**Equipe:** Joalisson Borges, Luis Gustavo Santini, Marina Souza Lucas, Diego Santos.

**Repositório:** https://github.com/joalissonborges94/medica-ia-multimodal

---

## 1. Resumo Executivo

O projeto entrega um sistema de monitoramento multimodal voltado à saúde da mulher, processando vídeo, áudio e texto para gerar nível de risco, relatório clínico e alerta estruturado. Cobre três das quatro funcionalidades do enunciado (análise de vídeo, processamento de áudio em consultas, integração Azure Cognitive Services) e quatro dos cinco objetivos (detecção precoce de riscos materno-ginecológicos, bem-estar psicológico, uso de cloud, detecção de anomalias por caso). O detector de vídeo usa YOLOv8 customizado sobre CholecSeg8k para identificar instrumentos de cirurgia laparoscópica (`Grasper`, `L-hook Electrocautery`). O pipeline de áudio combina `faster-whisper`, `librosa` e `wav2vec2`. RAG sobre nove diretrizes clínicas brasileiras (Ministério da Saúde, FEBRASGO, INCA) enriquece o relatório. LLM é Azure OpenAI GPT-4.1-mini via AI Foundry, com fallback determinístico offline. A UI é Gradio Blocks publicada em Hugging Face Spaces.

**Métricas-chave:** YOLOv8 customizado atinge mAP@50 = **0.989** e mAP@50-95 = **0.882** no test split do CholecSeg8k (808 imagens, 920 instâncias), com desempenho balanceado entre as 3 classes treinadas (`grasper`, `l_hook_electrocautery`, `blood`). Treino completo em ~17 min em GPU A100 (40 epochs, YOLOv8n, batch 320, imgsz 640). Pipeline multimodal cobre 3 das 4 funcionalidades do enunciado (vídeo, áudio e Azure Cognitive Services) e 4 dos 5 objetivos listados.

**Demo pública:** o sistema está disponível em [huggingface.co/spaces/joalissonborges/medica-ia-multimodal](https://huggingface.co/spaces/joalissonborges/medica-ia-multimodal), com as quatro abas (`Vídeo`, `Áudio`, `Multimodal`, `Auditoria`) e a aba `Configurações` operacionais sob o status `Running` do Hugging Face Spaces.

![Header do Hugging Face Space rodando com status Running e as quatro abas da UI Gradio visíveis](figures/screenshots/hf_space_header.png)

---

## 2. Contexto e Objetivo

O sistema é independente da entrega da Fase 3, com arquitetura voltada à análise multimodal e à detecção de anomalias em saúde da mulher (ADR-001).

O objetivo é demonstrar uma solução funcional que:

1. Analisa vídeos clínicos com detecção customizada via YOLOv8.
2. Analisa áudios de consultas com transcrição, features acústicas e classificação de emoção.
3. Recupera diretrizes ginecológicas e obstétricas via RAG.
4. Detecta anomalias e gera alertas estruturados por nível.
5. Integra Azure Cognitive Services como camada de serviços gerenciados.

### 2.1 Métricas de sucesso

1. App Gradio rodando fim a fim com as três modalidades integradas.
2. YOLO custom treinado com mAP aceitável para o caso de uso.
3. Pelo menos três cenários demonstrando alertas distintos (normal, moderado, crítico).
4. Integração Azure visível na demo (não simulada).
5. Relatório técnico com métricas, exemplos e diagramas (este documento).
6. Repositório com README e estrutura limpa.

---

## 3. Escopo e Funcionalidades Cobertas

### 3.1 Funcionalidades do enunciado

| # | Funcionalidade | Status | Observação |
|---|---|---|---|
| 1 | Análise de vídeos clínicos (YOLOv8 customizado) | Coberta | Instrumentos cirúrgicos laparoscópicos (ADR-012) |
| 2 | Processamento de gravações de voz em consultas | Coberta | Pipeline Whisper + librosa + wav2vec2 |
| 3 | Monitoramento de sinais vitais | Adiada | ADR-014, fora do escopo |
| 4 | Integração com Azure Cognitive Services | Coberta | Speech, Language, OpenAI (texto, vision, audio) |

### 3.2 Objetivos do enunciado

| # | Objetivo | Status |
|---|---|---|
| 1 | Detecção precoce de riscos em saúde materna e ginecológica | Coberto |
| 2 | Monitoramento de bem-estar psicológico feminino | Coberto |
| 3 | Uso de serviços em nuvem para ampliar capacidade | Coberto |
| 4 | Detecção de anomalias em tempo real | Coberto (processamento por caso, alerta imediato após análise) |
| 5 | Detecção de sinais de violência doméstica | Fora do escopo (risco ético, sem dataset rotulado) |

---

## 4. Arquitetura Multimodal

### 4.1 Visão Geral

O sistema recebe vídeo, áudio e texto opcional, processa cada modalidade em pipeline próprio, funde resultados em orquestrador único e gera saídas estruturadas (relatório clínico, alerta categorizado, log de auditoria).

```mermaid
flowchart TD
    A[Inputs] --> B[Video Pipeline]
    A --> C[Audio Pipeline]
    A --> D[Text Pipeline / RAG]
    B --> E[Multimodal Fusion]
    C --> E
    D --> E
    E --> F[Anomaly Detection]
    F --> G[Report Generator]
    F --> H[Alert System]
    F --> I[Audit Log]
```

Três artefatos coexistem com responsabilidades distintas:

| Artefato | Natureza | Quando entra |
|---|---|---|
| Dataset CholecSeg8k | 8080 frames anotados | Offline, antes do runtime. Treina `yolo_v1.pt` no Colab uma única vez |
| PDFs de diretrizes (RAG) | 9 documentos clínicos PT-BR | Runtime, recuperados por similaridade no Chroma |
| LLM Azure OpenAI | Modelo generativo | Runtime, etapa final. Apenas redige relatório sobre evidências já consolidadas |

```mermaid
flowchart LR
    subgraph Offline["Offline (uma vez, Colab)"]
        DS[CholecSeg8k<br/>8080 frames]
        TR[notebooks/train_yolo_colab.ipynb]
        WT[yolo_v1.pt]
        DS --> TR --> WT
    end

    subgraph Indexing["Indexacao (uma vez por release)"]
        PDF[9 PDFs<br/>MS/FEBRASGO/INCA]
        ING[scripts/build_rag_index.py]
        CHR[Chroma<br/>bge-m3]
        PDF --> ING --> CHR
    end

    subgraph Runtime["Runtime (por caso)"]
        VID[Video + Audio + Texto]
        YOLO[YOLO detector]
        AU[Pipeline audio]
        TX[Pipeline texto]
        RAG[RAG retriever]
        RULES[Anomaly Rules<br/>+ Statistical]
        LVL[Level<br/>normal/moderate/critical]
        LLM[Azure OpenAI<br/>GPT-4.1-mini]
        REP[Relatorio clinico]

        VID --> YOLO
        VID --> AU
        VID --> TX
        YOLO --> RULES
        AU --> RULES
        TX --> RULES
        RULES --> LVL
        LVL --> RAG
        CHR --> RAG
        RAG --> LLM
        LVL --> LLM
        LLM --> REP
    end

    WT -.carrega.-> YOLO
```

### 4.2 Pilar Vídeo

Pipeline em `src/video/pipeline.py`. Entrada: caminho de vídeo. Saída: lista de `VideoEvent` com detecções por frame.

#### Classificação de cena e gating por pilares

Antes de processar frame a frame, o pipeline classifica o tipo de cena via `src/video/scene_classifier.py`. A classificação amostra 5 frames distribuídos uniformemente e aplica duas heurísticas: detecção de face (MediaPipe Face Detection, com lazy import e fallback silencioso) e assinatura de saturação HSV. A saturação media por frame e o sinal primário de discriminação; o hue foi descartado como critério porque varia significativamente entre vídeos de cirurgia conforme o encoding MP4. O threshold de saturação (65) separa consultas clínicas (média 38-54 nos vídeos de demo) de cirurgias laparoscópicas (média 74-103).

O resultado classifica a cena em `SURGERY`, `CONSULTATION`, `MIXED` ou `UNKNOWN`. Essa classificação ativa o gating por pilares:

| Cena         | YOLO (instrumentos) | Emocao + linguagem corporal |
|--------------|---------------------|-----------------------------|
| SURGERY      | rodando             | pulado                      |
| CONSULTATION | pulado              | rodando                     |
| MIXED        | rodando             | rodando                     |
| UNKNOWN      | rodando             | rodando                     |

O gating evita dois problemas documentados: (1) em cenas de consulta, o YOLO customizado (treinado em laparoscopia) produz falsos positivos ao classificar equipamentos de exame como instrumentos cirúrgicos; (2) em cenas de cirurgia, a inferência de emoção e linguagem corporal não tem sinal útil porque a paciente não é visível.

A frequência de classificação de emoção facial é reduzida por padrão (`emotion_every_n_samples=3`), executando a cada terceiro frame amostrado. Isso reduz chamadas ao classificador de emoção em aproximadamente 3x sem perda significativa de fidelidade temporal.

#### Etapas de processamento por frame

1. Extração de frames com OpenCV (1 a 5 fps, configurável).
2. **YOLOv8 customizado** executado em frames de cenas SURGERY, MIXED ou UNKNOWN (interface model-agnostic em `src/video/detector.py`). Modelo treinado em **3 classes**: `grasper` (id 0), `l_hook_electrocautery` (id 1) e `blood` (id 2). As duas primeiras cobrem o requisito "Instrumentos cirúrgicos ginecológicos" do enunciado; a terceira cobre "Sinais de complicações em cirurgias ginecológicas" e dispara trigger `critical` no pipeline de anomalia quando sangramento é detectado.
3. **Emoção e linguagem corporal** via GPT-vision multimodal (`src/video/azure_openai_vision.py`, Azure OpenAI GPT-4o) executada a cada `emotion_every_n_samples` frames amostrados em cenas CONSULTATION, MIXED ou UNKNOWN. Além do `label` de emoção, o modelo emite agora o campo `body_language` em `EmotionScore`, classificando a linguagem corporal predominante em uma de cinco categorias (`tranquila`, `tensa`, `retraida`, `agitada`, `indefinida`). O modelo interpreta postura, gestos e expressão facial em conjunto, com a vantagem de identificar a paciente pelo contexto semântico (resolve o problema de cenas com médico + paciente simultâneos), sem depender de heurística geométrica sobre keypoints. O fallback local (`FER`) cobre apenas a emoção facial e deixa `body_language` em `None`.
4. Agregação em estrutura `VideoEvent` por frame.

A UI da aba Vídeo exibe progresso detalhado via `gr.Progress`, uma galeria com os 4 frames com maior densidade de detecções (com bounding boxes) e uma tabela "Eventos por janela" que agrega os eventos detectados em janelas de 5 segundos.

**Limite de upload (`ui/limits.py`):** vídeos até 60 MB e 120 segundos, validados via `ffprobe` antes de despachar para o pipeline. Acima disso, a UI rejeita com mensagem clara, evitando OOM em casos abusivos.

### 4.3 Pilar Áudio

Pipeline em `src/audio/pipeline.py`. Entrada: caminho de áudio. Saída: `AudioAnalysis` com transcrição, features acústicas, emoção, sentimento e frases-chave.

Etapas, reportadas via callback `progress(frac, desc)` que a aba **Áudio** repassa ao `gr.Progress` da UI (`show_progress="minimal"`):

1. **Transcrevendo audio... (5%)** via `faster-whisper` local (modelo `small`) ou Azure Speech (toggle por `USE_CLOUD_TRANSCRIPTION` no `.env`). O caminho cloud usa **reconhecimento contínuo** (`start_continuous_recognition_async`), com handlers de `recognized`, `session_stopped` e `canceled` que acumulam segmentos com timestamps. Esse modelo de execução substitui o `recognize_once_async` original, que só capturava a primeira frase antes de parar na primeira pausa: o reconhecimento contínuo processa o áudio inteiro (validado em trechos acima de 60 segundos) e converte os campos `offset`/`duration` de 100-ns ticks para milissegundos por segmento.
2. **Extraindo features acusticas... (45%)** com `librosa`: jitter, shimmer, energia RMS, pitch médio/desvio e MFCC.
3. **Classificando emocao vocal... (60%)** por um dos dois caminhos:
   - **Padrão (fallback local):** `wav2vec2-base-superb-er` (pré-treinado em RAVDESS, atores americanos em inglês).
   - **Multimodal cloud:** `AzureOpenAIAudioEmotion` em `src/audio/azure_openai_audio.py` envia o WAV em base64 + prompt JSON-mode para um deployment de modelo de áudio no Azure AI Foundry (ex.: `gpt-4o-mini-audio-preview`) e recebe a classificação no schema do `EmotionScore`. Quando `AZURE_OPENAI_AUDIO_DEPLOYMENT` está vazio, cai automaticamente no wav2vec2. Detalhes da motivação dessa decisão em 9.1.
4. **Analisando sentimento e frases-chave... (85%)** via Azure Language sobre a transcrição. Quando o serviço retorna o meta-label `mixed` (texto com partes positivas e negativas, sem confidence próprio), o cliente preenche `confidence = max(scores["positive"], scores["negative"])` em `src/audio/azure_language.py`. A aba **Áudio** expõe esse caso explicitamente: o KPI Sentimento mostra o hint `pos: X% / neg: Y%` e o sumário em markdown imprime `mixed (pos: X% / neg: Y%)`, evitando KPI com 0% que sugeriria ausência de sinal.

Ao final (100%, `Concluido.`) o pipeline devolve o `AudioAnalysis` agregado. A aba **Áudio** também reseta todos os outputs (status, KPIs, sumário, gráfico, JSON bruto) via callback `_on_clear` ligado ao evento `clear` do componente de áudio, garantindo estado consistente quando o usuário remove o arquivo antes de uma nova análise. O layout da aba foi reorganizado em blocos empilhados verticalmente (Entrada, Resumo, Transcrição + emoção + sentimento, Gráfico + features, JSON bruto), priorizando largura total para a transcrição e o gráfico de features.

Estratégia de dados híbrida (ADR-013): Azure TTS PT-BR Neural gera áudios scriptados como gold standard para a demo; CORAA-SER valida que o classificador não overfita ao timbre sintético.

**Limite de upload (`ui/limits.py`):** áudios até 30 MB e 120 segundos.

### 4.4 Pilar RAG (Diretrizes Clínicas)

Pipeline em `src/rag/`. Entrada: query textual derivada do contexto multimodal. Saída: lista de chunks relevantes.

Etapas:

1. Indexação offline via `scripts/build_rag_index.py` com chunking dos PDFs.
2. Embeddings `BAAI/bge-m3` (multilíngue, CPU, cerca de 1 GB).
3. Vector store Chroma persistido em `data/processed/chroma` (ADR-008).
4. Retriever LangChain com filtros opcionais por fonte e seção.
5. **Threshold de similaridade (`min_score=0.3` por default)** em `src/rag/retriever.py`: chunks com score abaixo do limiar são descartados. Quando a query toca tema fora dos 9 PDFs indexados (ex.: endometriose, SOP, mioma, infertilidade, menopausa, câncer de ovário), o retriever retorna lista vazia. O LLM é instruído pelo system prompt a sinalizar explicitamente "tema fora das diretrizes indexadas" em vez de redigir recomendações genéricas com chunks irrelevantes.
6. **Recuperação multi-query por eixo temático** (ADR-017) em `Orchestrator._retrieve_context`. Em vez de concatenar contexto, transcrição e triggers numa única query, o orquestrador detecta eixos ativos (`saude_mental`, `violencia`, `reprodutivo`, `rastreio`) via palavras-chave no haystack e dispara uma query focada por eixo, somadas à query base clínica. Os resultados são fundidos via interleaving round-robin com dedup por `chunk_id` em `Retriever.multi_search`, retornando até `rag_top_k=6` chunks distintos. Isso evita que sinais clínicos compitam com sinais emocionais no mesmo vetor e impede que o PDF mais volumoso (`manual_ms_prenatal`) domine o ranking em casos sem contexto gestacional.

Documentos indexados:

| Slug | Documento | Fonte |
|---|---|---|
| `manual_ms_prenatal` | Manual MS Pré-natal de baixo e alto risco | MS |
| `febrasgo_preeclampsia` | Diretriz FEBRASGO Pré-eclâmpsia | FEBRASGO |
| `ms_gestacao_alto_risco` | Manual MS Gestação de Alto Risco | MS |
| `inca_cancer_mama` | INCA Detecção Precoce do Câncer de Mama | INCA |
| `inca_cancer_colo_utero` | INCA Detecção Precoce do Câncer do Colo do Útero | INCA |
| `ms_pcdt_ist_violencia` | MS PCDT IST e Atenção a Vítimas de Violência | MS |
| `ms_parto_normal` | MS Diretrizes de Atenção ao Parto Normal | MS |
| `cab26_saude_sexual_reprodutiva` | Caderno AB nº 26 Saúde Sexual e Reprodutiva | MS |
| `cab34_saude_mental` | Caderno AB nº 34 Saúde Mental | MS |

### 4.5 Detecção de Anomalia

Pipeline em `src/anomaly/`. Entrada: agregação de `VideoAnalysis` e `AudioAnalysis`. Saída: `AnomalyResult` com nível, triggers acionados, explicação e ações recomendadas.

Camadas:

1. **Regras clínicas explícitas** em `src/anomaly/rules.py`. Exemplos:
   - `rule_surgical_instrument_presence`: detecção de Grasper ou L-hook em mais de N frames consecutivos eleva o nível para `moderate`, sinalizando procedimento invasivo em curso (semântica nova introduzida pela ADR-012, em que detecção de instrumento é estado NORMAL de cirurgia laparoscópica).
   - `rule_bleeding_detected`: detecção da classe `blood` em mais de 2 frames consecutivos OU em mais de 5% do vídeo dispara trigger **`critical`** com mensagem específica de protocolo de hemorragia. Threshold de confiança mais baixo (`0.4`) que o de instrumento (`0.5`) porque sangue tem forma irregular e o YOLO classifica com confiança menor.
   - `rule_vocal_distress`, `rule_facial_distress`, `rule_negative_sentiment`, `rule_critical_terms`: triggers por modalidade.
2. **Isolation Forest** em `src/anomaly/statistical.py` sobre features agregadas (energia da voz, frequência de movimento, jitter, shimmer).
3. **Classificador final** em `src/anomaly/classifier.py` combina rules e statistical com prioridade para regras críticas.

Níveis possíveis: `normal`, `moderate`, `critical`.

**Inconsistência multimodal como achado clínico:** o system prompt do `src/report.py` instrui explicitamente o LLM a destacar casos em que o texto e a voz divergem (ex.: paciente verbaliza "estou bem" mas tom é monótono e expressão facial mostra distress). Esse padrão de incongruência entre canais é clinicamente relevante em situações de minimização de sintomas emocionais e justifica o investimento em pipeline multimodal versus análise text-only.

### 4.6 Orquestrador e Auditoria

Orquestrador em `src/orchestrator.py`, função `process_case(input: CaseInput) -> CaseOutput`. Ponto único de entrada que recebe vídeo, áudio e contexto opcional, chama os pipelines, funde resultados, dispara anomaly detection, gera relatório via `src/report.py` e alerta via `src/alert.py`.

Auditoria em `src/audit.py` registra em SQLite: timestamp, case_id, modalidades processadas, modelos usados, custo Azure estimado, anomalia detectada. Acessível pela aba **Auditoria** da UI.

UI Gradio Blocks em `app.py` com quatro abas: **Vídeo**, **Áudio**, **Multimodal** e **Auditoria**. Tema dark indigo, componentes reutilizáveis em `ui/components.py`.

---

## 5. Modelos e Datasets

### 5.1 Modelos de Visão

| Modelo | Origem | Uso |
|---|---|---|
| YOLOv8n base | Ultralytics | Stub durante desenvolvimento (ADR-011) |
| YOLOv8n custom | Treino próprio sobre CholecSeg8k | Detector de instrumentos cirúrgicos (ADR-012) |
| FER | github/justinshenk/fer | Emoção facial (CPU, fallback local) |

### 5.2 Modelos de Áudio

| Modelo | Origem | Uso |
|---|---|---|
| faster-whisper small | SYSTRAN/faster-whisper | Transcrição local PT-BR (fallback) |
| Azure Speech (continuous) | Azure Cognitive Services | Transcrição cloud com reconhecimento contínuo (toggle `USE_CLOUD_TRANSCRIPTION`) |
| wav2vec2 emotion | superb/wav2vec2-base | Classificação de emoção (fallback local) |

### 5.3 Modelos de Texto e LLM

| Modelo | Origem | Uso |
|---|---|---|
| BAAI/bge-m3 | Hugging Face | Embeddings RAG multilíngue |
| Azure OpenAI GPT-4.1-mini | Azure AI Foundry | Geração de relatório clínico |

### 5.4 Datasets

| Dataset | Fonte | Uso | Licença |
|---|---|---|---|
| CholecSeg8k | HF `minwoosun/CholecSeg8k`, Hong et al. (arXiv 2012.12463) | Treino YOLO custom | CC BY-NC-SA 4.0 |
| CORAA-SER | HF `alefiury/CORAA-SER` | Validação cruzada wav2vec2 em PT-BR espontâneo | Termos HF |
| AVOS Open Surgery | research.bidmc.org | Referência comparativa em demo | Pesquisa |
| Geeky Medics (YouTube CC) | YouTube | Consulta clínica simulada para demo | CC |

PDFs de diretrizes do MS, FEBRASGO e INCA são públicos para uso educacional. Áudios reais de pacientes não são utilizados (LGPD).

---

## 6. Integração Azure

Quando as chaves estão preenchidas no `.env`, o sistema usa os serviços gerenciados abaixo; sem elas, cada pilar cai num fallback local equivalente.

| Serviço | Função | Onde é usado | Fallback offline |
|---|---|---|---|
| **Azure OpenAI** (GPT-4.1-mini, AI Foundry) | Gera o relatório clínico final em markdown | `src/llm/azure_openai.py`, `src/report.py` | Relatório determinístico em markdown construído a partir das triggers |
| **Azure Speech** | Transcrição de áudio (reconhecimento contínuo, captura áudios acima de 60s com timestamps por segmento) + TTS para gerar voz PT-BR | `src/audio/transcriber.py` (toggle `USE_CLOUD_TRANSCRIPTION`) | `faster-whisper` local (modelo `small`) |
| **Azure Language** | Análise de sentimento + key phrases na transcrição. Quando o label é `mixed`, expõe `max(positive, negative)` como confidence e a UI mostra a distribuição pos/neg | `src/audio/azure_language.py` | Pular esse pilar (não há substituto local equivalente) |
| **Azure OpenAI Vision** (GPT-4o multimodal) | Estado emocional via linguagem corporal + face em vídeo, sem o viés de FER-2013 | `src/video/azure_openai_vision.py` (ativado por `AZURE_OPENAI_VISION_DEPLOYMENT`) | `FER` local (Py 3.12) |

A aba **Configurações** da UI mostra imediatamente quais serviços estão ativos (cloud) ou em fallback (local).

A escolha por Azure puro (em vez de AWS ou híbrido) está formalizada na ADR-004: o vídeo demo da entrega lista "Integração dos serviços Azure" como obrigatória, e tratar como requisito é mais seguro que apostar em interpretação literal do enunciado.

---

## 7. Treino do YOLO Custom

### 7.1 Dataset (CholecSeg8k, ADR-012)

Após pesquisa empírica em Roboflow Universe, Kaggle, Hugging Face, PhysioNet, Synapse e repositórios acadêmicos, nenhuma fonte sustentou o alvo original "sangramento anômalo" no domínio ginecológico com reprodutibilidade aceitável. As alternativas avaliadas ao longo do projeto:

| Dataset | Veredito |
|---|---|
| WCEBleedGen | Sangramento gastrointestinal (cápsula endoscópica), domínio errado |
| BUSI | Ultrassom mamário diagnóstico, sem sangramento |
| Dresden Surgical Anatomy | Ginecológico real, mas 19 GB e licença restritiva inviabilizam Colab/HF Spaces |
| m2caiseg | Mesmo domínio do CholecSeg8k (colecistectomia), 307 imagens, escala insuficiente |
| AutoLaparo-T3 | Histerectomia laparoscópica real (único dataset ginecológico público), 1.800 frames com anotação pixel-wise. Acesso solicitado e autorizado, mas link de download fornecido pela equipe estava inacessível no momento da implementação. Licença restrita a pesquisa acadêmica |
| **CholecSeg8k** | 3.1 GB, anônimo via HF, classes adequadas, licença CC BY-NC-SA 4.0 |

A decisão foi pivotar para CholecSeg8k (Hong et al., arXiv 2012.12463): 8080 frames anotados de colecistectomia laparoscópica, contendo `Grasper` e `L-hook Electrocautery`, instrumentos idênticos aos usados em cirurgia ginecológica laparoscópica. A transferência de domínio é justificada clinicamente pela técnica (mesmo trocarte, mesma pinça, mesmo eletrocautério).

Splits utilizados: 5656 train, 1616 val, 808 test.

Conversão de máscaras de segmentação para bounding boxes YOLO em `data/processed/cholecseg8k_yolo/`.

### 7.2 Configuração de Treino

| Parâmetro | Valor |
|---|---|
| Arquitetura | YOLOv8n (3,006,233 parâmetros, 8.1 GFLOPs) |
| Classes | 3: `grasper` (id 0), `l_hook_electrocautery` (id 1), `blood` (id 2) |
| Épocas | 40 (patience=10 configurado; treino completou os 40 sem disparar early stopping) |
| Batch size | 320 (A100 40 GB, ~40.7 GB de uso) |
| Imagem (imgsz) | 640 |
| Workers (dataloader) | 16 |
| Otimizador | SGD (default Ultralytics) com momentum 0.937 |
| Learning rate inicial | 0.01 (default) com warmup |
| Augmentations | Mosaic, HSV, flip, mixup, translate, scale (default Ultralytics) |
| Mosaic disabled | últimos 10 epochs (refino com imagens reais) |
| Hardware | Google Colab Pro+ GPU A100-SXM4-40GB |
| Tempo total de treino | ~17 minutos (avg ~25s por epoch) |
| Notebook | `notebooks/train_yolo_colab.ipynb` |

### 7.3 Métricas Finais

Validação no **test split** (808 imagens, 920 instâncias, nunca vistas pelo modelo durante treino ou validação):

| Métrica | Valor (test) |
|---|---|
| mAP@50 | **0.9893** |
| mAP@50-95 | **0.8816** |
| Precision | **0.9817** |
| Recall | **0.9695** |

Por classe (test split):

| Classe | Instâncias | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| `grasper` | 615 | 0.984 | 0.990 | **0.993** | **0.929** |
| `l_hook_electrocautery` | 234 | 0.976 | 0.966 | **0.992** | **0.855** |
| `blood` | 71 | 0.985 | 0.952 | **0.982** | **0.860** |

**Observações:**

- Modelo aprendeu a classe minoritária (`blood`, apenas 71 instâncias no test) com qualidade comparável às majoritárias, demonstrando boa generalização mesmo com desbalanceamento.
- 154 frames de "background" no test (sem nenhum label) confirmam a calibração de precision (0.98): o modelo não inventa detecções em frames vazios.
- Speed: **1.8 ms por imagem em A100** (0.1 ms preprocess + 0.9 ms inference + 0.8 ms postprocess). Em CPU local (deploy HF Spaces) espera-se ~150-300 ms por imagem mantendo viabilidade pra demo.

Curvas, matriz de confusão e exemplos com bounding boxes detectadas pelo modelo final estão em `MyDrive/medica-ia/yolo_runs/surgical_instruments/`. Os outputs ficam preservados no notebook (`notebooks/train_yolo_colab.ipynb`) pra reprodutibilidade.

### 7.5 Validação Visual

A célula 4.5 do notebook seleciona automaticamente uma sequência consecutiva do test split contendo a classe `blood`, roda inferência do `best.pt` frame a frame, desenha bounding boxes anotadas e empacota um MP4 de demonstração:

- Output original (Colab): `MyDrive/medica-ia/yolo_runs/surgical_instruments/validation_blood_detected.mp4`
- Versionado no repo em [`reports/figures/validation_blood_detected.mp4`](figures/validation_blood_detected.mp4) (332 KB).
- Conteúdo: 11 frames consecutivos da sequência `video01_28660` (CholecSeg8k test split) com Blood visível, anotados com bboxes em `red` (Blood), `lime` (Grasper) e `magenta` (L-hook), inferidas pelo modelo final.
- Esse MP4 serve como evidência visual direta da capacidade do modelo de detectar sangramento intraoperatório.

### 7.4 Discussão

**Convergência e qualidade do treino.** O modelo alcançou mAP@50 = 0.989 no test split, com performance balanceada entre as 3 classes (variação de 0.982 a 0.993 entre `blood`, `l_hook_electrocautery` e `grasper`). Mesmo `blood`, a classe mais rara do dataset (apenas 71 instâncias no test, contra 615 de `grasper`), atingiu mAP@50 = 0.982, demonstrando que o desbalanceamento de classes não comprometeu o aprendizado. Convergência rápida (mAP@50 > 0.9 a partir do epoch 12), sem indícios visíveis de overfitting nos 40 epochs.

**Aderência da transferência de domínio.** Esses números refletem desempenho **dentro do domínio de treino** (colecistectomia laparoscópica). Em vídeo real de cirurgia ginecológica (histerectomia, salpingectomia, miomectomia), espera-se:

- **Recall consistente** para `grasper` (instrumento físico idêntico em ambos os procedimentos: mesma marca, mesma forma, mesma articulação).
- **Recall consistente** para `blood` (sangramento tem aparência similar em qualquer cavidade peritoneal: cor de sangue não muda entre procedimentos).
- **Recall reduzido** para `l_hook_electrocautery` (instrumento existe em ginecologia mas é menos frequente; Harmonic Scalpel e LigaSure dominam o papel de corte/coagulação).
- **Confiança média reduzida** devido a diferenças de background tecidual (útero/trompas/ligamentos rosados vs fígado/vesícula esverdeados).
- **Lacuna de cobertura** para instrumentos específicos de ginecologia que não estão nas 3 classes treinadas (Harmonic Scalpel, LigaSure, tesoura laparoscópica, endoclip applier, suction/irrigator). Esses instrumentos aparecem em 50-70% dos frames de uma histerectomia típica e o modelo os ignora silenciosamente (saída vazia naquela região, sem falso positivo).

A validação empírica dessa transferência (rodar `best.pt` em vídeo ginecológico real do YouTube CC) está na seção 8.

**Semântica de risco no anomaly classifier.** A detecção de `grasper`/`l_hook_electrocautery` em frames consecutivos dispara trigger `moderate` (registra "procedimento invasivo em curso"), nunca `critical` isoladamente: a presença de instrumental cirúrgico é estado esperado em cirurgia laparoscópica, não anomalia. Já a classe `blood` é tratada diferente: detecção em mais de 2 frames consecutivos OU em mais de 5% do vídeo dispara trigger **`critical`** com mensagem específica de protocolo de hemorragia (`src/anomaly/rules.py:rule_bleeding_detected`). Threshold de confiança menor (0.4) para `blood` que para instrumentos (0.5) reflete a forma irregular do sangue, que tende a ser classificado com confiança levemente menor pelo YOLO.

**Roadmap de expansão** (seção 9.2): pipeline híbrido com GPT-4o-vision pra cobrir os instrumentos não-treinados via identificação semântica em linguagem natural, mantendo o YOLO como detector estruturado das 3 classes core.

---

## 8. Resultados em Cenários Reais

Cada cenário foi rodado fim a fim pela UI Gradio. Os artefatos (vídeo, áudio, prints) estão em `data/examples/` e nos prints abaixo. O `manifest.json` da pasta de exemplos lista os **6 casos** abaixo, cobrindo os três níveis esperados (normal, moderado, crítico) tanto em consultas clínicas quanto em cirurgias laparoscópicas.

### 8.1 Consulta Normal

Primeira consulta clínica de avaliação geral, paciente refere dor torácica e alteração de pressão arterial. Material: vídeo de consulta simulada (acadêmica) em PT-BR.

| Modalidade | Entrada | Saída resumida |
|---|---|---|
| Vídeo | `data/examples/consultas/consulta_clinica_geral/video.mp4` | <!-- TODO --> |
| Áudio | `data/examples/consultas/consulta_clinica_geral/audio.wav` (extraído do vídeo) | <!-- TODO --> |
| Texto | Contexto clínico de avaliação geral com queixa torácica | <!-- TODO --> |
| Nível final | `normal` esperado | <!-- TODO --> |

<!-- TODO: print da aba Multimodal com relatório clínico gerado -->

### 8.2 Consulta Moderada (Depoimento de paciente)

Depoimento real de paciente sobre rastreio e diagnóstico de alteração mamária. Voz com tom emocional perceptível.

**Caso audio-only.** O vídeo original apresenta agulhas e outros instrumentos de exame que o YOLO custom classifica erroneamente como procedimento cirúrgico em curso (falso positivo `rule_surgical_instrument_presence`). Como o sinal clinicamente relevante deste caso está na voz (tom emocional, transcrição da experiência da paciente), o vídeo foi descartado da execução e apenas o áudio é processado. Esse caso ilustra uma decisão consciente de filtrar modalidades quando o conteúdo visual gera mais ruído que sinal.

| Modalidade | Entrada | Saída resumida |
|---|---|---|
| Vídeo | descartado (falso positivo do YOLO no conteúdo do exame) | n/a |
| Áudio | `data/examples/consultas/rastreio_mama/audio.wav` | <!-- TODO --> |
| Texto | Contexto sobre rastreio precoce de câncer de mama (alinhado com PDF INCA no RAG) | <!-- TODO --> |
| Nível final | `moderate` esperado | <!-- TODO --> |

Triggers acionados:

<!-- TODO: listar triggers -->

<!-- TODO: print da aba Multimodal -->

### 8.3 Consulta Moderada (Dermatológica com ansiedade)

Consulta dermatológica com queixa de manchas faciais e componente emocional acentuado. Caso útil para demonstrar **detecção de inconsistência multimodal**: voz com sinais de medo/tensão (`vocal_distress`, `vocal_strain`) enquanto a fala minimiza sofrimento ("não estou desesperada").

| Modalidade | Entrada | Saída resumida |
|---|---|---|
| Vídeo | `data/examples/consultas/dermatologica/video.mp4` | <!-- TODO --> |
| Áudio | `data/examples/consultas/dermatologica/audio.wav` | <!-- TODO --> |
| Texto | Contexto de dermatologia + ansiedade | <!-- TODO --> |
| Nível final | `moderate` esperado (pipeline corretamente não escalou para `critical` pois conteúdo real não é emergência) | <!-- TODO --> |

#### Demonstração visual da aba Multimodal

A figura 8.3.1 apresenta o formulário do caso já preenchido na aba Multimodal. As três modalidades estão ativas simultaneamente: vídeo de consulta dermatológica com paciente e profissional em sala equipada com EPI, áudio de aproximadamente 35 segundos com forma de onda visível no player, e Contexto clínico descrevendo paciente com manchas faciais hiperpigmentadas e componente emocional acentuado. O identificador `exemplo-dermatologica` etiqueta a execução para rastreio na auditoria, e o botão Processar caso fica pronto para acionar a orquestração dos pilares de vídeo, áudio e texto.

![Caso dermatológico preenchido na aba Multimodal](figures/screenshots/aba_multimodal_dermatologica_caso.png)

*Figura 8.3.1: card Caso clínico com vídeo de consulta, áudio de 35 s carregado com waveform, contexto textual descrevendo manchas faciais hiperpigmentadas com componente emocional e identificador `exemplo-dermatologica`.*

A figura 8.3.2 exibe o card Resultado consolidado após o processamento. O `AnomalyClassifier` atribui nível **MODERADO** a partir de três triggers oriundos exclusivamente do áudio, conforme os KPIs NIVEL=Moderado, TRIGGERS=3, MODALIDADES=3 (vídeo, áudio e texto) e DIRETRIZES=6 (chunks recuperados via RAG). A presença das três modalidades ativas demonstra que o gating de cena classificou o vídeo como CONSULTATION, suprimindo o YOLO custom e direcionando a análise visual para o `gpt-4o` multimodal (emoção facial e linguagem corporal). Case ID `case-9f2619d72072` e Audit ID `49` identificam o registro persistido em SQLite.

![Resultado consolidado do caso dermatológico](figures/screenshots/aba_multimodal_dermatologica_resultado.png)

*Figura 8.3.2: card Resultado consolidado com badge **MODERADO**, KPIs NIVEL=Moderado, TRIGGERS=3, MODALIDADES=3 e DIRETRIZES=6, com Case ID `case-9f2619d72072` e Audit ID `49`.*

A figura 8.3.3 mostra o relatório clínico gerado pelo Azure OpenAI `gpt-4.1-mini` (via AI Foundry). As seções Resumo, Achados, Diretrizes Aplicáveis e Recomendações seguem o contrato do `src/report.py:SYSTEM_PROMPT_PT_BR`. Os Achados registram emoção vocal `medo` com confiança 0.85, tensão vocal sustentada por jitter e shimmer elevados, energia baixa (RMS), além de verbalizações textuais de desespero ("não sei", "tô desesperada"). O ponto a destacar é o comportamento da regra 4 do prompt do sistema, que evita inventar inconsistência quando as modalidades convergem: o LLM declara explicitamente que não há divergência entre voz e fala, e que a convergência reforça a angústia moderada. Em Diretrizes Aplicáveis, o relatório cita `(cab34_saude_mental) 89 SAÚDE MENTAL` e `(cab34_saude_mental) 99 SAÚDE MENTAL 6.4` ao tratar atendimento de queixas de ansiedade e tristeza na Atenção Básica e seguimento compartilhado com psiquiatria, além de `(ms_pcdt_ist_violencia) 53` com foco em manejo integral. A regra 5 do prompt impede citações fora dos chunks recuperados, preservando a auditabilidade. As Recomendações enumeram cinco ações: exames dermatológicos, avaliação clínica detalhada das lesões, monitoramento emocional, encaminhamento à saúde mental na Atenção Primária e orientação sobre uso correto de pomadas.

![Relatório clínico gerado pelo LLM para o caso dermatológico](figures/screenshots/aba_multimodal_dermatologica_relatorio.png)

*Figura 8.3.3: relatório clínico em Resumo, Achados, Diretrizes Aplicáveis e Recomendações, com convergência reconhecida entre texto ("tô desesperada") e voz (medo 0.85, tensão acústica), citações a `cab34_saude_mental` (páginas 89 e 99) e cinco recomendações enumeradas.*

A figura 8.3.4 detalha o resumo de anomalia, a tabela de triggers e o conjunto de ações recomendadas. Três regras determinísticas do `RuleEngine` (`src/anomaly/rules.py`) sustentam o nível moderado: `audio.vocal_distress` (emoção `medo` com confiança 0.85), `audio.vocal_strain` (jitter 0.019 e shimmer 0.120 acima dos limiares) e `audio.low_energy` (RMS 0.0103 abaixo do limiar 0.015). Todos os triggers têm severidade Moderado e origem áudio, totalmente independentes do LLM. As três ações recomendadas pelas regras (conduzir entrevista focada em estado emocional, investigar fadiga, ansiedade ou dor, e avaliar exaustão, anemia ou sintomas depressivos) compõem a camada determinística e auditável da decisão.

![Resumo de anomalia e triggers do caso dermatológico](figures/screenshots/aba_multimodal_dermatologica_anomalia.png)

*Figura 8.3.4: painéis Resumo de anomalia (nível Moderado, três triggers, três ações recomendadas) e tabela de triggers com `audio.vocal_distress`, `audio.vocal_strain` e `audio.low_energy`, todos com severidade Moderado e origem áudio.*

A figura 8.3.5 apresenta o painel Diretrizes consultadas com os seis chunks recuperados pelo retriever RAG. Três dos seis chunks são do CAB 34 Saúde Mental (Ministério da Saúde), nas páginas 90, 95 e 100, ao lado de dois chunks de `inca_cancer_colo_utero` (p. 10 e p. 76) e um de `ms_pcdt_ist_violencia` (p. 54). Essa distribuição valida o desenho multi-query por eixo temático (ADR-017): o eixo `saude_mental` foi ativado pelas palavras "ansiedade", "tristeza" e "medo" no haystack consolidado a partir das saídas dos pilares, disparando uma query focada nas sources `cab34_saude_mental` e `manual_ms_prenatal`. Como a paciente não está em contexto gestacional, a denylist obstétrica excluiu `manual_ms_prenatal` da query base, e a allowlist por source garantiu que o eixo recuperasse exatamente os capítulos relevantes de Saúde Mental do CAB 34.

![Diretrizes consultadas no caso dermatológico](figures/screenshots/aba_multimodal_dermatologica_diretrizes.png)

*Figura 8.3.5: painel Diretrizes consultadas com os seis chunks recuperados: `inca_cancer_colo_utero` p. 10, `cab34_saude_mental` p. 100, `ms_pcdt_ist_violencia` p. 54, `cab34_saude_mental` p. 90, `inca_cancer_colo_utero` p. 76 e `cab34_saude_mental` p. 95.*

As cinco figuras 8.3.1 a 8.3.5 fecham a demonstração fim a fim do caso dermatológico moderado: o orquestrador processou as três modalidades de entrada, o gating de cena escolheu corretamente o backend visual cloud para um contexto de consulta, o `RuleEngine` produziu três triggers de áudio auditáveis, o retriever multi-query ativou o eixo `saude_mental` e recuperou três chunks do CAB 34, e o LLM reconheceu convergência entre modalidades e citou somente diretrizes presentes nos chunks recuperados.

### 8.4 Pré-natal (Acolhimento emocional)

Primeira consulta gestacional, trecho de acolhimento ao resultado positivo. Paciente expressa ansiedade e dúvidas sobre como comunicar a notícia ao parceiro e à família. Postura corporal levemente retraída, fala entrecortada, sinais de ansiedade situacional. Caso útil para demonstrar sinal vocal moderado em contexto não-emergencial.

| Modalidade | Entrada | Saída resumida |
|---|---|---|
| Vídeo | `data/examples/consultas/prenatal_acolhimento/video.mp4` | <!-- TODO --> |
| Áudio | `data/examples/consultas/prenatal_acolhimento/audio.wav` | Emoção vocal `medo` (85%), sentimento textual `mixed` (pos 28% / neg 65%), 8 segmentos transcritos em 60s |
| Texto | Contexto de pré-natal com ansiedade situacional | <!-- TODO --> |
| Nível final | `moderate` esperado | Badge **MODERADO** atribuído pela heurística da aba Áudio |

#### Demonstração visual da aba Áudio

A figura 8.4.1 apresenta o ponto de entrada da aba Áudio após o upload do arquivo de 60 segundos. O stepper Upload, Processar, Resultado sinaliza o estágio atual, e o painel Resumo recebe o badge **MODERADO** derivado da heurística local da aba. A legenda abaixo do badge explicita que essa classificação é restrita ao pilar áudio, remetendo o leitor à aba Multimodal para a decisão final integrada com os demais pilares.

![Resumo da aba Áudio para o caso de acolhimento pré-natal](figures/screenshots/aba_audio_prenatal_resumo.png)

*Figura 8.4.1: aba Áudio após upload do arquivo de 60 segundos, com player de forma de onda completa, botão de análise e badge **MODERADO** no painel Resumo.*

A identificação do caso gestacional combinada ao sofrimento psíquico apoia-se em três sinais convergentes: o canal vocal indica medo com 85% de confiança via `gpt-4o-mini-audio-preview` (ADR de migração para classificação multimodal cloud), o canal textual aponta sentimento misto via Azure Language e o canal acústico evidencia jitter e shimmer elevados em conjunto com energia RMS baixa, padrão consistente com fala entrecortada e tensão vocal.

![KPIs, badge de provider e transcrição completa](figures/screenshots/aba_audio_prenatal_transcricao.png)

*Figura 8.4.2: linha de KPIs com Duração 60.0 s, Emoção `medo` (gpt-audio-mini via Azure OpenAI), Sentimento `mixed` com hint `pos: 28% / neg: 65%` e 8 segmentos. Abaixo, a seção Transcrição, emoção e sentimento exibe o parágrafo completo da fala da paciente e a linha-resumo com emoção vocal, sentimento textual e frases-chave extraídas.*

O Azure Speech operando em modo de reconhecimento contínuo (ver seção 4.3 e ADR-016) entrega oito segmentos com `start_ms` e `end_ms` ao longo dos 60 segundos, recuperando o diálogo inteiro sem interrupção na primeira pausa, comportamento que o uso anterior de `recognize_once_async` não suportava. O rótulo `mixed` retornado pelo Azure Language reflete a ambivalência da fala, em que a paciente alterna nervosismo e tentativas de racionalização. A exibição da distribuição `pos: 28% / neg: 65%` no hint do KPI evita a leitura enganosa de confidence próxima de zero que ocorria antes da correção em `src/audio/azure_language.py`, dado que o meta-label `mixed` não possui confidence própria no contrato do serviço.

![Gráfico de features acústicas, tabela equivalente e JSON bruto](figures/screenshots/aba_audio_prenatal_features.png)

*Figura 8.4.3: gráfico de barras com pitch médio 174.7 Hz, pitch std 76.1, RMS 0.0164, ZCR 0.0577, jitter 0.0369 e shimmer 0.1321; tabela com os mesmos valores e suas unidades; início do JSON bruto exibindo `transcription` e a lista de `segments` com timestamps.*

As features acústicas exibidas no painel da figura 8.4.3 alimentam as regras determinísticas declaradas em `src/anomaly/rules.py`, em que `jitter > 0.02` e `shimmer > 0.1` disparam o trigger `voice_tension`. No caso atual, os valores observados (0.0369 e 0.1321) ultrapassam ambos os limiares, sustentando a classificação `moderate` da heurística da aba sem necessidade de acionar gatilhos de severidade crítica reservados a contextos clínicos de maior gravidade.

### 8.5 Cirurgia Normal (Laparoscopia sem intercorrência)

Procedimento laparoscópico em andamento, sem evento crítico visível. Demonstra que o pipeline diferencia cirurgia rotineira de cirurgia com complicação (próxima seção), mantendo o nível em `moderate` quando há apenas instrumental em uso e ausência de sangramento.

| Modalidade | Entrada | Saída resumida |
|---|---|---|
| Vídeo | `data/examples/cirurgias/rotina/video.mp4` | Detecção persistente de instrumental cirúrgico (`l_hook_electrocautery`, `grasper`) em 25 frames consecutivos, sem ocorrência da classe `blood` |
| Áudio | n/a (cirurgia sem voz do paciente) | n/a |
| Texto | Contexto de procedimento sem intercorrências (visualização da cavidade abdominal, uso de grasper e eletrocautério, conduta de continuidade) | Indicação textual de rotina alinhada com a saída visual |
| Nível final | `moderate` (trigger `rule_surgical_instrument_presence`) | Badge **MODERADO** atribuído corretamente, sem escalada para `critical` |

#### Demonstração visual da aba Multimodal

A figura 8.5.1 apresenta o formulário do caso já preenchido na aba Multimodal. O usuário carrega o vídeo laparoscópico padrão em `data/examples/cirurgias/rotina/video.mp4`, que exibe a cavidade abdominal com instrumental em uso e sem evento hemorrágico. No campo Contexto clínico, descreve um procedimento ginecológico de rotina com visualização da cavidade, manipulação por grasper e eletrocautério, ausência de sangramento e conduta de continuidade. O caso opera com duas modalidades ativas (vídeo e texto), uma vez que o campo cirúrgico não expõe fala da paciente e o áudio permanece vazio. O identificador `exemplo-cirurgia-rotina` rotula o caso para auditoria.

![Card Caso clínico com vídeo de cirurgia laparoscópica de rotina e contexto textual sem intercorrências](figures/screenshots/aba_multimodal_rotina_caso.png)

*Figura 8.5.1: card Caso clínico com vídeo carregado, contexto textual descrevendo procedimento sem intercorrências, campo de áudio sem arquivo e identificador `exemplo-cirurgia-rotina`.*

A figura 8.5.2 exibe o card Resultado consolidado após o processamento. O `AnomalyClassifier` atribui nível **MODERADO** sustentado por um único trigger oriundo da modalidade vídeo, conforme os KPIs NIVEL=Moderado, TRIGGERS=1, MODALIDADES=2 (vídeo e texto) e DIRETRIZES=0. A regra acionada é `surgical_instrument_presence` em nível Moderado, e a ausência de chunks recuperados ocorre porque o caso cirúrgico não ativa eixos humanos: conforme ADR-017, o orquestrador suprime a query base nesse cenário, evitando que o retriever retorne diretrizes tangencialmente relacionadas à ginecologia obstétrica indexada. Case ID `case-69af9a8169b6` e Audit ID `50` identificam o registro persistido.

![Resultado consolidado com badge moderado, 1 trigger e 0 diretrizes recuperadas](figures/screenshots/aba_multimodal_rotina_resultado.png)

*Figura 8.5.2: card Resultado consolidado com badge **MODERADO**, KPIs de nível, triggers, modalidades e diretrizes, além de Case ID `case-69af9a8169b6` e Audit ID `50`.*

A figura 8.5.3 mostra o relatório clínico gerado pelo Azure OpenAI `gpt-4.1-mini` (via AI Foundry) e estabelece o contraste explícito com o caso de sangramento intraoperatório (§8.6). Sem o trigger `bleeding_detected`, o nível permanece moderado e as recomendações do LLM convergem para cirurgia rotineira: monitoramento contínuo dos sinais vitais, controle de esterilidade e instrumental, avaliação pós-operatória imediata e orientação à paciente quanto ao seguimento. A regra de salvaguarda contra alucinação (regra 5 do `src/report.py:SYSTEM_PROMPT_PT_BR`) continua ativa: como o contexto RAG vem vazio, o LLM declara que "Nenhuma das diretrizes indexadas aborda diretamente este tema específico de cirurgia ginecológica rotineira com nível de risco moderado", sem fabricar referências a documentos inexistentes.

![Relatório clínico LLM com resumo, achados, declaração de ausência de diretrizes e quatro recomendações de cirurgia rotineira](figures/screenshots/aba_multimodal_rotina_relatorio.png)

*Figura 8.5.3: relatório clínico em Resumo, Achados, Diretrizes Aplicáveis e Recomendações, com declaração explícita de ausência de diretrizes aplicáveis indexadas e quatro ações enumeradas para o manejo de cirurgia rotineira.*

A figura 8.5.4 detalha o resumo de anomalia, a tabela de triggers e o painel de diretrizes consultadas. A única regra disparada é `video.surgical_instrument_presence` com severidade Moderado (instrumental cirúrgico detectado em 25 frames consecutivos, acima do limiar mínimo de 3), e o nível final acompanha esse máximo sem qualquer escalada para `critical`. As duas ações recomendadas vinculadas ao trigger (documentar instrumental utilizado no prontuário e correlacionar a detecção com a fase cirúrgica registrada) são genéricas e auditáveis, provêm do `RuleEngine` (`src/anomaly/rules.py`) e independem do LLM. O painel Diretrizes consultadas exibe "Sem diretrizes recuperadas no contexto.", evidenciando que o `AnomalyClassifier` diferencia corretamente cirurgia rotineira de cirurgia com evento crítico, sem inflar artificialmente o nível.

![Painéis resumo de anomalia, tabela de triggers com surgical_instrument_presence moderado e painel diretrizes vazio](figures/screenshots/aba_multimodal_rotina_anomalia.png)

*Figura 8.5.4: painéis Resumo de anomalia (nível moderado, um trigger, duas ações), tabela de triggers com `video.surgical_instrument_presence` (Moderado, origem vídeo) e painel Diretrizes consultadas exibindo "Sem diretrizes recuperadas no contexto.".*

As quatro figuras 8.5.1 a 8.5.4 fecham a demonstração fim a fim do caso cirúrgico de rotina e funcionam como contraponto direto ao caso crítico da seção seguinte: mesma cena cirúrgica laparoscópica, mas sem evento hemorrágico, com o pipeline distinguindo corretamente os níveis (`moderate` versus `critical`) e o LLM preservando a honestidade epistêmica ao declarar tema fora da cobertura das diretrizes indexadas.

### 8.6 Cirurgia Crítica (Sangramento intraoperatório)

Procedimento laparoscópico com sangramento em foco operatório. Vídeo gerado por `scripts/build_cirurgia_demo_video.py` filtrando frames do CholecSeg8k onde o YOLO custom v1 detectou a classe `blood` com confiança > 95%.

| Modalidade | Entrada | Saída resumida |
|---|---|---|
| Vídeo | `data/examples/cirurgias/sangramento/video.mp4` (30 frames @ 5 fps, ~6s) | `blood` e `l_hook_electrocautery` detectados em 6 frames consecutivos, dispara trigger crítico |
| Áudio | n/a (cirurgia sem voz do paciente) | n/a |
| Texto | Contexto de hemorragia intraoperatória + necessidade de hemostasia | Indicação textual de hemorragia reforça a saída visual |
| Nível final | `critical` (trigger `rule_bleeding_detected`) | Badge **CRÍTICO** atribuído corretamente |

A figura 8.6.1 apresenta o resumo da aba Vídeo após processamento. O classificador de cena identificou a entrada como `Cirurgia`, ativando o gating que suprime os pilares de emoção facial e linguagem corporal (rosto e corpo do paciente não estão visíveis em campo operatório laparoscópico). Os KPIs evidenciam 6 frames amostrados, 12 detecções totais distribuídas em 2 classes (`blood` e `l_hook_electrocautery`) e ausência intencional dos pilares afetivos.

![Resumo da aba Vídeo para o caso de sangramento intraoperatório](figures/screenshots/aba_video_sangramento_resumo.png)

*Figura 8.6.1: aba Vídeo com badge crítico, classificação de cena `Cirurgia` e KPIs do processamento. O badge "Emocao via: Azure GPT-vision" sinaliza o backend ativo para análise facial, enquanto os pilares de emoção e linguagem corporal aparecem como `n/a` por decisão do gating.*

A figura 8.6.2 detalha a galeria de frames anotados. As bounding boxes em magenta marcam o instrumento `l_hook_electrocautery` (confiança entre 91% e 92%), e as bounding boxes em vermelho marcam a classe `blood`. O modelo demonstra estabilidade temporal: as duas classes co-ocorrem ao longo da sequência amostrada, confirmando o cenário de cauterização ativa em região com sangramento.

![Galeria de frames com bounding boxes de blood e l_hook_electrocautery](figures/screenshots/aba_video_sangramento_deteccoes.png)

*Figura 8.6.2: quatro thumbnails do campo cirúrgico com detecções do YOLO custom v1. Bounding boxes em magenta indicam `l_hook_electrocautery` e em vermelho indicam `blood`. Abaixo da galeria, a timeline de eventos agrega as 12 detecções em 6 frames.*

A figura 8.6.3 mostra a tabela de eventos agregados por janelas de 5 segundos. A janela 0 a 4s concentra 5 frames com 10 detecções (`blood` 96% e `l_hook` 92% como classes dominantes); a janela 5 a 9s registra mais 1 frame com 2 detecções, mantendo a co-ocorrência. Esse padrão satisfaz o critério do `rule_bleeding_detected` (presença persistente de `blood` em frames consecutivos), justificando a atribuição automática do nível `critical`.

![Tabela de eventos por janela e JSON bruto da auditoria](figures/screenshots/aba_video_sangramento_eventos.png)

*Figura 8.6.3: agregação temporal das detecções em janelas de 5 segundos e início da visualização do JSON bruto exportado para a aba Auditoria. A persistência de `blood` em ambas as janelas alimenta o trigger crítico.*

A combinação das três figuras demonstra o caminho fim a fim: o YOLO custom v1, treinado em CholecSeg8k (seção 7), detecta de forma consistente as classes alvo no vídeo de demonstração; o gating de cena bloqueia pilares afetivos irrelevantes para o contexto cirúrgico; e o anomaly classifier converte a evidência visual em nível de risco `critical`.

#### Demonstração visual da aba Multimodal

A figura 8.6.4 apresenta o formulário do caso já preenchido na aba Multimodal. O usuário carrega o vídeo laparoscópico em `data/examples/cirurgias/sangramento/video.mp4` e descreve, no campo Contexto clínico, um quadro de sangramento intraoperatório com lesão vascular, monitoramento hemodinâmico e necessidade de intervenção hemostática. O campo de áudio permanece vazio porque não há fala da paciente em campo cirúrgico, e o identificador é deixado em branco para o caso ad hoc. O botão Processar caso fica pronto para disparar a orquestração dos pilares.

![Caso de sangramento intraoperatório preenchido na aba Multimodal](figures/screenshots/aba_multimodal_sangramento_caso.png)

*Figura 8.6.4: card Caso clínico com vídeo carregado, contexto textual descrevendo lesão vascular e manejo hemodinâmico, campo de áudio sem arquivo e identificador da paciente vazio.*

A figura 8.6.5 exibe o card Resultado consolidado após o processamento. O `AnomalyClassifier` atribui nível **CRÍTICO** somando dois triggers oriundos da modalidade vídeo, conforme os KPIs NIVEL=Crítico, TRIGGERS=2, MODALIDADES=2 (vídeo e texto) e DIRETRIZES=0. A ausência de diretrizes recuperadas reflete a refatoração multi-query (ADR-017): para casos cirúrgicos sem eixo humano ativado, o orquestrador suprime a query base e o retriever não retorna chunks tangencialmente relacionados à ginecologia obstétrica indexada. Case ID `case-6ecf6b9d3caf` e Audit ID `48` identificam o registro persistido para auditoria.

![Resultado consolidado do caso cirúrgico crítico](figures/screenshots/aba_multimodal_sangramento_resultado.png)

*Figura 8.6.5: card Resultado consolidado com badge **CRÍTICO**, KPIs de nível, triggers, modalidades e diretrizes, além de Case ID e Audit ID.*

A figura 8.6.6 mostra o relatório clínico gerado pelo Azure OpenAI `gpt-4.1-mini` (via AI Foundry). As seções Resumo, Achados, Diretrizes Aplicáveis e Recomendações seguem o contrato do `src/report.py:SYSTEM_PROMPT_PT_BR`. O ponto a destacar é a regra de salvaguarda contra alucinação: quando o contexto RAG vem vazio, o LLM declara explicitamente que nenhuma das diretrizes indexadas aborda o manejo específico de sangramento intraoperatório em cirurgia ginecológica e orienta o seguimento de protocolos institucionais, sem fabricar referências a documentos inexistentes (comportamento previsto e auditável após a correção descrita em ADR-017).

![Relatório clínico gerado pelo LLM para o caso de sangramento](figures/screenshots/aba_multimodal_sangramento_relatorio.png)

*Figura 8.6.6: relatório clínico em Resumo, Achados, Diretrizes Aplicáveis e Recomendações, com declaração explícita de ausência de diretrizes aplicáveis indexadas e cinco ações enumeradas para o manejo hemodinâmico e hemostático.*

A figura 8.6.7 detalha o resumo de anomalia, a tabela de triggers e o painel de diretrizes consultadas. Duas regras determinísticas do `RuleEngine` (`src/anomaly/rules.py`) sustentam o nível final: `video.surgical_instrument_presence` com severidade `Moderado` (instrumental em uso em 6 frames consecutivos) e `video.bleeding_detected` com severidade `Crítico` (sangramento em 6 de 6 frames, maior streak consecutivo igual a 6). O nível final é o máximo entre as severidades dos triggers, isto é, `critical`. As cinco ações recomendadas vinculadas a esses triggers (documentar instrumental, correlacionar fase cirúrgica, acionar protocolo de hemorragia, verificar hemodinâmica e reforçar equipe) provêm das regras e independem do LLM, preservando a auditabilidade dessa camada.

![Resumo de anomalia, tabela de triggers e diretrizes consultadas](figures/screenshots/aba_multimodal_sangramento_anomalia.png)

*Figura 8.6.7: painéis Resumo de anomalia (nível crítico, dois triggers, cinco ações), tabela de triggers com `video.surgical_instrument_presence` (Moderado) e `video.bleeding_detected` (Crítico) e painel Diretrizes consultadas exibindo "Sem diretrizes recuperadas no contexto.".*

As quatro figuras 8.6.4 a 8.6.7 fecham a demonstração fim a fim do caso cirúrgico crítico: a entrada multimodal foi processada pelos pilares relevantes (vídeo e texto), a camada determinística do `RuleEngine` produziu nível e ações auditáveis, o RAG corretamente identificou ausência de cobertura temática e o LLM respeitou essa lacuna sem alucinar diretrizes.

### 8.7 Auditoria e Reprodutibilidade

A aba **Auditoria** consolida em uma única interface o histórico de execuções registradas pelo `src/audit.py` em `data/processed/audit.sqlite`. Cada chamada de `Orchestrator.process_case` produz um registro com snapshot completo dos artefatos do caso, viabilizando inspeção post hoc das decisões do pipeline sem necessidade de reprocessar entradas.

#### Demonstração visual da aba Auditoria

A figura 8.7.1 apresenta o topo da aba após o carregamento inicial. O painel exibe quatro KPIs derivados da contagem agregada no SQLite: TOTAL=20 casos listados, NORMAIS=0 sem anomalia, MODERADOS=15 em atenção e CRÍTICOS=5 em ação imediata. A ausência de casos classificados como normais reflete o desenho dos cenários de demonstração, que sempre disparam ao menos um trigger determinístico do `RuleEngine`. O dropdown Limite=20 controla a paginação, e a tabela Casos registrados exibe as primeiras cinco linhas (IDs 50, 49, 48, 47 e 46) com as colunas ID (chave auto-incremento do SQLite), Case ID (UUID curto), Criado em (timestamp UTC ISO 8601), Risco (badge Moderado ou Crítico) e Modalidades processadas, além do botão Ver detalhes por linha.

![Topo da aba Auditoria com KPIs e início da tabela de casos](figures/screenshots/aba_auditoria_lista.png)

*Figura 8.7.1: topo da aba Auditoria com KPIs TOTAL=20, NORMAIS=0, MODERADOS=15 e CRÍTICOS=5, dropdown Limite=20 e primeiras cinco linhas da tabela Casos registrados (IDs 50 a 46).*

A figura 8.7.2 estende a visualização da tabela para onze linhas (IDs 50 a 40), evidenciando o volume acumulado de execuções incrementais durante o desenvolvimento. As linhas alternam entre badges Críticos e Moderados, com modalidades variando entre `video,text`, `video,audio,text` e combinações análogas conforme o cenário processado. No rodapé inicia-se o accordion fechado Detalhar e exportar um caso, que dá acesso à inspeção individual de cada registro. A interface não implementa autenticação ou separação por tenant neste estágio do projeto, decisão alinhada ao escopo acadêmico; uma evolução natural para ambiente produtivo seria adicionar escopo por usuário e controle de acesso.

![Tabela completa da aba Auditoria com 11 casos visíveis](figures/screenshots/aba_auditoria_tabela.png)

*Figura 8.7.2: tabela Casos registrados expandida exibindo onze linhas (IDs 50 a 40) com diversidade de níveis de risco e modalidades, e accordion fechado Detalhar e exportar um caso no rodapé.*

A figura 8.7.3 mostra o accordion expandido após o clique em Ver detalhes na linha do caso ID 50. O campo Audit ID já vem pré-preenchido com o identificador do caso selecionado, e o botão Buscar detalhe materializa o registro lado a lado em duas colunas. Acima dos painéis, o badge **Moderado** acompanha o resumo `case_id=case-69af9a8169b6, criado em 2026-05-18T06:40:27.470873+00:00`. À esquerda, Relatório do caso renderiza o campo `report_md` em markdown clínico estruturado (Resumo, Achados, Diretrizes Aplicáveis e Recomendações), exatamente como produzido pelo `gpt-4.1-mini` no momento da inferência. À direita, Registro completo (JSON) exibe os campos auditáveis: `id`, `case_id`, `created_at`, `risk_level`, `modalities`, `triggers_json` com a lista de regras disparadas e seus campos de evidência, `explanation` com a mensagem agregada do classificador e `report_md` com o texto completo do relatório. A persistência é feita no momento da inferência, sem reprocessamento sob demanda.

![Detalhe expandido do caso 50 com relatório markdown e JSON lado a lado](figures/screenshots/aba_auditoria_detalhe.png)

*Figura 8.7.3: accordion Detalhar e exportar um caso expandido, com badge Moderado, painel Relatório do caso à esquerda renderizando o `report_md` e painel Registro completo (JSON) à direita exibindo `triggers_json`, `explanation` e demais campos auditáveis.*

A figura 8.7.4 apresenta a continuação do detalhe, com o fechamento do JSON exibindo o campo `metadata_json` (que carrega `patient_id` e `text_context`) e, abaixo, a seção Exportar registro. O botão Exportar JSON gera um arquivo temporário (`audit_50_awgzks7p.json`, 2.3 KB) contendo o registro íntegro do caso, útil para compartilhar evidência clínica com auditores externos ou alimentar análise estatística post hoc. A persistência local em SQLite atende o escopo deste trabalho; em ambiente produtivo, uma evolução possível seria espelhar os registros em armazenamento cloud (Azure Blob Storage ou equivalente) com políticas de retenção e cifragem em repouso. O rodapé da aba mantém o footer do aplicativo com a versão `v0.1.0`.

![Continuação do detalhe com metadata_json e botão Exportar JSON](figures/screenshots/aba_auditoria_exportar.png)

*Figura 8.7.4: final do JSON do caso 50 com `metadata_json` contendo `patient_id` e `text_context`, seção Exportar registro com botão Exportar JSON e arquivo `audit_50_awgzks7p.json` (2.3 KB) disponibilizado para download.*

As quatro figuras 8.7.1 a 8.7.4 fecham a demonstração da aba Auditoria: a tabela agrega o histórico de execuções com KPIs e paginação, o detalhe expandido reúne o relatório clínico e o registro JSON com todos os campos auditáveis, e o botão de exportação materializa cada caso em arquivo individual, fechando o ciclo de rastreabilidade fim a fim entre entrada multimodal, decisão do classificador e relatório clínico gerado.

---

## 9. Limitações e Trabalho Futuro

### 9.1 Limitações Conhecidas

- **Viés do classificador de emoção vocal (wav2vec2).** Em validação empírica com TTS Azure Speech e TTS de alta qualidade (ElevenLabs/Fish Audio), o `wav2vec2-base-superb-er` (pré-treinado em RAVDESS, atores americanos em inglês) classifica praticamente toda voz feminina em PT-BR como `angry` com confiança superior a 0.95, **independentemente da emoção real expressa**. Análise de features acústicas (F0 std=47.8 Hz, range tonal=198 Hz) confirma que o áudio testado é expressivo; o modelo está enviesado. Mitigação implementada: cliente alternativo `AzureOpenAIAudioEmotion` (`src/audio/azure_openai_audio.py`) usa GPT-4o multimodal sem o viés daquele domínio de treino. Quando o deployment de áudio não está provisionado, o pipeline cai no wav2vec2 com peso reduzido na decisão de risco final (sentimento textual via Azure Language passa a ser o pilar primário de afeto).
- **Viés provável do classificador de emoção facial (FER).** Hipótese análoga à do wav2vec2: o modelo `justinshenk/fer` foi treinado em FER-2013 (fotos atuadas frontais) e pode atribuir distress a expressões neutras em iluminação variável ou ângulos atípicos. O mesmo `AzureOpenAIAudioEmotion` pode ser estendido para receber frames (GPT-4o suporta imagem) e substituir o pilar facial pela mesma lógica.
- **Transferência de domínio do YOLO.** Treinado em colecistectomia (CholecSeg8k), aplicado a cirurgia ginecológica. Técnica laparoscópica e instrumental são idênticos (Grasper, L-hook), mas tecidos e contexto visual diferem. **Cobertura parcial da taxonomia ginecológica:** outros instrumentos comuns em histerectomia laparoscópica (Harmonic Scalpel, LigaSure, tesoura laparoscópica) não estão nas classes treinadas e não serão detectados.
- **Áudios de consulta sintéticos.** O gold standard da demo é TTS Azure (vozes Francisca, Antonio, Brenda com estilos `sad`, `empathetic`, `terrified`). Áudios reais de pacientes não são usados por LGPD e ausência de comitê de ética. Validação em fala espontânea é feita via CORAA-SER, mas mesmo aí a fidelidade do wav2vec2 é baixa (ver acima).
- **Cobertura limitada do RAG (9 PDFs).** Documentos indexados cobrem pré-natal, pré-eclâmpsia, alto risco, câncer mama/colo, IST/violência, parto normal, saúde reprodutiva e saúde mental (Caderno AB nº 34). Temas como endometriose, SOP, mioma, infertilidade, menopausa e câncer de ovário ficam fora. O threshold de score 0.3 no retriever evita responder com chunks irrelevantes, mas a lacuna de cobertura permanece.
- **Deploy em HF Spaces.** 16 GB RAM, 2 vCPU, sem GPU. Inferência foi planejada para CPU. Hibernação ao ocioso é aceitável para demonstração.

### 9.2 Caminhos de Extensão Identificados

Levantamentos feitos durante o projeto que apontam direções possíveis de melhoria, sem compromisso de execução dentro do escopo deste módulo:

- O cliente `AzureOpenAIAudioEmotion` (`src/audio/azure_openai_audio.py`) e `AzureOpenAIVisionEmotion` (`src/video/azure_openai_vision.py`) ficam plugáveis ao Azure AI Foundry sem alteração de pipeline: basta preencher `AZURE_OPENAI_AUDIO_DEPLOYMENT` e `AZURE_OPENAI_VISION_DEPLOYMENT` no `.env`. Substituem wav2vec2 e FER respectivamente.
- Expansão da taxonomia do detector visual com datasets de maior escala e cobertura de classes ginecológicas.
- Expansão da cobertura do RAG para temas hoje fora dos 9 PDFs indexados (endometriose, SOP, infertilidade, menopausa, câncer de ovário).
- Substituição do `wav2vec2-base-superb-er` por modelo fine-tunado em CORAA-SER, caso queira manter classificação de emoção vocal totalmente local.

---

## 10. Conclusão

O projeto entrega uma solução multimodal funcional cobrindo três das quatro funcionalidades e quatro dos cinco objetivos do enunciado, com integração Azure real (não simulada) e fallback local quando as chaves não estão disponíveis. A pivotagem do alvo YOLO de "sangramento anômalo" para "instrumentos cirúrgicos laparoscópicos" (ADR-012) sustentou aderência LITERAL ao alvo 1 do enunciado mediante dataset público reproduzível (CholecSeg8k), preservando coerência clínica via transferência de domínio entre colecistectomia e cirurgia ginecológica laparoscópica. A separação explícita entre os três artefatos (dataset, RAG, LLM) torna o pipeline auditável e cada peça substituível.

Quantitativamente, o detector custom convergiu com mAP@50 = 0.989 e mAP@50-95 = 0.882 no test split, mantendo desempenho equilibrado entre as 3 classes (variação de 0.982 a 0.993 em mAP@50), inclusive na classe minoritária `blood` (71 instâncias). A decisão de adicionar a classe `Blood` ao escopo das 3 classes finais, justificada em ADR-013, ampliou a cobertura do detector de 1 para 2 dos 4 entregáveis sugeridos pelo enunciado (detecção de instrumentos e detecção de sangramento intraoperatório), com custo de treino marginal. As limitações de viés reconhecidas em wav2vec2 e FER foram mitigadas arquiteturalmente com clientes alternativos plugáveis (`AzureOpenAIAudioEmotion` e `AzureOpenAIVisionEmotion`), que assumem o pilar quando os respectivos deployments Azure são provisionados, mantendo fallback local funcional caso contrário.

---

## 11. Referências

### Datasets

- Hong, W.Y. et al. *CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80*. arXiv:2012.12463, 2020. Licença CC BY-NC-SA 4.0.
- CORAA-SER. `alefiury/CORAA-SER` no Hugging Face.
- RAVDESS. Universidade Ryerson.

### Diretrizes Clínicas Indexadas no RAG

- Ministério da Saúde. *Manual de Pré-natal de Baixo e Alto Risco*.
- FEBRASGO. *Diretriz de Pré-eclâmpsia*.
- Ministério da Saúde. *Manual de Gestação de Alto Risco*.
- INCA. *Diretrizes de Detecção Precoce do Câncer de Mama*.
- INCA. *Diretrizes de Detecção Precoce do Câncer do Colo do Útero*.
- Ministério da Saúde. *PCDT de IST e Atenção a Vítimas de Violência*.
- Ministério da Saúde. *Diretrizes de Atenção ao Parto Normal*.
- Ministério da Saúde. *Caderno de Atenção Básica nº 26: Saúde Sexual e Reprodutiva*.
- Ministério da Saúde. *Caderno de Atenção Básica nº 34: Saúde Mental*.

### Modelos e Frameworks

- Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
- MediaPipe. https://github.com/google/mediapipe
- faster-whisper. https://github.com/SYSTRAN/faster-whisper
- wav2vec2 (superb/wav2vec2-base). Hugging Face.
- BAAI/bge-m3. Hugging Face.
- Chroma. https://www.trychroma.com/
- LangChain. https://www.langchain.com/
- Gradio. https://www.gradio.app/

### Serviços Azure

- Azure OpenAI Service (AI Foundry).
- Azure Cognitive Services: Speech, Language.

### Documentação Interna

- `docs/overview.md`
- `docs/arquitetura/arquitetura.md`
- `docs/arquitetura/decisoes_tecnicas.md` (ADRs 001 a 019)
- `docs/arquitetura/modelos_e_datasets.md`
- `docs/arquitetura/padroes_codigo.md`
- `README.md`
