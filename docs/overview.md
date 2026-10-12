# Overview - medica-ia-multimodal

## O Que É

Sistema de monitoramento multimodal aplicado à saúde da mulher, desenvolvido para o Tech Challenge Fase 4 da pós-graduação Tech IADT. Processa vídeo, áudio e texto para identificar precocemente sinais de risco em saúde materna e bem-estar psicológico, gerando alertas estruturados para a equipe clínica.

## Contexto

Sistema independente da entrega da Fase 3. A arquitetura parte do zero, voltada para análise multimodal e detecção de anomalias em saúde da mulher.

## Objetivo

Demonstrar uma solução multimodal funcional que:

- Analisa vídeos clínicos com detecção customizada via YOLOv8
- Analisa áudios de consultas com transcrição, features acústicas e classificação de emoção
- Recupera diretrizes ginecológicas e obstétricas via RAG
- Detecta anomalias e gera alertas estruturados
- Integra Azure Cognitive Services como camada de serviços gerenciados

## Escopo

### Funcionalidades selecionadas (3 de 4 do enunciado)

1. Análise de vídeos clínicos (YOLOv8 customizado para instrumentos cirúrgicos: Grasper e L-hook Electrocautery, treinado em CholecSeg8k)
2. Processamento de gravações de voz em consultas
3. Integração com Azure Cognitive Services

### Objetivos selecionados (4 de 5 do enunciado)

1. Detecção precoce de riscos em saúde materna e ginecológica
2. Monitoramento do bem-estar psicológico feminino
3. Uso de serviços em nuvem para ampliar capacidade
4. Detecção de anomalias em tempo real

### Foco do YOLOv8 customizado

Detecção de instrumental cirúrgico em vídeo laparoscópico real (classes `Grasper` e `L-hook Electrocautery`). Treino sobre CholecSeg8k (Hong et al., 2020, CC BY-NC-SA 4.0). Justificativa: aderência LITERAL ao alvo 1 do enunciado, reprodutibilidade (3.1 GB, anônimo na Hugging Face) e transferência clínica válida, já que a técnica laparoscópica é idêntica entre colecistectomia (treino) e cirurgia ginecológica (aplicação). Detalhes em [ADR-012](arquitetura/decisoes_tecnicas.md).

## Fora de Escopo

- Funcionalidade de sinais vitais (4ª opção do enunciado)
- Objetivo de detecção de violência doméstica (5º objetivo)
- Reaproveitamento da Fase 3 (LLM fine-tunado, MIMIC-IV, LangGraph)
- Integração com AWS Textract e Comprehend Medical
- Backend FastAPI separado, frontend React

## Entregáveis

- Repositório Git público
- Código-fonte completo seguindo os padrões em [arquitetura/padroes_codigo.md](arquitetura/padroes_codigo.md)
- Relatório técnico em markdown descrevendo fluxo multimodal, modelos, resultados
- Demo funcional em Gradio Blocks acessível via URL pública (Hugging Face Spaces)
- Vídeo de até 15 minutos demonstrando a solução

## Stack Técnica Resumida

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| Linter/formatter | Ruff |
| UI | Gradio Blocks |
| Vídeo | YOLOv8 (ultralytics), MediaPipe, OpenCV |
| Áudio | faster-whisper, librosa, wav2vec2 |
| RAG | Chroma + bge-m3 |
| LLM | Azure OpenAI (GPT-4.1-mini via AI Foundry) |
| TTS | Azure TTS |
| Cloud | Azure Cognitive Services (Speech, Language) |
| Logs | SQLite |
| Container | Docker + Docker Compose |
| Deploy | Hugging Face Spaces |

Detalhes em [arquitetura/arquitetura.md](arquitetura/arquitetura.md) e [arquitetura/modelos_e_datasets.md](arquitetura/modelos_e_datasets.md).

## Equipe e Ambiente

- Grupo: Joalisson Borges, Luis Gustavo Santini, Marina Souza Lucas, Diego Santos
- Treino de YOLO no Google Colab (uso pontual de GPU gratuita)

## Métricas de Sucesso

1. App Gradio rodando fim a fim com vídeo, áudio e texto integrados
2. YOLO custom treinado com mAP aceitável para o caso de uso
3. Pelo menos 3 cenários sintéticos demonstrando alertas distintos (normal, alerta moderado, alerta crítico)
4. Integração Azure visível na demo (não simulada)
5. Relatório técnico com métricas, exemplos e diagramas
6. Repositório com README e estrutura limpa
