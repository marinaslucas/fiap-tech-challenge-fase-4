# Padrões de Código

Convenções gerais que guiam o projeto. São diretrizes pragmáticas, não regras absolutas (trabalho acadêmico, escopo finito).

## Linguagem e Versão

- Python 3.12

## Lint e Formatação

Ruff cobre lint e formatação. Configuração em `pyproject.toml` (seção `[tool.ruff]`).

Regras ativas: pycodestyle (E/W), pyflakes (F), isort (I), pep8-naming (N), bugbear (B), pyupgrade (UP), simplificações (SIM). Linha máxima: 100.

```bash
ruff check src/ tests/ ui/ scripts/
ruff format src/ tests/ ui/ scripts/
```

## Idiomas

- **Nomes** (variáveis, funções, classes, módulos): EN
- **Comentários, docstrings e mensagens de log:** PT-BR

## Type Hints

Usar em interfaces públicas (funções exportadas, classes Pydantic, dataclasses). Em código interno fica a critério da clareza.

```python
def detect_surgical_instrument(video_path: Path, weights: Path) -> list[Detection]:
    ...
```

## Docstrings

Google style, em funções/classes públicas. Resumir o que faz e descrever args/returns relevantes, sem encher de boilerplate.

```python
def detect_surgical_instrument(video_path: Path, weights: Path) -> list[Detection]:
    """Detecta instrumental cirúrgico em frames de vídeo laparoscópico.

    Args:
        video_path: caminho do arquivo de vídeo.
        weights: caminho dos pesos YOLO treinados em CholecSeg8k.

    Returns:
        Lista de detecções com classe, confiança e bounding box.
    """
    ...
```

## Imports

Ruff/isort cuida da ordem: stdlib → third-party → locais, blocos separados por linha em branco.

## Nomenclatura

| Tipo | Convenção | Exemplo |
|---|---|---|
| Módulos | snake_case | `azure_speech.py` |
| Classes | PascalCase | `SurgicalInstrumentDetector` |
| Funções e variáveis | snake_case | `detect_surgical_instrument` |
| Constantes | SCREAMING_SNAKE_CASE | `MAX_FRAMES_PER_VIDEO` |
| Privadas | underscore prefix | `_calculate_confidence` |

## Dados e Configuração

- **Modelos de dados:** Pydantic v2 (ou dataclasses quando suficiente)
- **Configuração:** `src/config/settings.py` via Pydantic Settings
- **Acesso ao `.env`:** sempre via `settings`, nunca `os.getenv` espalhado
- **Nunca:** hardcode de chaves, commit de `.env`

## Erros e Logging

- Falhar cedo, mensagem clara em PT-BR
- Logger por módulo: `logger = logging.getLogger(__name__)`
- INFO para eventos normais, WARNING para inesperado-não-fatal, ERROR para falhas reais
- Evitar `except Exception` sem motivo

## Testes

Pytest, smoke tests cobrindo o caminho feliz de cada módulo público. Mocks isolam dependências externas (Azure, Whisper, wav2vec2).

```
tests/
├── unit/                 (smoke por módulo)
└── integration/          (pipeline end-to-end)
```

## Commits

Mensagem em PT-BR no imperativo, primeira letra maiúscula. Ex: `Adiciona pipeline de áudio com Whisper`, `Corrige path do RAG no settings`.
