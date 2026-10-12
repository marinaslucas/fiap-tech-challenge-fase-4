"""[OBSOLETO] Script legado de seed sintetico - NAO USAR.

Substituido por `seed_real_examples.py`, que opera no schema atual de
5 casos em subpastas `consultas/` e `cirurgias/` com videos reais do
YouTube + frames CSeg8k + TTS Azure.

Mantido como stub para preservar referencia historica no git. Codigo
funcional removido para evitar execucao acidental que sobrescreveria
`data/examples/manifest.json` com schema antigo de 3 casos flat.

Uso atual:

    python scripts/seed_real_examples.py
"""

raise SystemExit(
    "scripts/seed_examples.py e obsoleto. "
    "Use 'python scripts/seed_real_examples.py' para regenerar exemplos."
)
