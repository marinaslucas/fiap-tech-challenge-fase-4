"""Gera PDFs sinteticos curtos com texto ficticio sobre pre-natal e pre-eclampsia.

Util para destravar o RAG enquanto os PDFs reais nao estao disponiveis.
**Nao representam diretrizes
oficiais**: substituir pelos arquivos reais do MS e da FEBRASGO antes da demo.

Uso:
    python scripts/gen_synthetic_pdfs.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gen_synthetic_pdfs")

OUTPUT_DIR = Path("data/raw")

PDF_PRENATAL = """
Diretriz sintetica de pre-natal (texto ficticio para validacao do RAG).

A consulta de pre-natal de baixo risco recomenda no minimo seis encontros
durante a gestacao, com aferição de pressao arterial, peso, altura uterina
e batimentos cardiofetais em cada visita. Solicitar hemograma completo,
glicemia de jejum, sorologia para HIV, sifilis, hepatite B e toxoplasmose
no primeiro trimestre.

Sinais de alerta:
- Sangramento vaginal em qualquer trimestre.
- Cefaleia persistente acompanhada de escotomas ou epigastralgia.
- Edema progressivo de mãos e face.
- Reducao perceptivel de movimentacao fetal apos 28 semanas.
- Pressao arterial sistolica acima de 140 mmHg ou diastolica acima de 90 mmHg
  em duas medidas com intervalo minimo de quatro horas.

A vacinacao da gestante inclui dT/dTpa, hepatite B e influenza segundo o
calendario do Ministerio da Saude. Suplementacao de acido folico 400 mcg/dia
antes da concepcao e durante o primeiro trimestre, e de sulfato ferroso a
partir da 20 semana.

A bem-estar emocional deve ser avaliado em todas as consultas. Sintomas
como anedonia persistente, choro frequente, ideacao suicida ou ansiedade
intensa indicam encaminhamento para acompanhamento de saude mental.
"""

PDF_PREECLAMPSIA = """
Diretriz sintetica de pre-eclampsia (texto ficticio para validacao do RAG).

Pre-eclampsia e definida como hipertensao gestacional (PA >= 140/90 mmHg)
apos 20 semanas em gestante previamente normotensa, associada a proteinuria
(>= 300 mg/24h) ou criterio de gravidade.

Criterios de gravidade:
- Pressao arterial sistolica >= 160 mmHg ou diastolica >= 110 mmHg.
- Plaquetopenia (< 100.000/mm3).
- Elevacao de transaminases ao dobro do limite superior.
- Insuficiencia renal aguda (creatinina serica > 1,1 mg/dL).
- Edema pulmonar.
- Sintomas neurologicos (cefaleia intensa, escotomas, alteracao visual).
- Dor epigastrica ou no quadrante superior direito.

Conduta no pronto atendimento:
1. Repouso em decubito lateral esquerdo.
2. Aferir pressao arterial a cada 15 minutos.
3. Sulfato de magnesio para profilaxia de eclampsia em casos graves.
4. Anti-hipertensivo (hidralazina ou nifedipino) se PA >= 160/110 mmHg.
5. Considerar resolucao da gestacao em pre-eclampsia grave a partir de 34 semanas.

A sindrome HELLP cursa com hemolise, elevacao de transaminases e plaquetopenia.
A presença de dor epigastrica ou em quadrante superior direito em gestante
hipertensa exige investigacao laboratorial imediata.
"""


def _gerar(destino: Path, texto: str, titulo: str) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(destino), pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(titulo, styles["Title"]), Spacer(1, 12)]
    for paragrafo in texto.strip().split("\n\n"):
        story.append(Paragraph(paragrafo.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 8))
    doc.build(story)
    logger.info("PDF gerado em %s", destino)
    return destino


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _gerar(
        OUTPUT_DIR / "manual_ms_prenatal_sintetico.pdf",
        PDF_PRENATAL,
        "Manual ficticio de pre-natal (sintetico)",
    )
    _gerar(
        OUTPUT_DIR / "febrasgo_preeclampsia_sintetico.pdf",
        PDF_PREECLAMPSIA,
        "Diretriz ficticia de pre-eclampsia (sintetica)",
    )


if __name__ == "__main__":
    main()
