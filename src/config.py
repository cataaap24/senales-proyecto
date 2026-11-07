from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
BANCO = RAIZ / "datos" / "banco_aves"
MODELO = RAIZ / "modelo" / "vectores_energia.json"
FIGURAS = RAIZ / "resultados" / "figuras"

FS = 22050
DURACION = 5.0
N = int(FS * DURACION)

PREFILTRO_FC1 = 200.0
PREFILTRO_FC2 = 8000.0
