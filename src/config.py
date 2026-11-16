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

UMBRAL_ACTIVIDAD_DB = 6.0

K_DESVIACION = 3.0
DESVIACION_MINIMA_DB = 1.0
