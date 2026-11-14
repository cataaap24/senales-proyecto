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

# Rechazo (no es parte del método principal, ver README):
# 1) Sin canto: rango dinámico de la señal filtrada (cuadros de 50 ms, percentil 95 frente a percentil 10).
#    Un canto tiene sílabas y pausas (rango alto); el ruido constante, el zumbido o el silencio no.
UMBRAL_ACTIVIDAD_DB = 6.0
# 2) No reconocido: nota del profesor (pág. 45), a los promedios de energía se les agrega la desviación
#    estándar para tener un intervalo por subbanda. Se mide en dB y con una desviación mínima porque, al
#    normalizar la energía, la subbanda dominante casi no varía en el banco (desviación ~0.1 dB).
K_DESVIACION = 3.0
DESVIACION_MINIMA_DB = 1.0
