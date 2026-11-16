from math import gcd

import numpy as np

from config import FS, N, UMBRAL_ACTIVIDAD_DB
from espectro import calcular_fft_completa  # noqa: F401
from preprocesamiento import acondicionar


def cargar_audio(ruta):
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, frecuencia = sf.read(str(ruta), dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if frecuencia != FS:
        divisor = gcd(int(frecuencia), FS)
        audio = resample_poly(audio, FS // divisor, int(frecuencia) // divisor)
    return audio


def recortar(audio):
    audio = np.asarray(audio, dtype=float)
    if len(audio) < N:
        return np.pad(audio, (0, N - len(audio)))

    energia_acumulada = np.concatenate(([0.0], np.cumsum(audio ** 2)))
    inicio = int(np.argmax(energia_acumulada[N:] - energia_acumulada[:-N]))
    return audio[inicio:inicio + N]


def espectro(audio):
    audio_acondicionado = acondicionar(recortar(audio))
    return calcular_fft_completa(audio_acondicionado, FS)


def validar_bordes(bordes):
    bordes = np.asarray(bordes, dtype=float)
    if (bordes.ndim != 1 or not 4 <= len(bordes) <= 9
            or not np.all(np.isfinite(bordes))
            or bordes[0] < 0 or bordes[-1] > FS
            or not np.all(np.diff(bordes) > 0)):
        raise ValueError(
            "Se requieren de 3 a 8 subbandas con bordes crecientes entre 0 y fs."
        )
    return bordes


def energias(magnitud, frecuencias, bordes):
    """Aplica E_i = (1/L_i) * sum(|X(k)|²) en cada subbanda."""
    bordes = validar_bordes(bordes)
    magnitud = np.asarray(magnitud, dtype=float)
    frecuencias = np.asarray(frecuencias, dtype=float)
    if magnitud.shape != frecuencias.shape:
        raise ValueError("La magnitud y las frecuencias deben tener la misma longitud.")

    resultado = []
    for inferior, superior in zip(bordes[:-1], bordes[1:]):
        indices = (frecuencias >= inferior) & (frecuencias < superior)
        cantidad = int(np.count_nonzero(indices))
        if cantidad == 0:
            raise ValueError("Una subbanda no contiene muestras de frecuencia.")
        resultado.append(float(np.sum(magnitud[indices] ** 2) / cantidad))
    return np.asarray(resultado)


def distancia(energia_audio, energia_especie):
    return float(np.sum(np.abs(energia_audio - energia_especie)))


def rango_dinamico(audio):

    senal = acondicionar(recortar(audio), normalizar=False)
    largo = int(0.05 * FS)
    cuadros = len(senal) // largo
    energia = np.mean(senal[:cuadros * largo].reshape(cuadros, largo) ** 2, axis=1) + 1e-20
    return float(10 * np.log10(np.percentile(energia, 95) / np.percentile(energia, 10)))


def hay_canto(audio):
    return rango_dinamico(audio) >= UMBRAL_ACTIVIDAD_DB


def dentro_de_intervalo(energia_audio, intervalo):
    energia_db = 10 * np.log10(np.asarray(energia_audio, dtype=float) + 1e-20)
    media = np.asarray(intervalo["media_db"], dtype=float)
    ancho = intervalo["k"] * np.maximum(np.asarray(intervalo["desviacion_db"], dtype=float),
                                        intervalo["desviacion_minima_db"])
    return bool(np.all(np.abs(energia_db - media) <= ancho))
