"""Operaciones compartidas por el entrenamiento y el reconocimiento.

El orden sigue el método descrito en el README: acondicionar, obtener la
magnitud de la FFT y calcular la energía de cada subbanda. Las dos fases llaman
estas mismas funciones para evitar diferencias entre entrenamiento y uso.
"""

from math import gcd

import numpy as np

from config import FS, N, UMBRAL_ACTIVIDAD_DB
from espectro import calcular_fft
from preprocesamiento import acondicionar


def cargar_audio(ruta):
    """Lee un archivo, convierte sus canales a mono y lo lleva a FS Hz."""
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
    """Elige N muestras consecutivas de mayor energía o completa con ceros."""
    audio = np.asarray(audio, dtype=float)
    if len(audio) < N:
        return np.pad(audio, (0, N - len(audio)))

    # La suma acumulada permite comparar todas las ventanas sin repetir sumas.
    energia_acumulada = np.concatenate(([0.0], np.cumsum(audio ** 2)))
    inicio = int(np.argmax(energia_acumulada[N:] - energia_acumulada[:-N]))
    return audio[inicio:inicio + N]


def espectro(audio):
    """Calcula |X(k)| tras aplicar el mismo acondicionamiento en ambas fases."""
    audio_acondicionado = acondicionar(recortar(audio))
    return calcular_fft(audio_acondicionado, FS)


def validar_bordes(bordes):
    """Exige entre tres y ocho subbandas ordenadas dentro de Nyquist."""
    bordes = np.asarray(bordes, dtype=float)
    if (bordes.ndim != 1 or not 4 <= len(bordes) <= 9
            or not np.all(np.isfinite(bordes))
            or bordes[0] < 0 or bordes[-1] > FS / 2
            or not np.all(np.diff(bordes) > 0)):
        raise ValueError(
            "Se requieren de 3 a 8 subbandas con bordes crecientes entre 0 y Nyquist."
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
    """Suma las diferencias absolutas entre energías de subbandas."""
    return float(np.sum(np.abs(energia_audio - energia_especie)))


def rango_dinamico(audio):
    """Rango dinámico en dB de la señal filtrada (sin normalizar), en cuadros de 50 ms.

    Compara la energía de los cuadros más fuertes (percentil 95) con la de los más débiles
    (percentil 10). Un canto alterna sílabas y pausas, así que su rango es alto (en el banco,
    mínimo 11 dB); el silencio, el ventilador o el zumbido eléctrico dan menos de 2 dB.
    """
    senal = acondicionar(recortar(audio), normalizar=False)
    largo = int(0.05 * FS)
    cuadros = len(senal) // largo
    energia = np.mean(senal[:cuadros * largo].reshape(cuadros, largo) ** 2, axis=1) + 1e-20
    return float(10 * np.log10(np.percentile(energia, 95) / np.percentile(energia, 10)))


def hay_canto(audio):
    """True si la señal tiene la variación de energía propia de un canto."""
    return rango_dinamico(audio) >= UMBRAL_ACTIVIDAD_DB


def dentro_de_intervalo(energia_audio, intervalo):
    """Nota del profesor: intervalo de energía = promedio ± k·desviación, en cada subbanda (dB)."""
    energia_db = 10 * np.log10(np.asarray(energia_audio, dtype=float) + 1e-20)
    media = np.asarray(intervalo["media_db"], dtype=float)
    ancho = intervalo["k"] * np.maximum(np.asarray(intervalo["desviacion_db"], dtype=float),
                                        intervalo["desviacion_minima_db"])
    return bool(np.all(np.abs(energia_db - media) <= ancho))
