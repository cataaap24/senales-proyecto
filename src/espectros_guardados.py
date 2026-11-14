"""Guarda y lee las curvas promedio que usa la interfaz."""

import hashlib
import json
from pathlib import Path

import numpy as np

from config import MODELO, RAIZ


RUTA_ESPECTROS = RAIZ / "modelo" / "espectros_promedio.json"
MAX_PUNTOS = 4096


def guardar_espectros(modelo, frecuencias, promedios, destino=RUTA_ESPECTROS,
                     ruta_modelo=MODELO):
    """Exporta la FFT completa, agrupada solo para dibujarla con fluidez."""
    frecuencias = np.asarray(frecuencias, dtype=float)
    bordes = np.asarray(modelo["bordes_hz"], dtype=float)
    puntos_fft = int(modelo["fs"] * modelo["duracion"])
    if (frecuencias.ndim != 1 or len(frecuencias) != puntos_fft
            or not np.all(np.isfinite(frecuencias))
            or not np.all(np.diff(frecuencias) > 0)
            or frecuencias[0] < 0 or frecuencias[-1] >= modelo["fs"]
            or set(promedios) != set(modelo["especies"])):
        raise ValueError("Los espectros promedio no corresponden al modelo.")

    cantidad = min(len(frecuencias), MAX_PUNTOS)
    limites = np.linspace(0, len(frecuencias), cantidad + 1, dtype=int)
    tamaños = np.diff(limites)
    eje = np.add.reduceat(frecuencias, limites[:-1]) / tamaños
    series = {}
    for especie, valores in promedios.items():
        magnitudes = np.asarray(valores, dtype=float)
        if (magnitudes.shape != frecuencias.shape
                or not np.all(np.isfinite(magnitudes)) or np.any(magnitudes < 0)):
            raise ValueError(f"Espectro promedio inválido para {especie}.")
        series[especie] = (
            np.add.reduceat(magnitudes, limites[:-1]) / tamaños
        ).tolist()

    datos = {
        "version": 2,
        "modelo_sha256": hashlib.sha256(Path(ruta_modelo).read_bytes()).hexdigest(),
        "fft_puntos": puntos_fft,
        "bordes_hz": bordes.tolist(),
        "frecuencias_hz": eje.tolist(),
        "magnitudes": series,
    }
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(destino.name + ".tmp")
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    temporal.replace(destino)
    return destino


def cargar_espectros(modelo, ruta=RUTA_ESPECTROS, ruta_modelo=MODELO):
    """Valida las series de la gráfica sin calcular FFT ni modificar el modelo."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(
            f"Falta {ruta.name}. La fase 2 debe exportar los espectros promedio."
        )

    datos = json.loads(ruta.read_text(encoding="utf-8"))
    if not isinstance(datos, dict) or datos.get("version") != 2:
        raise ValueError("Versión del archivo de espectros no compatible.")
    firma = hashlib.sha256(Path(ruta_modelo).read_bytes()).hexdigest()
    if datos.get("modelo_sha256") != firma:
        raise ValueError("Los espectros no corresponden al modelo activo.")

    bordes = np.asarray(datos.get("bordes_hz", []), dtype=float)
    bordes_modelo = np.asarray(modelo["bordes_hz"], dtype=float)
    if bordes.shape != bordes_modelo.shape or not np.array_equal(bordes, bordes_modelo):
        raise ValueError("Los cortes de los espectros no coinciden con el modelo activo.")
    puntos_fft = int(modelo["fs"] * modelo["duracion"])
    if datos.get("fft_puntos") != puntos_fft:
        raise ValueError("La longitud de la FFT no coincide con el modelo activo.")

    frecuencias = np.asarray(datos.get("frecuencias_hz", []), dtype=float)
    if (frecuencias.ndim != 1 or not 2 <= len(frecuencias) <= MAX_PUNTOS
            or not np.all(np.isfinite(frecuencias))
            or not np.all(np.diff(frecuencias) > 0)
            or frecuencias[0] < 0 or frecuencias[-1] >= modelo["fs"]):
        raise ValueError("Las frecuencias de la FFT deben ser crecientes y estar entre 0 y fs.")

    magnitudes = datos.get("magnitudes")
    if not isinstance(magnitudes, dict) or set(magnitudes) != set(modelo["especies"]):
        raise ValueError("La gráfica debe contener las mismas especies que el modelo.")
    series = {}
    for especie, valores in magnitudes.items():
        magnitud = np.asarray(valores, dtype=float)
        if (magnitud.shape != frecuencias.shape
                or not np.all(np.isfinite(magnitud)) or np.any(magnitud < 0)):
            raise ValueError(f"Magnitudes inválidas para {especie}.")
        series[especie] = magnitud

    return frecuencias, series
