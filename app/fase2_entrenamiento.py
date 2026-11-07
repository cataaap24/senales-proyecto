"""Fase 2: construir el modelo; nunca se ejecuta al abrir la interfaz.

Uso: python app/fase2_entrenamiento.py --bordes 0,2000,4000,6000,8000,10000
Los bordes iniciales son una base ajustable, no una selección definitiva.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from banco_filtros import cargar_audio, energias, espectro, validar_bordes
from config import BANCO, DURACION, FS, MODELO, PREFILTRO_FC1, PREFILTRO_FC2


# Añadir grabaciones al banco no exige cambiar código. Estos nombres solo
# describen las dos clases que se estudian en el proyecto.
ESPECIES = {
    "paloma": {"nombre": "Paloma torcaza", "cientifico": "Zenaida auriculata"},
    "copeton": {"nombre": "Copetón", "cientifico": "Zonotrichia capensis"},
}
BORDES_INICIALES = (0, 2000, 4000, 6000, 8000, 10000)


def entrenar(banco=BANCO, bordes=BORDES_INICIALES, progreso=None):
    """Obtiene un espectro promedio y su vector de energías por especie."""
    banco = Path(banco)
    bordes = validar_bordes(bordes)
    rutas = {
        especie: sorted((banco / "entrenamiento" / especie).glob("*.wav"))
        for especie in ESPECIES
    }
    if any(not archivos for archivos in rutas.values()):
        raise FileNotFoundError("Faltan WAV de entrenamiento para alguna especie.")

    total = sum(map(len, rutas.values()))
    completados = 0
    vectores = {}
    promedios = {}
    frecuencias = None

    for especie, archivos in rutas.items():
        espectros = []
        for archivo in archivos:
            frecuencias, magnitud = espectro(cargar_audio(archivo))
            espectros.append(magnitud)
            completados += 1
            if progreso is not None:
                progreso(completados, total)

        # Primero se promedian los espectros; después se calcula la energía
        # del espectro promedio, tal como indica la secuencia del README.
        promedio = np.sum(espectros, axis=0) / len(espectros)
        promedios[especie] = promedio
        vectores[especie] = energias(promedio, frecuencias, bordes).tolist()

    modelo = {
        "version": 1,
        "fs": FS,
        "duracion": DURACION,
        "prefiltro_hz": [PREFILTRO_FC1, PREFILTRO_FC2],
        "bordes_hz": bordes.tolist(),
        "especies": ESPECIES,
        "vectores_energia": vectores,
        "archivos_entrenamiento": {especie: len(archivos) for especie, archivos in rutas.items()},
    }
    return modelo, frecuencias, promedios


def guardar_modelo(modelo, destino=MODELO):
    """Escribe el modelo solo cuando se invoca explícitamente la fase 2."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(destino.name + ".tmp")
    temporal.write_text(json.dumps(modelo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporal.replace(destino)
    return destino


def guardar_grafica(frecuencias, promedios, destino):
    """Dibuja los dos espectros para revisar dónde conviene cortar las bandas."""
    import matplotlib.pyplot as plt

    figura, eje = plt.subplots(figsize=(10, 4))
    for especie, magnitud in promedios.items():
        eje.plot(frecuencias, magnitud, label=ESPECIES[especie]["nombre"])
    eje.set(xlabel="Frecuencia (Hz)", ylabel="Magnitud |C(k)|",
            xlim=(0, float(frecuencias[-1])), title="Espectros promedio de entrenamiento")
    eje.legend()
    figura.tight_layout()
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=150)
    plt.close(figura)


def main():
    parser = argparse.ArgumentParser(description="Fase 2: entrenar el banco de filtros")
    parser.add_argument("--banco", type=Path, default=BANCO)
    parser.add_argument("--salida", type=Path, default=MODELO)
    parser.add_argument("--bordes", default=",".join(map(str, BORDES_INICIALES)),
                        help="Límites en Hz, separados por comas; definen de 3 a 8 subbandas")
    parser.add_argument("--grafica", type=Path,
                        help="Ruta opcional para guardar los espectros promedio")
    args = parser.parse_args()

    bordes = [float(valor) for valor in args.bordes.split(",")]
    modelo, frecuencias, promedios = entrenar(args.banco, bordes)
    destino = guardar_modelo(modelo, args.salida)
    if args.grafica:
        guardar_grafica(frecuencias, promedios, args.grafica)
    print(f"Modelo guardado en {destino}")
    print(f"Audios por especie: {modelo['archivos_entrenamiento']}")


if __name__ == "__main__":
    main()
