"""Fase 3: cargar un modelo existente, reconocer y medir su error.

Esta fase nunca entrena ni modifica el archivo del modelo.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from banco_filtros import (cargar_audio, dentro_de_intervalo, hay_canto, recortar,
                           validar_bordes)
from config import DURACION, FS, MODELO, PREFILTRO_FC1, PREFILTRO_FC2
from algoritmo_reconocimiento import VECTORES, reconocer_especie


def adaptar_modelo_provisional(modelo):
    """Lee el JSON provisional anterior sin alterar sus vectores ni su archivo."""
    if "version" in modelo or "firma" not in modelo:
        return modelo

    firma = modelo["firma"]
    especies = {
        "paloma": {"nombre": "Paloma torcaza", "cientifico": "Zenaida auriculata"},
        "copeton": {"nombre": "Copetón", "cientifico": "Zonotrichia capensis"},
    }
    return {
        "version": 1,
        "fs": firma["fs"],
        "duracion": DURACION,
        "prefiltro_hz": firma["prefiltro"],
        "bordes_hz": firma["bordes"],
        "especies": especies,
        "vectores_energia": modelo["vectores"],
        "archivos_entrenamiento": firma.get("archivos", {}),
    }


def cargar_modelo(ruta=MODELO):
    """Lee y valida el contrato que la fase 2 entrega a esta fase."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta primero la fase 2.")
    modelo = adaptar_modelo_provisional(json.loads(ruta.read_text(encoding="utf-8")))
    if modelo.get("version") != 1:
        raise ValueError("Versión de modelo no compatible.")
    if (modelo.get("fs") != FS or modelo.get("duracion") != DURACION
            or modelo.get("prefiltro_hz") != [PREFILTRO_FC1, PREFILTRO_FC2]):
        raise ValueError("El modelo no usa los parámetros de captura y filtro actuales.")

    bordes = validar_bordes(modelo.get("bordes_hz", []))
    especies = modelo.get("especies", {})
    vectores = modelo.get("vectores_energia", {})
    if (not isinstance(especies, dict) or not isinstance(vectores, dict)
            or len(especies) != 2 or set(especies) != set(vectores)):
        raise ValueError("El modelo debe contener dos especies y sus vectores de energía.")
    for especie, datos in especies.items():
        if not isinstance(datos, dict) or not datos.get("nombre"):
            raise ValueError(f"Falta el nombre de la especie {especie}.")
        vector = np.asarray(vectores[especie], dtype=float)
        if (vector.ndim != 1 or len(vector) != len(bordes) - 1
                or not np.all(np.isfinite(vector)) or np.any(vector < 0)):
            raise ValueError(f"Vector de energía inválido para {especie}.")
        vectores[especie] = vector
    modelo["bordes_hz"] = bordes
    return modelo


def reconocer(audio, modelo):
    """Compara el audio con ambos vectores y escoge la menor distancia.

    estado = "especie"        se reconoce la especie con menor distancia (método del profesor)
             "sin_canto"      la señal no tiene variación de energía de un canto (ruido o silencio)
             "no_reconocido"  la especie más cercana existe, pero el audio cae fuera de su intervalo
                              promedio ± k·desviación (nota del profesor, pág. 45)
    "especie" siempre trae la más cercana, para poder graficar la comparación.
    """
    especie, energia_audio, distancias, magnitud = reconocer_especie(recortar(audio))
    frecuencias = np.arange(len(magnitud)) * FS / len(magnitud)
    if not hay_canto(audio):
        estado = "sin_canto"
    elif "intervalos" in modelo and not dentro_de_intervalo(
            energia_audio, intervalo_de(modelo, especie)):
        estado = "no_reconocido"
    else:
        estado = "especie"
    return {
        "estado": estado,
        "especie": especie,
        "distancias": distancias,
        "Ex": energia_audio,
        "vectores": VECTORES,
        "bordes_hz": modelo["bordes_hz"],
        "especies": modelo["especies"],
        "mag": magnitud,
        "frecuencias": frecuencias,
    }


def intervalo_de(modelo, especie):
    """Intervalo de energías de una especie, en el formato que espera dentro_de_intervalo."""
    datos = modelo["intervalos"]
    return {"k": datos["k"], "desviacion_minima_db": datos["desviacion_minima_db"],
            "media_db": datos["media_db"][especie],
            "desviacion_db": datos["desviacion_db"][especie]}


def evaluar(banco, modelo):
    """Mide errores solo con los WAV de prueba, separados del entrenamiento."""
    banco = Path(banco)
    resultados = []
    for esperada in modelo["especies"]:
        archivos = sorted((banco / "prueba" / esperada).glob("*.wav"))
        if not archivos:
            raise FileNotFoundError(f"No hay audios de prueba para {esperada}.")
        for archivo in archivos:
            res = reconocer(cargar_audio(archivo), modelo)
            predicha = res["especie"] if res["estado"] == "especie" else res["estado"]
            resultados.append((archivo, esperada, predicha))
    return resultados


def main():
    parser = argparse.ArgumentParser(description="Fase 3: reconocer o evaluar sin entrenar")
    accion = parser.add_mutually_exclusive_group(required=True)
    accion.add_argument("--audio", type=Path, help="Archivo que se va a reconocer")
    accion.add_argument("--evaluar", type=Path, metavar="BANCO",
                       help="Banco con carpeta prueba/<especie>/")
    parser.add_argument("--modelo", type=Path, default=MODELO)
    args = parser.parse_args()
    try:
        modelo = cargar_modelo(args.modelo)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")

    if args.audio:
        resultado = reconocer(cargar_audio(args.audio), modelo)
        especie = resultado["especie"]
        if resultado["estado"] == "sin_canto":
            print("No se detectó canto")
        elif resultado["estado"] == "no_reconocido":
            print(f"No reconocido (lo más cercano: {modelo['especies'][especie]['nombre']})")
        else:
            print(modelo["especies"][especie]["nombre"])
        print(f"Distancias: {resultado['distancias']}")
    else:
        resultados = evaluar(args.evaluar, modelo)
        errores = [fila for fila in resultados if fila[1] != fila[2]]
        for archivo, esperada, predicha in errores:
            print(f"Error: {archivo.name}: {esperada} → {predicha}")
        print(f"Aciertos: {len(resultados) - len(errores)}/{len(resultados)}")
        print(f"Error: {100 * len(errores) / len(resultados):.2f} %")


if __name__ == "__main__":
    main()
