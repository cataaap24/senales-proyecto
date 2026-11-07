"""Comprueba el contrato entre entrenamiento y reconocimiento sin usar el banco real."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "src"), str(RAIZ / "app")]

import banco_filtros
import fase2_entrenamiento
import fase3_reconocimiento


class PruebasDeFases(unittest.TestCase):
    def test_energia_por_banda_sigue_la_formula(self):
        frecuencias = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        magnitudes = np.array([1, 3, 2, 4, 3, 5], dtype=float)
        bordes = [0, 2, 4, 6]

        resultado = banco_filtros.energias(magnitudes, frecuencias, bordes)

        np.testing.assert_allclose(resultado, [5, 10, 17])

    def test_entrenar_guarda_un_modelo_que_fase3_puede_cargar(self):
        with tempfile.TemporaryDirectory() as temporal:
            banco = Path(temporal) / "banco"
            for especie in ("paloma", "copeton"):
                carpeta = banco / "entrenamiento" / especie
                carpeta.mkdir(parents=True)
                for indice in (1, 2):
                    (carpeta / f"{especie}_{indice}.wav").touch()

            frecuencias = np.array([0, 1, 2, 3, 4, 5], dtype=float)
            magnitudes = {
                "paloma_1.wav": np.array([1, 1, 1, 1, 1, 1], float),
                "paloma_2.wav": np.array([3, 3, 3, 3, 3, 3], float),
                "copeton_1.wav": np.array([2, 2, 2, 2, 2, 2], float),
                "copeton_2.wav": np.array([4, 4, 4, 4, 4, 4], float),
            }
            with patch.object(fase2_entrenamiento, "cargar_audio",
                              side_effect=lambda ruta: Path(ruta).name), \
                 patch.object(fase2_entrenamiento, "espectro",
                              side_effect=lambda nombre: (frecuencias, magnitudes[nombre])):
                modelo, _, _ = fase2_entrenamiento.entrenar(banco, [0, 2, 4, 6])

            # Promediar 1 y 3 antes de elevar al cuadrado da 4, no 5.
            self.assertEqual(modelo["vectores_energia"]["paloma"], [4.0] * 3)
            destino = Path(temporal) / "modelo.json"
            fase2_entrenamiento.guardar_modelo(modelo, destino)
            cargado = fase3_reconocimiento.cargar_modelo(destino)
            self.assertEqual(len(cargado["bordes_hz"]) - 1, 3)
            self.assertEqual(set(cargado["especies"]), {"paloma", "copeton"})

    def test_fase3_no_entrena_si_falta_el_modelo(self):
        with tempfile.TemporaryDirectory() as temporal:
            destino = Path(temporal) / "modelo.json"
            with self.assertRaises(FileNotFoundError):
                fase3_reconocimiento.cargar_modelo(destino)
            self.assertFalse(destino.exists())

    def test_modelo_provisional_se_lee_sin_reescribirlo(self):
        anterior = {
            "firma": {
                "fs": fase3_reconocimiento.FS,
                "prefiltro": [200.0, 8000.0],
                "bordes": [0, 2000, 4000, 6000],
                "archivos": {"paloma": 1, "copeton": 1},
            },
            "vectores": {"paloma": [1, 2, 3], "copeton": [3, 2, 1]},
            "umbrales": {"paloma": 4, "copeton": 4},
        }
        with tempfile.TemporaryDirectory() as temporal:
            destino = Path(temporal) / "modelo.json"
            contenido = json.dumps(anterior)
            destino.write_text(contenido, encoding="utf-8")

            modelo = fase3_reconocimiento.cargar_modelo(destino)

            self.assertEqual(modelo["version"], 1)
            self.assertEqual(len(modelo["bordes_hz"]) - 1, 3)
            self.assertEqual(destino.read_text(encoding="utf-8"), contenido)


if __name__ == "__main__":
    unittest.main()
