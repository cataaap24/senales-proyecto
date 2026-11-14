"""Comprueba el contrato de lectura de las curvas, sin ejecutar entrenamiento."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from espectros_guardados import cargar_espectros, guardar_espectros


class PruebasEspectrosGuardados(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        carpeta = Path(self.temporal.name)
        self.ruta_modelo = carpeta / "modelo.json"
        self.ruta_modelo.write_text('{"modelo":"ejemplo"}\n', encoding="utf-8")
        self.ruta_espectros = carpeta / "espectros.json"
        self.modelo = {
            "fs": 8000,
            "duracion": 0.001,
            "bordes_hz": [200, 1500, 3000, 8000],
            "especies": {"paloma": {}, "copeton": {}},
        }
        self.datos = {
            "version": 2,
            "modelo_sha256": hashlib.sha256(self.ruta_modelo.read_bytes()).hexdigest(),
            "fft_puntos": 8,
            "bordes_hz": [200, 1500, 3000, 8000],
            "frecuencias_hz": [0, 1000, 4000, 7000],
            "magnitudes": {"paloma": [1, 3, 1, 0], "copeton": [0, 1, 2, 1]},
        }

    def cargar(self):
        self.ruta_espectros.write_text(json.dumps(self.datos), encoding="utf-8")
        return cargar_espectros(self.modelo, self.ruta_espectros, self.ruta_modelo)

    def test_lee_las_dos_curvas_sin_modificar_archivos(self):
        frecuencias, series = self.cargar()
        self.assertEqual(frecuencias.tolist(), [0, 1000, 4000, 7000])
        self.assertEqual(series["copeton"].tolist(), [0, 1, 2, 1])
        self.assertEqual(json.loads(self.ruta_espectros.read_text()), self.datos)

    def test_exporta_los_promedios_y_los_vincula_al_modelo(self):
        frecuencias = np.arange(8) * 1000.0
        promedios = {
            "paloma": np.linspace(0, 4, len(frecuencias)),
            "copeton": np.linspace(2, 0, len(frecuencias)),
        }
        with patch("espectros_guardados.MAX_PUNTOS", 4):
            guardar_espectros(self.modelo, frecuencias, promedios,
                              self.ruta_espectros, self.ruta_modelo)
        eje, series = cargar_espectros(self.modelo, self.ruta_espectros, self.ruta_modelo)
        self.assertEqual(len(eje), 4)
        self.assertEqual(len(series["paloma"]), 4)
        self.assertAlmostEqual(series["paloma"][0], np.mean(promedios["paloma"][:2]))
        self.assertAlmostEqual(series["copeton"][-1], np.mean(promedios["copeton"][-2:]))

    def test_rechaza_curvas_de_otro_modelo(self):
        self.datos["modelo_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "modelo activo"):
            self.cargar()

    def test_rechaza_frecuencias_desordenadas(self):
        self.datos["frecuencias_hz"] = [0, 4000, 1000, 7000]
        with self.assertRaisesRegex(ValueError, "frecuencias"):
            self.cargar()


if __name__ == "__main__":
    unittest.main()
