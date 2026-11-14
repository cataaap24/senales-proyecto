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

            # Método del profesor: energía de cada grabación (1 y 9) y luego el promedio = 5.
            # (Promediar primero los espectros daría 4.)
            self.assertEqual(modelo["vectores_energia"]["paloma"], [5.0] * 3)
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

    def test_ruido_constante_no_es_canto_y_silabas_si(self):
        rng = np.random.default_rng(0)
        n = banco_filtros.N
        ruido = rng.normal(size=n)
        # Sílabas de 0.2 s a 3 kHz separadas por pausas, sobre ruido débil.
        t = np.arange(n) / banco_filtros.FS
        silabas = np.sin(2 * np.pi * 3000 * t) * ((t % 0.5) < 0.2) + 0.01 * rng.normal(size=n)
        with patch.object(banco_filtros, "acondicionar", side_effect=lambda x, normalizar=True: x):
            self.assertFalse(banco_filtros.hay_canto(ruido))
            self.assertTrue(banco_filtros.hay_canto(silabas))

    def test_intervalo_usa_desviacion_minima(self):
        intervalo = {"k": 3.0, "desviacion_minima_db": 1.0,
                     "media_db": [10.0, -20.0], "desviacion_db": [0.1, 5.0]}
        # 10 dB ± 3·max(0.1, 1) = [7, 13];  -20 dB ± 15 = [-35, -5]
        self.assertTrue(banco_filtros.dentro_de_intervalo([10 ** 1.2, 10 ** -3.0], intervalo))
        self.assertFalse(banco_filtros.dentro_de_intervalo([10 ** 1.4, 10 ** -2.0], intervalo))


    def test_ejercicio_del_tablero(self):
        """Ejercicio de clase: A = {(1,0,0,2), (2,1,0,0)}, 2 subbandas iguales de la FFT completa."""
        bordes = [0, 2, 4]                       # fs = 4  ->  la frecuencia de cada punto es k
        vectores = []
        for x in ([1, 0, 0, 2], [2, 1, 0, 0]):
            f, mag = banco_filtros.calcular_fft_completa(np.array(x, float), 4)
            # El ejercicio usa 2 subbandas; el proyecto exige de 3 a 8, así que se omite esa validación.
            with patch.object(banco_filtros, "validar_bordes", side_effect=np.asarray):
                vectores.append(banco_filtros.energias(mag, f, bordes))
        np.testing.assert_allclose(vectores[0], [7, 3])     # E_x1 = [7 3]
        np.testing.assert_allclose(vectores[1], [7, 3])     # E_x2 = [7 3]
        np.testing.assert_allclose(np.mean(vectores, axis=0), [7, 3])   # E_mA = [7 3]
        # Reconocimiento del tablero: Ex = (2 1 1), EA = (7 2 4), EB = (4 5 2)
        ex = np.array([2, 1, 1.0])
        self.assertEqual(banco_filtros.distancia(ex, np.array([7, 2, 4.0])), 9)
        self.assertEqual(banco_filtros.distancia(ex, np.array([4, 5, 2.0])), 7)


if __name__ == "__main__":
    unittest.main()
