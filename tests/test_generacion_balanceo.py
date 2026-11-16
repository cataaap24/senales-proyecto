"""Pruebas del generador de balanceo: frases reales reordenadas y ecualización."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "src"), str(RAIZ / "herramientas")]

import generar_balanceo as gb
from config import FS, N


def audio_con_tres_frases():
    """Tres tonos de 0,2 s separados por pausas de 0,3 s."""
    audio = np.zeros(N)
    for inicio in (0.5, 2.0, 3.5):
        a = int(inicio * FS)
        b = a + int(0.2 * FS)
        t = np.arange(b - a) / FS
        audio[a:b] = 0.5 * np.sin(2 * np.pi * 1000 * t)
    return audio


class PruebasGeneracionBalanceo(unittest.TestCase):
    def test_detectar_frases_encuentra_las_tres_silabas(self):
        frases = gb.detectar_frases(audio_con_tres_frases())

        self.assertEqual(len(frases), 3)
        for inicio, fin in frases:
            self.assertGreater(fin - inicio, int(0.15 * FS))

    def test_detectar_frases_no_inventa_frases_en_silencio(self):
        self.assertEqual(gb.detectar_frases(np.zeros(N)), [])

    def test_reordenar_frases_conserva_duracion_pico_y_semilla(self):
        audio = audio_con_tres_frases()
        primero = gb.reordenar_frases(audio, np.random.default_rng(7))
        segundo = gb.reordenar_frases(audio, np.random.default_rng(7))

        np.testing.assert_array_equal(primero, segundo)
        self.assertEqual(len(primero), N)
        self.assertTrue(np.all(np.isfinite(primero)))
        self.assertAlmostEqual(float(np.max(np.abs(primero))), 0.5, places=4)
        self.assertFalse(np.array_equal(primero, audio))

    def test_reordenar_frases_devuelve_none_si_no_hay_frases(self):
        self.assertIsNone(gb.reordenar_frases(np.zeros(N), np.random.default_rng(1)))

    def test_ecualizar_con_ganancia_constante_es_un_escalado(self):
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(N) * 0.1

        np.testing.assert_allclose(gb.ecualizar(audio, [0.0, 0.0, 0.0]), audio,
                                   rtol=0, atol=1e-15)
        ganancia = 0.5
        esperado = audio * 10 ** (ganancia / 20)
        np.testing.assert_allclose(gb.ecualizar(audio, [ganancia] * 3), esperado,
                                   rtol=1e-12, atol=1e-14)

    def test_la_curva_de_ganancia_pasa_por_los_centros(self):
        ganancias = np.array([1.0, -2.0, 0.5])
        np.testing.assert_allclose(
            gb.curva_ganancia(gb.CENTROS_HZ, ganancias),
            10 ** (ganancias / 20), rtol=1e-12)
        self.assertTrue(np.all(gb.curva_ganancia(
            np.array([0.0, 100.0, 20000.0]), ganancias) > 0))

    def test_normalizar_pico(self):
        audio = np.array([0.1, -0.2, 0.05])
        normalizado = gb.normalizar_pico(audio)

        self.assertAlmostEqual(float(np.max(np.abs(normalizado))), 0.9, places=12)
        np.testing.assert_allclose(normalizado / np.max(np.abs(normalizado)),
                                   audio / np.max(np.abs(audio)))

    def test_fraccion_banda_coincide_con_la_fft(self):
        rng = np.random.default_rng(3)
        audio = rng.standard_normal(N) * 0.1
        energia = gb.energia_bandas_rapida(audio)

        filtrado = lfilter(gb._B, gb._A, audio)
        filtrado = filtrado / np.sqrt(np.sum(filtrado ** 2))
        magnitud = np.abs(np.fft.fft(filtrado)) ** 2
        frecuencias = np.arange(N) * FS / N
        dentro = np.zeros(N, dtype=bool)
        for inferior, superior in zip(gb.BORDES[:-1], gb.BORDES[1:]):
            dentro |= (frecuencias >= inferior) & (frecuencias < superior)
        esperado = float(np.sum(magnitud[dentro]) / np.sum(magnitud))

        self.assertAlmostEqual(gb.fraccion_banda(energia), esperado, places=12)
        np.testing.assert_array_equal(gb.BINS_BANDA, [6500, 7500, 25000])

    def test_error_de_momentos_usa_db_absolutos(self):
        energias = np.ones((10, 3))
        objetivo = gb.momentos(energias)
        desplazadas = energias.copy()
        desplazadas[:, 1] *= 10 ** (0.1 / 10)

        # El error dominante es la media dB de B2: 0.1 dB.
        self.assertAlmostEqual(gb.error_momentos(desplazadas, objetivo), 0.1,
                               places=10)
        self.assertEqual(gb.error_momentos(energias, objetivo), 0.0)

    def test_escribir_y_medir_los_wav_generados(self):
        rng = np.random.default_rng(5)
        base = [rng.standard_normal(N) * 0.1]
        with tempfile.TemporaryDirectory() as temporal:
            destino = Path(temporal)
            rutas = gb.escribir_variantes(base, np.zeros((1, 3)), destino)

            self.assertEqual(len(rutas), 1)
            info = sf.info(str(rutas[0]))
            self.assertEqual(info.frames, N)
            self.assertEqual(info.samplerate, FS)
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.subtype, "PCM_16")
            self.assertAlmostEqual(float(np.max(np.abs(
                gb.cargar_audio(rutas[0])))), 0.9, places=4)

            energia = gb.medir_archivos(rutas)
            self.assertEqual(energia.shape, (1, 3))
            self.assertTrue(np.all(np.isfinite(energia)))

    def test_solo_usa_la_carpeta_de_entrenamiento(self):
        contenido = (RAIZ / "herramientas" / "generar_balanceo.py").read_text(
            encoding="utf-8")
        self.assertIn('BANCO / "entrenamiento" / "paloma"', contenido)
        self.assertNotIn('BANCO / "prueba"', contenido)
        self.assertNotIn("banco_aves/prueba", contenido)


if __name__ == "__main__":
    unittest.main()
