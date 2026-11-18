import json
import sys
import unittest
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "src"), str(RAIZ / "app")]

import algoritmo_reconocimiento as alg
import banco_filtros
from filtros import butterworth_bilineal


class PruebasAlgoritmoExplicito(unittest.TestCase):
    def test_constantes_coinciden_con_el_modelo_entrenado(self):
        datos = json.loads((RAIZ / "modelo" / "vectores_energia.json").read_text(encoding="utf-8"))
        np.testing.assert_allclose(alg.E_PALOMA, datos["vectores_energia"]["paloma"])
        np.testing.assert_allclose(alg.E_COPETON, datos["vectores_energia"]["copeton"])
        np.testing.assert_allclose(alg.BORDES_HZ, datos["bordes_hz"])

    def test_filtro_es_el_butterworth_bilineal(self):
        b, a = butterworth_bilineal(2, (200.0, 8000.0), 22050)
        np.testing.assert_allclose(alg.B, b)
        np.testing.assert_allclose(alg.A, a)

    def test_mismas_energias_que_el_entrenamiento(self):
        x = np.random.default_rng(0).normal(size=alg.N)
        f, mag = banco_filtros.espectro(x)
        esperado = banco_filtros.energias(mag, f, alg.BORDES_HZ)
        _, Ex, _, _ = alg.reconocer_especie(x)
        np.testing.assert_allclose(Ex, esperado, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
