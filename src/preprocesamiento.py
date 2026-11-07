from functools import lru_cache

import numpy as np

from config import FS, PREFILTRO_FC1, PREFILTRO_FC2
from filtros import butterworth_bilineal, filtrar_iir
from captura import cargar_wav


@lru_cache(maxsize=None)
def coeficientes_prefiltro():
    b, a = butterworth_bilineal(2, (PREFILTRO_FC1, PREFILTRO_FC2), FS)
    return tuple(b), tuple(a)


def acondicionar(x, normalizar=True):
    b, a = coeficientes_prefiltro()
    y = filtrar_iir(np.asarray(x, float), np.array(b), np.array(a))
    if normalizar:
        y = y / (np.sqrt(np.sum(y ** 2)) + 1e-12)
    return y


def cargar_y_acondicionar(ruta, normalizar=True):
    return acondicionar(cargar_wav(ruta), normalizar)
