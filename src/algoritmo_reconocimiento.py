import numpy as np

FS = 22050
N = 110250

BORDES_HZ = (200.0, 1500.0, 3000.0, 8000.0)

E_PALOMA = np.array([8.282388157360728, 0.04072474896049113, 0.03084585519185932])
E_COPETON = np.array([0.04558125423452512, 0.147066540513066, 2.143881149001103])

VECTORES = {"paloma": E_PALOMA, "copeton": E_COPETON}

B = [0.5142453945823573, 0.0, -1.0284907891647146, 0.0, 0.5142453945823573]
A = [1.0, -1.0805108879854952, -0.383781977208491, 0.19085573593813998, 0.280300338080451]


def filtrar(x):
    y = np.zeros(len(x))
    for n in range(len(x)):
        acumulado = 0.0
        for k in range(len(B)):
            if n - k >= 0:
                acumulado += B[k] * x[n - k]
        for k in range(1, len(A)):
            if n - k >= 0:
                acumulado -= A[k] * y[n - k]
        y[n] = acumulado / A[0]
    return y


def normalizar(x):
    return x / (np.sqrt(np.sum(x ** 2)) + 1e-12)


def espectro_magnitud(x):
    y = np.fft.fft(x)
    z = np.abs(y)
    return z


def energias_subbandas(z):
    E = []
    for f1, f2 in zip(BORDES_HZ[:-1], BORDES_HZ[1:]):
        k1 = int(np.ceil(f1 * N / FS))
        k2 = int(np.ceil(f2 * N / FS))
        L = k2 - k1
        E.append(np.sum(z[k1:k2] ** 2) / L)
    return np.array(E)


def distancia(Ex, Ec):
    return float(np.sum(np.abs(Ex - Ec)))


def reconocer_especie(x):
    x = normalizar(filtrar(np.asarray(x, dtype=float)))
    z = espectro_magnitud(x)
    Ex = energias_subbandas(z)
    d = {especie: distancia(Ex, Ec) for especie, Ec in VECTORES.items()}
    especie = min(d, key=d.get)
    return especie, Ex, d, z
