import numpy as np
from numpy.polynomial import polynomial as P

def ventana(nombre, N):
    if nombre != "hamming":
        raise ValueError(f"ventana no incluida en la entrega: {nombre}")
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))


def h_ideal(tipo, N, fc, fs):
    if tipo != "lp":
        raise ValueError(f"tipo no incluido en la entrega: {tipo}")
    assert N % 2 == 1, "el orden N de un FIR por ventanas debe ser impar"
    m = (N - 1) // 2
    n = np.arange(-m, m + 1)
    nz = n != 0               
    Wc = 2 * np.pi * fc / fs  
    h = np.zeros(N)
    h[nz] = np.sin(Wc * n[nz]) / (np.pi * n[nz])
    h[m] = Wc / np.pi
    return h


def fir_ventaneado(tipo, N, fc, fs, nombre_ventana="hamming"):
    return h_ideal(tipo, N, fc, fs) * ventana(nombre_ventana, N)


def convolucion(x, h):
    return np.convolve(x, h)


def filtrar_fir(x, h):
    y = convolucion(x, h)
    r = (len(h) - 1) // 2
    return y[r:r + len(x)]

def polinomio_butterworth(N):
    return {2: [1.0, np.sqrt(2), 1.0]}[N]


def transformar_analogico(N, wd):
    B = polinomio_butterworth(N)
    w1, w2 = wd
    w0, BW = np.sqrt(w1 * w2), w2 - w1
    den = np.zeros(2 * N + 1)
    for k in range(N + 1):
        termino = P.polymul(P.polypow([w0 ** 2, 0.0, 1.0], k), P.polypow([0.0, BW], N - k))
        den = P.polyadd(den, B[k] * termino)
    num = P.polypow([0.0, BW], N)
    return np.array(num, float), np.array(den, float)


def bilineal(num, den, fs):
    c = 2.0 * fs
    D = max(len(num), len(den)) - 1

    def mapear(q):
        out = np.zeros(D + 1)
        for m, qm in enumerate(q):
            if qm != 0:
                out += qm * c ** m * P.polymul(P.polypow([1.0, -1.0], m), P.polypow([1.0, 1.0], D - m))
        return out

    b, a = mapear(num), mapear(den)
    return b / a[0], a / a[0]


def butterworth_bilineal(N, fc, fs):
    c = 2.0 * fs
    wd = (c * np.tan(np.pi * fc[0] / fs), c * np.tan(np.pi * fc[1] / fs))
    num, den = transformar_analogico(N, wd)
    return bilineal(num, den, fs)


def filtrar_iir(x, b, a):
    y = np.zeros(len(x))
    for n in range(len(x)):
        acc = 0.0
        for k in range(len(b)):
            if n - k >= 0:
                acc += b[k] * x[n - k]
        for k in range(1, len(a)):
            if n - k >= 0:
                acc -= a[k] * y[n - k]
        y[n] = acc / a[0]
    return y

def respuesta_frecuencia(b, a, fs, puntos=2048):
    Om = np.linspace(0, np.pi, puntos)
    num = np.exp(-1j * np.outer(Om, np.arange(len(b)))) @ np.asarray(b, float)
    den = np.exp(-1j * np.outer(Om, np.arange(len(a)))) @ np.asarray(a, float)
    return Om * fs / (2 * np.pi), np.abs(num / den)
