import numpy as np


def calcular_fft(x, fs):
    X = np.fft.rfft(x)
    return np.fft.rfftfreq(len(x), d=1.0 / fs), np.abs(X)


def calcular_fft_completa(x, fs):
    """FFT completa de N puntos, como en clase: y = fft(x), z = abs(y), k = 0 .. N-1.
    La frecuencia de cada punto es k*fs/N (de 0 a fs). La segunda mitad es el espejo de la primera
    porque la señal es real."""
    X = np.fft.fft(x)
    return np.arange(len(x)) * fs / len(x), np.abs(X)
