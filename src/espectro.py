import numpy as np


def calcular_fft(x, fs):
    X = np.fft.rfft(x)
    return np.fft.rfftfreq(len(x), d=1.0 / fs), np.abs(X)


def calcular_fft_completa(x, fs):
    X = np.fft.fft(x)
    return np.arange(len(x)) * fs / len(x), np.abs(X)
