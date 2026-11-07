import numpy as np


def calcular_fft(x, fs):
    X = np.fft.rfft(x)
    return np.fft.rfftfreq(len(x), d=1.0 / fs), np.abs(X)
