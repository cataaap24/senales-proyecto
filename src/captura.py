import numpy as np

from config import FS, DURACION
from filtros import fir_ventaneado, filtrar_fir


def grabar(duracion=DURACION, fs=FS):
    import sounddevice as sd

    n = int(duracion * fs)
    try:
        x = sd.rec(n, samplerate=fs, channels=1, dtype="float32")
        sd.wait()
    except sd.PortAudioError:
        fs_hw = 2 * fs
        x = sd.rec(int(duracion * fs_hw), samplerate=fs_hw, channels=1, dtype="float32")
        sd.wait()
        return diezmar(x[:, 0].astype(float), 2, fs_hw)
    x = x[:, 0].astype(float)
    if np.max(np.abs(x)) < 1e-4:
        print("AVISO: la señal es casi nula. Revisa el micrófono predeterminado y los permisos de "
              "micrófono.")
    return x


def cargar_wav(ruta, fs=FS):
    import soundfile as sf

    x, fs_archivo = sf.read(str(ruta), dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs_archivo != fs:
        raise ValueError(f"{ruta}: está a {fs_archivo} Hz y el proyecto trabaja a {fs} Hz. ")
    return x


def diezmar(x, M, fs_in):
    fc = 0.45 * fs_in / M
    h = fir_ventaneado("lp", 101, fc, fs_in, "hamming")
    return filtrar_fir(x, h)[::M]


def reproducir(x, fs=FS):
    import sounddevice as sd

    sd.play(0.9 * x / (np.max(np.abs(x)) + 1e-12), fs)
    sd.wait()
