"""
piloto_aves.py - USASO PARA: elegir la pareja de especies (no es codigo del proyecto final)

Compara  paloma vs copeton  y  paloma vs siriri (FFT -> magnitud -> espectro promedio -> energia por subbanda ->
suma de diferencias absolutas), y mide cual pareja se reconoce mejor
"""
import os
import sys
import glob
import itertools
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

FS = 22050
DUR = 5.0 
N = int(FS * DUR)
HP = 150
FMAX = 10000 
N_SUBBANDAS = range(3, 9)

FREQS = np.fft.rfftfreq(N, 1 / FS)
_SOS = butter(4, HP, btype="highpass", fs=FS, output="sos")


def espectro(ruta):
    """Devuelve |X(k)| de una grabacion ya recortada y normalizada."""
    x, _ = librosa.load(ruta, sr=FS, mono=True)
    x = sosfiltfilt(_SOS, x)
    if len(x) < N:
        x = np.pad(x, (0, N - len(x)))
    c = np.concatenate(([0.0], np.cumsum(x ** 2)))
    i = int(np.argmax(c[N:] - c[:-N]))
    x = x[i:i + N]

    x = x / (np.sqrt(np.sum(x ** 2)) + 1e-12)
    return np.abs(np.fft.rfft(x))


def energias(Xmag, bordes):
    """E_i = (1/L) * sum |X(k)|^2 dentro de la subbanda i"""
    E = []
    for a, b in zip(bordes[:-1], bordes[1:]):
        m = (FREQS >= a) & (FREQS < b)
        E.append(np.mean(Xmag[m] ** 2))
    return np.array(E)


def distancias(Ex, vectores):
    """Suma de diferencias absolutas contra el vector de cada especie"""
    return {esp: np.sum(np.abs(Ex - Ev)) for esp, Ev in vectores.items()}


def clasificar(Ex, vectores):
    """Gana la especie con la menor suma de diferencias absolutas"""
    d = distancias(Ex, vectores)
    return min(d, key=d.get)


# ---------------- Metricas ----------------
def perfil(Xmag, ancho=100):
    """Energia en bandas finas de 100 Hz, normalizada"""
    bordes = np.arange(0, FMAX + ancho, ancho)
    p = energias(Xmag, bordes)
    return p / p.sum()


def consistencia(espectros):
    """Correlacion media de cada grabacion con el promedio de las demas
    Alta = el canto se parece entre grabaciones -> espectro promedio fiabl"""
    P = np.array([perfil(X) for X in espectros])
    r = []
    for i in range(len(P)):
        resto = np.delete(P, i, axis=0).mean(axis=0)
        r.append(np.corrcoef(P[i], resto)[0, 1])
    return float(np.mean(r))


def solapamiento(espA, espB):
    """Area comun de los perfiles promedio (0 = bandas disjuntas, 1 = iguales)"""
    pa = np.mean([perfil(X) for X in espA], axis=0)
    pb = np.mean([perfil(X) for X in espB], axis=0)
    return float(np.sum(np.minimum(pa, pb)))


def exactitud_loo(bancos, n):
    """Leave-one-out: cada grabacion se prueba contra vectores entrenados
    sin ella. Desempata cuando ambas parejas dan 100%"""
    bordes = np.linspace(0, FMAX, n + 1)
    aciertos = total = 0
    margenes = []
    for esp, lista in bancos.items():
        for i in range(len(lista)):
            vect = {}
            for e2, l2 in bancos.items():
                train = [X for j, X in enumerate(l2) if not (e2 == esp and j == i)]
                vect[e2] = energias(np.mean(train, axis=0), bordes)
            Ex = energias(lista[i], bordes)
            d = distancias(Ex, vect)
            d_otra = min(v for k, v in d.items() if k != esp)
            margenes.append((d_otra - d[esp]) / (d_otra + d[esp] + 1e-12))
            aciertos += clasificar(Ex, vect) == esp
            total += 1
    return aciertos / total, float(np.mean(margenes))


def suavizar(v, w=200):
    return np.convolve(v, np.ones(w) / w, mode="same")


def graficar(bancos, nombre, n):
    plt.figure(figsize=(10, 4))
    for esp, lista in bancos.items():
        plt.plot(FREQS, suavizar(np.mean(lista, axis=0)), label=esp)
    for f in np.linspace(0, FMAX, n + 1)[1:-1]:
        plt.axvline(f, color="k", ls="--", lw=0.8)
    plt.xlim(0, FMAX)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("|X(k)| promedio")
    plt.title(f"Espectros promedio - {nombre} ({n} subbandas iguales)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"espectros_{nombre}.png", dpi=130)
    plt.close()


def main(raiz):
    bancos = {}
    for d in sorted(os.listdir(raiz)):
        rutas = sorted(glob.glob(os.path.join(raiz, d, "*.mp3")) +
                       glob.glob(os.path.join(raiz, d, "*.wav")))
        if rutas:
            print(f"Cargando {d}: {len(rutas)} grabaciones")
            bancos[d] = [espectro(r) for r in rutas]

    print("\nCONSISTENCIA por especie (mas alto = mejor, maximo 1)")
    for esp, lista in bancos.items():
        print(f"  {esp:10s} {consistencia(lista):.3f}")

    resultados = []
    for a, b in itertools.combinations(bancos, 2):
        par = {a: bancos[a], b: bancos[b]}
        nombre = f"{a}_vs_{b}"
        res = {n: exactitud_loo(par, n) for n in N_SUBBANDAS}
        acc = {n: r[0] for n, r in res.items()}
        marg = float(np.mean([r[1] for r in res.values()]))
        mejor_n = max(res, key=lambda n: res[n])
        sol = solapamiento(bancos[a], bancos[b])
        resultados.append((nombre, np.mean(list(acc.values())), sol, acc, mejor_n, marg))
        graficar(par, nombre, mejor_n)

    print("\nPAREJAS (exactitud leave-one-out por numero de subbandas)")
    print("  " + "pareja".ljust(22) + "".join(f"n={n:<6}" for n in N_SUBBANDAS)
          + "media   margen  solapamiento")
    for nombre, media, sol, acc, _, marg in resultados:
        fila = "".join(f"{acc[n]:<8.2f}" for n in N_SUBBANDAS)
        print(f"  {nombre:22s}{fila}{media:<8.2f}{marg:<8.3f}{sol:.3f}")

    if len(bancos) >= 3:
        acc3 = {n: exactitud_loo(bancos, n)[0] for n in N_SUBBANDAS}
        print("\nTRES ESPECIES juntas: " +
              "  ".join(f"n={n}: {v:.2f}" for n, v in acc3.items()))
        graficar(bancos, "tres_especies", max(acc3, key=acc3.get))

    resultados.sort(key=lambda r: (-r[1], -r[5], r[2]))
    print(f"\nMejor pareja en este piloto: {resultados[0][0]}")
    print("Graficas guardadas como espectros_*.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "banco_piloto")