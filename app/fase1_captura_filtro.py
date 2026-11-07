import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

import captura
from config import FS, DURACION, FIGURAS, PREFILTRO_FC1, PREFILTRO_FC2
from espectro import calcular_fft
from filtros import butterworth_bilineal, filtrar_iir, respuesta_frecuencia


def porcentaje_fuera(f, X, fc1, fc2):
    fuera = (f < fc1) | (f > fc2)
    return 100 * np.sum(X[fuera] ** 2) / np.sum(X ** 2)


def main():
    ap = argparse.ArgumentParser(description="Captura y filtrado digital")
    ap.add_argument("--wav", help="usar este archivo en vez del micrófono")
    ap.add_argument("--dur", type=float, default=DURACION, help="segundos a grabar")
    ap.add_argument("--fc1", type=float, default=PREFILTRO_FC1)
    ap.add_argument("--fc2", type=float, default=PREFILTRO_FC2)
    ap.add_argument("--reproducir", action="store_true")
    ap.add_argument("--no-mostrar", action="store_true", help="solo guardar la figura")
    args = ap.parse_args()

    if args.wav:
        x = captura.cargar_wav(args.wav)
        origen = Path(args.wav).name
    else:
        print(f"Grabando {args.dur:.0f} s: acerca el sonido al micrófono...")
        x = captura.grabar(args.dur)
        origen = "micrófono"

    b, a = butterworth_bilineal(2, (args.fc1, args.fc2), FS)
    y = filtrar_iir(x, b, a)
    nombre = f"Butterworth pasabandas orden 2 (bilineal), {args.fc1:.0f}-{args.fc2:.0f} Hz"
    print("Ecuación en diferencias  a0 y(n) + a1 y(n-1) + ... = b0 x(n) + b1 x(n-1) + ...")
    print("  b =", np.round(b, 6))
    print("  a =", np.round(a, 6))

    f, X = calcular_fft(x, FS)
    _, Y = calcular_fft(y, FS)
    fh, H = respuesta_frecuencia(b, a, FS)

    print(f"\nEnergía fuera de [{args.fc1:.0f}, {args.fc2:.0f}] Hz:  "
          f"antes {porcentaje_fuera(f, X, args.fc1, args.fc2):.1f} %   "
          f"después {porcentaje_fuera(f, Y, args.fc1, args.fc2):.1f} %")

    t = np.arange(len(x)) / FS
    fig, ax = plt.subplots(4, 1, figsize=(10, 12))
    ax[0].plot(t, x, lw=0.6, label="original", alpha=0.8)
    ax[0].plot(t, y, lw=0.6, label="filtrada")
    ax[0].set(title=f"Señal en el tiempo - {origen}", xlabel="Tiempo (s)", ylabel="Amplitud")
    ax[0].legend(loc="upper right")

    ax[1].plot(f, X, lw=0.7, label="original (sin filtrar)", alpha=0.8)
    ax[1].plot(f, Y, lw=0.7, label="filtrada")
    for fc in (args.fc1, args.fc2):
        ax[1].axvline(fc, color="k", ls="--", lw=0.8)
    ax[1].set(title="Espectro ||X(f)||: banda de paso y banda de rechazo", xlabel="Frecuencia (Hz)",
              ylabel="Magnitud", xlim=(0, 10000))
    ax[1].legend(loc="upper right")

    ax[2].plot(f, 20 * np.log10(X + 1e-9), lw=0.7, label="original (sin filtrar)", alpha=0.8)
    ax[2].plot(f, 20 * np.log10(Y + 1e-9), lw=0.7, label="filtrada")
    for fc in (args.fc1, args.fc2):
        ax[2].axvline(fc, color="k", ls="--", lw=0.8)
    ax[2].set(title="Espectro en dB", xlabel="Frecuencia (Hz)", ylabel="Magnitud (dB)",
              xlim=(0, 10000))
    ax[2].legend(loc="upper right")

    ax[3].plot(fh, H, color="tab:green")
    ax[3].axhline(1 / np.sqrt(2), color="gray", ls=":", lw=0.8, label="-3 dB (0.707)")
    for fc in (args.fc1, args.fc2):
        ax[3].axvline(fc, color="k", ls="--", lw=0.8)
    ax[3].set(title=f"Respuesta en frecuencia del filtro ||H(f)|| - {nombre}",
              xlabel="Frecuencia (Hz)", ylabel="Magnitud", xlim=(0, 10000))
    ax[3].legend(loc="upper right")
    fig.tight_layout()

    FIGURAS.mkdir(parents=True, exist_ok=True)
    salida = FIGURAS / "fase1_iir.png"
    fig.savefig(salida, dpi=130)
    print(f"Figura guardada en {salida}")

    if args.reproducir:
        print("Reproduciendo ORIGINAL..."); captura.reproducir(x)
        print("Reproduciendo FILTRADA..."); captura.reproducir(y)

    if not args.no_mostrar:
        plt.show()


if __name__ == "__main__":
    main()
