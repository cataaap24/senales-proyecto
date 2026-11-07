"""
curar_banco.py - Revisa audios nuevos y deja en una carpeta de CANDIDATOS solo los que sirven.

NO toca datos/banco_aves: los aceptados quedan en datos/candidatos/ para que los revises (oídos +
espectrograma) y luego los copies tú al banco. Así un audio malo nunca entra al entrenamiento sin que lo
veas.

Qué hace con cada archivo
  1. Lo lee, lo pasa a mono y a 22050 Hz.
  2. Lo filtra con el MISMO pasabandas del proyecto (200-8000 Hz) SOLO para medir. El archivo que se
     guarda NO va filtrado: el filtro se aplica en acondicionar(), tanto al entrenar como en vivo. Si se
     guardara ya filtrado, el entrenamiento filtraría dos veces y los vectores de energía no serían
     comparables con lo que entra por el micrófono.
  3. Busca la(s) ventana(s) de 5 s con más energía dentro de la banda del ave.
  4. Mide la calidad de cada ventana y decide: acepta o rechaza, con el motivo.
  5. Reparte por GRABACIÓN (no por recorte) entre entrenamiento y prueba: dos recortes de la misma
     grabación nunca quedan uno en cada conjunto, porque eso inflaría el resultado.

Métricas (reconstruidas a partir de LEEME.txt del banco; los números no coinciden exactos con los de
creditos.csv, que se calcularon con otro script, pero miden lo mismo):
  snr_db      = 10 log10( energía de cuadros de 50 ms en percentil 95 / en percentil 10 )
  actividad   = fracción de cuadros con energía > ruido (p10) + 10 dB
  frac_banda  = energía en la banda del ave / energía de 0 a 10 kHz
  clipping    = fracción de muestras saturadas en el original

Uso (desde la raíz del repo):
  python herramientas/curar_banco.py <carpeta_con_audios> --especie paloma --prueba 3
  python herramientas/curar_banco.py <carpeta_con_audios> --especie copeton
"""
import argparse
import csv
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly   # herramienta de preparación: no es el método evaluado

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from config import FS, N, DURACION                  # noqa: E402
from preprocesamiento import coeficientes_prefiltro  # noqa: E402

# banda del canto, y umbrales mínimos (de LEEME.txt; para la paloma el SNR no estaba fijado: se usa 12 dB)
ESPECIES = {
    "paloma":  dict(banda=(300, 1200),  frac_min=0.70, snr_min=12.0, recortes=4),
    "copeton": dict(banda=(2500, 7000), frac_min=0.85, snr_min=20.0, recortes=1),
}
ACTIVIDAD_MIN = 0.15
CLIPPING_MAX = 0.001
EXTENSIONES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def leer(ruta):
    """Audio mono a FS Hz (float64) y la fracción de muestras saturadas del original."""
    x, fs = sf.read(str(ruta), dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    sat = float(np.mean(np.abs(x) >= 0.999 * max(np.max(np.abs(x)), 1e-12))) if np.max(np.abs(x)) >= 0.999 else 0.0
    if fs != FS:
        r = Fraction(FS, int(fs)).limit_denominator(1000)
        x = resample_poly(x, r.numerator, r.denominator)
    return x, sat


def metricas(v, banda):
    """Calidad de una ventana YA filtrada con el pasabandas del proyecto."""
    n = int(0.05 * FS)
    m = len(v) // n
    E = (v[:m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-12
    ruido = np.percentile(E, 10)
    snr = 10 * np.log10(np.percentile(E, 95) / ruido)
    actividad = float(np.mean(E > ruido * 10))
    P = np.abs(np.fft.rfft(v)) ** 2
    f = np.fft.rfftfreq(len(v), 1 / FS)
    frac = P[(f >= banda[0]) & (f <= banda[1])].sum() / P[f <= 10000].sum()
    return snr, actividad, float(frac)


def ventanas(xf, banda, k, salto=0.5):
    """Inicios (en muestras) de hasta k ventanas de 5 s sin solape, de mayor a menor energía en banda."""
    pasos = np.arange(0, len(xf) - N + 1, int(salto * FS))
    if len(pasos) == 0:
        return []
    f = np.fft.rfftfreq(N, 1 / FS)
    sel = (f >= banda[0]) & (f <= banda[1])
    e = np.array([np.sum(np.abs(np.fft.rfft(xf[i:i + N])[sel]) ** 2) for i in pasos])
    elegidos = []
    for i in pasos[np.argsort(e)[::-1]]:
        if all(abs(i - j) >= N for j in elegidos):
            elegidos.append(i)
        if len(elegidos) == k:
            break
    return sorted(elegidos)


def main():
    ap = argparse.ArgumentParser(description="Revisa audios nuevos y separa los que sirven")
    ap.add_argument("carpeta", help="carpeta con los audios nuevos (wav, flac, mp3, ogg)")
    ap.add_argument("--especie", choices=list(ESPECIES), required=True)
    ap.add_argument("--prueba", type=int, default=0, help="cuántas GRABACIONES apartar para prueba")
    ap.add_argument("--salida", default=str(RAIZ / "datos" / "candidatos"))
    args = ap.parse_args()

    cfg = ESPECIES[args.especie]
    b, a = map(np.array, coeficientes_prefiltro())
    archivos = sorted(p for p in Path(args.carpeta).rglob("*") if p.suffix.lower() in EXTENSIONES)
    if not archivos:
        sys.exit(f"No encontré audios en {args.carpeta}")

    informe, aceptadas = [], []          # aceptadas: (grabación, ventana_id, señal, fila)
    for p in archivos:
        try:
            x, sat = leer(p)
        except Exception as e:           # archivo dañado o formato no soportado
            informe.append(dict(archivo=p.name, ventana="", decision="RECHAZADO", motivo=f"no se pudo leer: {e}"))
            continue
        if len(x) < N:
            informe.append(dict(archivo=p.name, ventana="", decision="RECHAZADO",
                                motivo=f"dura {len(x) / FS:.1f} s (< {DURACION:.0f} s)"))
            continue
        xf = lfilter(b, a, x)            # equivale a la ecuación en diferencias (verificado en tests)
        for j, i in enumerate(ventanas(xf, cfg["banda"], cfg["recortes"]), start=1):
            snr, act, frac = metricas(xf[i:i + N], cfg["banda"])
            motivos = []
            if frac < cfg["frac_min"]:
                motivos.append(f"poca energía en banda ({frac:.2f} < {cfg['frac_min']})")
            if snr < cfg["snr_min"]:
                motivos.append(f"SNR bajo ({snr:.1f} < {cfg['snr_min']} dB)")
            if act < ACTIVIDAD_MIN:
                motivos.append(f"casi sin canto ({act:.2f} < {ACTIVIDAD_MIN})")
            if sat > CLIPPING_MAX:
                motivos.append(f"saturado ({100 * sat:.2f} % de muestras)")
            fila = dict(archivo=p.name, ventana=j, inicio_s=round(i / FS, 2), snr_db=round(snr, 1),
                        actividad=round(act, 2), frac_banda=round(frac, 3),
                        decision="RECHAZADO" if motivos else "ACEPTADO", motivo="; ".join(motivos))
            informe.append(fila)
            if not motivos:
                aceptadas.append((p.stem, j, x[i:i + N], fila))

    # reparto por grabación: las últimas `prueba` grabaciones (orden fijo) van a prueba
    grabaciones = sorted({g for g, *_ in aceptadas})
    a_prueba = set(grabaciones[::-1][:args.prueba]) if args.prueba else set()
    salida = Path(args.salida)
    for g, j, v, fila in aceptadas:
        conjunto = "prueba" if g in a_prueba else "entrenamiento"
        d = salida / conjunto / args.especie
        d.mkdir(parents=True, exist_ok=True)
        v = 0.9 * v / (np.max(np.abs(v)) + 1e-12)        # mismo criterio del banco: pico 0.9, SIN filtrar
        sf.write(str(d / f"{args.especie}_{g}_r{j}.wav"), v, FS, subtype="PCM_16")
        fila["conjunto"] = conjunto

    salida.mkdir(parents=True, exist_ok=True)
    ruta_csv = salida / f"informe_{args.especie}.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["archivo", "ventana", "inicio_s", "snr_db", "actividad",
                                           "frac_banda", "decision", "conjunto", "motivo"], restval="")
        w.writeheader()
        w.writerows(informe)

    ok = sum(r["decision"] == "ACEPTADO" for r in informe)
    print(f"{len(archivos)} archivos -> {len(informe)} ventanas revisadas: {ok} aceptadas, "
          f"{len(informe) - ok} rechazadas")
    print(f"Grabaciones con al menos un recorte aceptado: {len(grabaciones)}  (a prueba: {len(a_prueba)})")
    print(f"Informe: {ruta_csv}\nCandidatos: {salida}")


if __name__ == "__main__":
    main()
