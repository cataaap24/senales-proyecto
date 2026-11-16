import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
BANCO = RAIZ / "datos" / "banco_aves"
CREDITOS = BANCO / "creditos.csv"


def escoger(filas, cantidad):
    """Toma `cantidad` filas por turnos entre grabadores."""
    grupos = defaultdict(list)
    for fila in filas:
        grupos[fila["grabador"]].append(fila)
    for lista in grupos.values():
        lista.sort(key=lambda f: (f["pais"] != "Colombia", -float(f["snr_db"] or 0), f["archivo"]))
    orden = sorted(grupos, key=lambda g: (-len(grupos[g]), g))
    elegidas = []
    ronda = 0
    while len(elegidas) < cantidad:
        for grabador in orden:
            if ronda < len(grupos[grabador]) and len(elegidas) < cantidad:
                elegidas.append(grupos[grabador][ronda])
        ronda += 1
    return elegidas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aplicar", action="store_true")
    args = parser.parse_args()

    with open(CREDITOS, encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        campos = lector.fieldnames
        filas = list(lector)

    mover = []
    for conjunto in ("entrenamiento", "prueba"):
        por_especie = {}
        for fila in filas:
            if fila["conjunto"] == conjunto:
                carpeta = Path(fila["archivo"]).parts[1]
                por_especie[carpeta] = por_especie.get(carpeta, []) + [fila]
        minimo = min(len(v) for v in por_especie.values())
        for carpeta, lista in por_especie.items():
            if len(lista) > minimo:
                quedan = {f["archivo"] for f in escoger(lista, minimo)}
                mover += [f for f in lista if f["archivo"] not in quedan]
        print(conjunto, {k: len(v) for k, v in por_especie.items()}, "->", minimo, "por especie")

    print(f"Pasan a reserva: {len(mover)} audios")
    if not args.aplicar:
        return
    for fila in mover:
        origen = BANCO / fila["archivo"]
        partes = Path(fila["archivo"]).parts  
        nuevo = Path("reserva") / partes[0] / partes[1] / partes[2]
        destino = BANCO / nuevo
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(origen), str(destino))
        fila["archivo"] = nuevo.as_posix()
        fila["conjunto"] = "reserva"
    with open(CREDITOS, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


if __name__ == "__main__":
    sys.exit(main())
