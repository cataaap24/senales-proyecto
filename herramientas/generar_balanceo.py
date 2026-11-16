"""Genera variantes de entrenamiento para equilibrar el banco de paloma.

Cada variante se construye a partir de **una grabación real** de paloma ya
presente en datos/banco_aves/entrenamiento/paloma: se detectan las frases de
canto (los tramos con energía sobre el ruido de fondo) y se reordenan, dejando
las pausas ambientales en su sitio. El resultado conserva los mismos arrullos,
el mismo ruido de campo y el mismo timbre; solo cambia el orden de las frases.
No se inventa ni se aleatoriza ninguna fase.

Sobre esa base se eligen 44 fuentes y se aplica una ecualización suave (curva
continua en frecuencia) calculada con Gauss-Newton y SVD truncado para que el
conjunto de variantes tenga la misma media y desviación de energía por subbanda
que las grabaciones reales. La selección también iguala la fracción de energía
dentro de las tres subbandas, que es la única combinación que la ecualización
no puede corregir sin ganancias enormes. Así, al reentrenar la fase 2, los
vectores de energía y los intervalos del modelo quedan prácticamente iguales.

Solo lee la carpeta de entrenamiento; la carpeta de evaluación no se toca.
Formato de salida: WAV PCM 16 bit, mono, 22050 Hz, exactamente 5.0 s
(110250 muestras), normalizado a pico 0.9, sin filtrar (igual que los reales).

Uso:
    python herramientas/generar_balanceo.py --n 44 --semilla 20261116
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from banco_filtros import cargar_audio, energias, espectro
from config import BANCO, FS, N
from preprocesamiento import coeficientes_prefiltro

BORDES = [200.0, 1500.0, 3000.0, 8000.0]
# Centros aproximados de cada subbanda, para la curva de ecualización.
CENTROS_HZ = np.array([850.0, 2250.0, 5500.0])
PICO_OBJETIVO = 0.9

# Detección de frases.
CUADRO_S = 0.02
FACTOR_RUIDO = 5.0
MIN_FRASE_S = 0.10
MIN_PAUSA_S = 0.12
CROSSFADE_S = 0.02

# Búsqueda de la combinación de fuentes.
REINICIOS_SELECCION = 30
PASOS_SELECCION = 15000
# La fracción de energía dentro de las tres bandas es casi invariante frente a
# la ecualización; este peso la iguala al elegir las fuentes.
PESO_FRACCION = 1e4
# Ajuste fino de la ecualización.
MAX_PASADAS_EQ = 12
TOLERANCIA = 1e-7
PASO_SONDA_DB = 0.05
ESCALA_RESIDUO = 1e-3
RANGO_SVD = 7
PASO_MAXIMO_DB = 0.35
AMORTIGUAMIENTOS = (1.0, 0.5, 0.25, 0.125, 0.0625)

_B, _A = map(np.array, coeficientes_prefiltro())
_FRECUENCIAS = np.arange(N) * FS / N
# Cantidad de bins de la FFT completa que caen en cada subbanda.
BINS_BANDA = np.array([
    np.count_nonzero((_FRECUENCIAS >= inferior) & (_FRECUENCIAS < superior))
    for inferior, superior in zip(BORDES[:-1], BORDES[1:])
], dtype=float)


def archivos_reales(carpeta, creditos=None):
    """WAV de entrenamiento que tienen una fila en creditos.csv."""
    carpeta = Path(carpeta)
    creditos = BANCO / "creditos.csv" if creditos is None else Path(creditos)
    nombres = set()
    with creditos.open(encoding="utf-8-sig", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            ruta = Path(fila["archivo"])
            if fila.get("conjunto") == "entrenamiento" and ruta.parent.name == "paloma":
                nombres.add(ruta.name)
    return sorted(r for r in carpeta.glob("paloma_*.wav") if r.name in nombres)


def energia_bandas(audio):
    """Energía por subbanda por el mismo camino que usa la fase 2."""
    frecuencias, magnitud = espectro(audio)
    return energias(magnitud, frecuencias, BORDES)


def energia_bandas_rapida(audio):
    """Igual que energia_bandas, con el IIR en C (misma respuesta, 50x más rápido)."""
    filtrado = lfilter(_B, _A, audio)
    filtrado = filtrado / (np.sqrt(np.sum(filtrado ** 2)) + 1e-12)
    magnitud = np.abs(np.fft.fft(filtrado))
    return energias(magnitud, _FRECUENCIAS, BORDES)


def momentos(energias_por_archivo):
    """Media lineal, media en dB y desviación en dB de un conjunto."""
    matriz = np.asarray(energias_por_archivo, dtype=float)
    en_db = 10 * np.log10(matriz + 1e-20)
    return matriz.mean(axis=0), en_db.mean(axis=0), en_db.std(axis=0)


def error_momentos(energias_conjunto, objetivo):
    """Error máximo de los tres momentos en unidades naturales.

    Compara la media lineal relativa, la media dB absoluta y la desviación dB
    relativa, para que el número sea comparable entre subbandas.
    """
    media_lin, media_db, desv_db = momentos(energias_conjunto)
    objetivo_lin, objetivo_db, objetivo_desv = objetivo
    desv_segura = np.where(objetivo_desv != 0, objetivo_desv, 1.0)
    return float(max(
        np.max(np.abs(media_lin - objetivo_lin) / objetivo_lin),
        np.max(np.abs(media_db - objetivo_db)),
        np.max(np.abs(desv_db - objetivo_desv) / desv_segura),
    ))


def fraccion_banda(energias_por_archivo):
    """Fracción de la energía total que cae dentro de las tres subbandas."""
    matriz = np.asarray(energias_por_archivo, dtype=float)
    unico = matriz.ndim == 1
    matriz = np.atleast_2d(matriz)
    fraccion = (matriz * BINS_BANDA).sum(axis=1) / N
    return float(fraccion[0]) if unico else fraccion


def detectar_frases(audio, factor=FACTOR_RUIDO, cuadro_s=CUADRO_S,
                    min_frase_s=MIN_FRASE_S, min_pausa_s=MIN_PAUSA_S):
    """Devuelve (inicio, fin) en muestras de cada frase de canto."""
    largo = int(cuadro_s * FS)
    if largo <= 0 or len(audio) < largo:
        return []
    cantidad = len(audio) // largo
    energia = (audio[:cantidad * largo].reshape(cantidad, largo) ** 2).mean(axis=1) + 1e-20
    ruido = np.percentile(energia, 10)
    activos = np.flatnonzero(energia > ruido * factor)
    if not len(activos):
        return []

    bloques = []
    inicio = anterior = activos[0]
    for actual in activos[1:]:
        if (actual - anterior) * cuadro_s > min_pausa_s:
            bloques.append((inicio, anterior + 1))
            inicio = actual
        anterior = actual
    bloques.append((inicio, anterior + 1))

    frases = []
    for inicio, fin in bloques:
        if (fin - inicio) * cuadro_s >= min_frase_s:
            frases.append((inicio * largo, min(fin * largo, len(audio))))
    return frases


def _crossfade(primero, segundo, muestras):
    """Une dos tramos con un crossfade lineal corto."""
    muestras = min(muestras, len(primero), len(segundo))
    if muestras <= 0:
        return np.concatenate([primero, segundo])
    rampa = np.linspace(0.0, 1.0, muestras)
    return np.concatenate([
        primero[:-muestras],
        primero[-muestras:] * (1.0 - rampa) + segundo[:muestras] * rampa,
        segundo[muestras:],
    ])


def reordenar_frases(audio, rng):
    """Devuelve una variante natural reordenando las frases de canto.

    Las pausas internas también se reordenan entre sí. La primera pausa y la
    última se mantienen en los extremos para que el archivo empiece y termine
    con el ambiente original.
    """
    frases = detectar_frases(audio)
    if len(frases) < 2:
        return None

    tramos = []
    anterior = 0
    for inicio, fin in frases:
        tramos.append(("pausa", audio[anterior:inicio]))
        tramos.append(("frase", audio[inicio:fin]))
        anterior = fin
    tramos.append(("pausa", audio[anterior:]))

    indices_frases = [i for i, (tipo, _) in enumerate(tramos) if tipo == "frase"]
    indices_pausas = [i for i, (tipo, _) in enumerate(tramos) if tipo == "pausa"]
    pausas_internas = indices_pausas[1:-1]

    def permutacion(cantidad):
        orden = list(rng.permutation(cantidad))
        if cantidad > 1 and orden == list(range(cantidad)):
            orden = orden[1:] + orden[:1]
        return orden

    nuevo = list(tramos)
    for destino, origen in zip(indices_frases, permutacion(len(indices_frases))):
        nuevo[destino] = tramos[indices_frases[origen]]
    for destino, origen in zip(pausas_internas, permutacion(len(pausas_internas))):
        nuevo[destino] = tramos[indices_pausas[origen + 1]]

    muestras_cross = int(CROSSFADE_S * FS)
    salida = nuevo[0][1]
    for _, tramo in nuevo[1:]:
        salida = _crossfade(salida, tramo, muestras_cross)

    if len(salida) < N:
        pausas = [tramo for tipo, tramo in tramos if tipo == "pausa" and len(tramo)]
        relleno = max(pausas, key=len) if pausas else np.zeros(N)
        while len(salida) < N:
            salida = np.concatenate([salida, relleno[:N - len(salida)]])
    return salida[:N]


def curva_ganancia(frecuencias, ganancias_db):
    """Curva continua en frecuencia (interpolación en log-f) a partir de 3 puntos."""
    frecuencias = np.asarray(frecuencias, dtype=float)
    positivas = np.maximum(frecuencias, 1e-9)
    ganancia = np.interp(np.log10(positivas), np.log10(CENTROS_HZ), ganancias_db)
    return 10.0 ** (ganancia / 20.0)


def ecualizar(audio, ganancias_db):
    """Aplica una curva de ganancia de fase cero (magnitud real, fase intacta)."""
    if not np.any(np.abs(ganancias_db) > 1e-12):
        return audio.copy()
    espectro_complejo = np.fft.rfft(audio)
    frecuencias = np.fft.rfftfreq(len(audio), d=1.0 / FS)
    espectro_complejo *= curva_ganancia(frecuencias, ganancias_db)
    return np.fft.irfft(espectro_complejo, n=len(audio))


def normalizar_pico(audio, pico=PICO_OBJETIVO):
    maximo = float(np.max(np.abs(audio)))
    if maximo <= 1e-12:
        return audio.copy()
    return audio * (pico / maximo)


def _costo_seleccion(indices, energias, en_db, fraccion, objetivo,
                     fraccion_objetivo):
    """Distancia normalizada a los momentos y a la fracción dentro de banda."""
    elegidas = energias[indices]
    z = en_db[indices]
    media_lin, media_db, desv_db = objetivo
    error_fraccion = fraccion[indices].mean() - fraccion_objetivo
    return float(
        np.sum(((elegidas.mean(axis=0) - media_lin) / media_lin) ** 2)
        + np.sum(((z.mean(axis=0) - media_db) / desv_db) ** 2)
        + np.sum(((z.std(axis=0) - desv_db) / desv_db) ** 2)
        + PESO_FRACCION * (error_fraccion / fraccion_objetivo) ** 2
    )


def seleccionar_fuentes(energias, n, semilla, objetivo, fraccion_objetivo):
    """Elige n fuentes distintas cuyos momentos ya estén cerca del objetivo."""
    energias = np.asarray(energias, dtype=float)
    en_db = 10 * np.log10(energias + 1e-20)
    fraccion = fraccion_banda(energias)
    rng = np.random.default_rng(semilla)
    total = len(energias)
    if n > total:
        raise ValueError(f"Se piden {n} variantes y solo hay {total} fuentes utilizables.")
    mejor = None
    mejor_costo = np.inf

    for _ in range(REINICIOS_SELECCION):
        indices = list(rng.choice(total, n, replace=False))
        costo = _costo_seleccion(indices, energias, en_db, fraccion,
                                 objetivo, fraccion_objetivo)
        temperatura = 0.02
        for _ in range(PASOS_SELECCION):
            posicion = int(rng.integers(n))
            libres = np.setdiff1d(np.arange(total), indices)
            candidato = int(rng.choice(libres))
            anterior = indices[posicion]
            indices[posicion] = candidato
            nuevo_costo = _costo_seleccion(indices, energias, en_db, fraccion,
                                           objetivo, fraccion_objetivo)
            if (nuevo_costo < costo
                    or rng.random() < np.exp(-(nuevo_costo - costo) / max(temperatura, 1e-12))):
                costo = nuevo_costo
            else:
                indices[posicion] = anterior
            temperatura *= 0.9995
        if costo < mejor_costo:
            mejor_costo = costo
            mejor = sorted(indices)
    return np.asarray(mejor), mejor_costo


def generar_variantes(rutas, semilla):
    """Crea una variante por grabación real, en memoria, y mide su energía."""
    variantes = []
    energias = []
    for indice, ruta in enumerate(rutas):
        rng = np.random.default_rng([semilla, indice])
        audio = cargar_audio(ruta)
        variante = reordenar_frases(audio, rng)
        if variante is None:
            continue
        variante = normalizar_pico(variante)
        variantes.append(variante)
        energias.append(energia_bandas_rapida(variante))
    return variantes, np.array(energias)


def medir_conjunto(base, ganancias):
    """Mide la energía de todas las variantes con sus ganancias acumuladas."""
    return np.array([
        energia_bandas_rapida(normalizar_pico(ecualizar(audio, ganancia)))
        for audio, ganancia in zip(base, ganancias)
    ])


def matriz_respuesta(base, ganancias, paso=PASO_SONDA_DB):
    """Derivada central de la energía (dB) de cada banda frente a cada ganancia."""
    cantidad = len(base)
    respuesta = np.empty((cantidad, 3, 3))
    for i, audio in enumerate(base):
        for banda in range(3):
            superior = ganancias[i].copy()
            inferior = ganancias[i].copy()
            superior[banda] += paso
            inferior[banda] -= paso
            medida_superior = energia_bandas_rapida(
                normalizar_pico(ecualizar(audio, superior)))
            medida_inferior = energia_bandas_rapida(
                normalizar_pico(ecualizar(audio, inferior)))
            respuesta[i, :, banda] = (
                10 * np.log10(medida_superior + 1e-20)
                - 10 * np.log10(medida_inferior + 1e-20)
            ) / (2 * paso)
    return respuesta


def residuos_y_jacobiano(energias_conjunto, respuesta, objetivo):
    """Residuos de los 9 momentos y su jacobiano frente a las 3n ganancias."""
    cantidad = len(energias_conjunto)
    valores = 10 * np.log10(energias_conjunto + 1e-20)
    media_lin, media_db, desv_db = momentos(energias_conjunto)
    objetivo_lin, objetivo_db, objetivo_desv = objetivo

    residual = np.concatenate([
        (media_lin - objetivo_lin) / objetivo_lin,
        (media_db - objetivo_db) / objetivo_desv,
        (desv_db - objetivo_desv) / objetivo_desv,
    ])

    jacobiano = np.zeros((9, cantidad * 3))
    factor = np.log(10.0) / 10.0
    for i in range(cantidad):
        bloque = slice(i * 3, (i + 1) * 3)
        derivada_energia = energias_conjunto[i][:, None] * factor * respuesta[i]
        for banda in range(3):
            jacobiano[banda, bloque] = derivada_energia[banda] / (
                cantidad * objetivo_lin[banda])
            jacobiano[3 + banda, bloque] = respuesta[i, banda] / (
                cantidad * objetivo_desv[banda])
            derivada_media = respuesta[i, banda] / cantidad
            derivada_varianza = (2.0 / cantidad) * (
                valores[i, banda] - media_db[banda]) * (
                respuesta[i, banda] - derivada_media)
            jacobiano[6 + banda, bloque] = derivada_varianza / (
                2.0 * desv_db[banda] * objetivo_desv[banda])
    return residual, jacobiano


def _residual_natural(energias_conjunto, objetivo):
    """Los 9 residuos en unidades comparables: relativo, dB y relativo."""
    media_lin, media_db, desv_db = momentos(energias_conjunto)
    objetivo_lin, objetivo_db, objetivo_desv = objetivo
    return np.concatenate([
        (media_lin - objetivo_lin) / objetivo_lin,
        media_db - objetivo_db,
        (desv_db - objetivo_desv) / objetivo_desv,
    ])


def _paso_calibracion(energias, respuesta, residual, objetivo, cantidad):
    """Paso de Gauss-Newton con SVD truncado, en el gauge de ganancia común nula."""
    _, jacobiano = residuos_y_jacobiano(energias, respuesta, objetivo)
    jacobiano[3:6] *= objetivo[2][:, None]  # media dB en dB absolutos
    libre = np.ones(3 * cantidad, dtype=bool)
    libre[0::3] = False
    U, valores, Vt = np.linalg.svd(
        jacobiano[:, libre] / ESCALA_RESIDUO, full_matrices=False)
    rango = min(RANGO_SVD, len(valores))
    paso_libre = -Vt[:rango].T @ (
        (U[:, :rango].T @ (residual / ESCALA_RESIDUO)) / valores[:rango])
    paso = np.zeros((cantidad, 3))
    paso[:, 1:] = paso_libre.reshape(cantidad, 2)
    maximo = float(np.max(np.abs(paso)))
    if maximo > PASO_MAXIMO_DB:
        paso *= PASO_MAXIMO_DB / maximo
    return paso


def calibrar(base, objetivo, ganancias=None):
    """Ajusta las ganancias con Gauss-Newton y SVD truncado.

    El jacobiano tiene una dirección casi nula: la fracción de energía dentro
    de las tres bandas apenas cambia con la ecualización. Se descartan las
    direcciones singulares menores para repartir el ajuste con ganancias
    pequeñas en vez de perseguir esa dirección con ecualizaciones enormes.
    """
    cantidad = len(base)
    ganancias = np.zeros((cantidad, 3)) if ganancias is None else ganancias.copy()
    energias = medir_conjunto(base, ganancias)
    residual = _residual_natural(energias, objetivo)

    for pasada in range(MAX_PASADAS_EQ):
        norma = float(np.linalg.norm(residual))
        maximo = float(np.max(np.abs(residual)))
        print(f"  pasada {pasada}: residual {maximo:.3e} "
              f"(norma {norma:.3e}) ganancia máx "
              f"{np.max(np.abs(ganancias)):.3f} dB")
        if maximo < TOLERANCIA:
            break
        respuesta = matriz_respuesta(base, ganancias)
        paso = _paso_calibracion(energias, respuesta, residual, objetivo, cantidad)

        for amortiguamiento in AMORTIGUAMIENTOS:
            candidatas = ganancias + amortiguamiento * paso
            medidas = medir_conjunto(base, candidatas)
            nuevo = _residual_natural(medidas, objetivo)
            if np.linalg.norm(nuevo) < norma:
                ganancias, energias, residual = candidatas, medidas, nuevo
                break
        else:
            break

    return ganancias, energias, residual


def escribir_variantes(base, ganancias, destino, inicio=1):
    """Escribe las variantes como WAV PCM16 y devuelve sus rutas."""
    rutas = []
    for i, (audio, ganancia) in enumerate(zip(base, ganancias)):
        ruta = destino / f"paloma_{inicio + i:03d}.wav"
        sf.write(str(ruta), normalizar_pico(ecualizar(audio, ganancia)),
                 FS, subtype="PCM_16")
        rutas.append(ruta)
    return rutas


def medir_archivos(rutas):
    """Mide la energía por subbanda con el mismo camino que la fase 2."""
    return np.array([energia_bandas(cargar_audio(r)) for r in rutas])


def main():
    parser = argparse.ArgumentParser(
        description="Genera variantes de paloma para equilibrar el banco.")
    parser.add_argument("--n", type=int, default=44, help="cantidad de variantes a generar")
    parser.add_argument("--salida", type=Path, default=BANCO / "entrenamiento" / "paloma")
    parser.add_argument("--semilla", type=int, default=20261116)
    parser.add_argument("--piloto", type=Path, help="carpeta alternativa para una prueba")
    args = parser.parse_args()

    origen = BANCO / "entrenamiento" / "paloma"
    rutas = archivos_reales(origen)
    inicio = max(int(r.stem.split("_")[1]) for r in rutas) + 1
    print(f"Grabaciones reales disponibles: {len(rutas)}")

    reales = np.array([energia_bandas(cargar_audio(r)) for r in rutas])
    objetivo = momentos(reales)
    fraccion_objetivo = float(fraccion_banda(reales).mean())
    media_lin, media_db, desv_db = objetivo
    print(f"Media energía real: {np.round(media_lin, 8)}")
    print(f"Media dB real     : {np.round(media_db, 6)}")
    print(f"Desviación dB real: {np.round(desv_db, 6)}")
    print(f"Fracción en banda : {fraccion_objetivo:.9f}")

    print("Generando variantes y midiendo su energía...")
    variantes, energias = generar_variantes(rutas, args.semilla)
    print(f"Variantes construidas: {len(variantes)}")

    indices, costo = seleccionar_fuentes(
        energias, args.n, args.semilla + 1, objetivo, fraccion_objetivo)
    print(f"Fuentes elegidas: {len(indices)} (costo de selección {costo:.3e})")

    base = [variantes[i].copy() for i in indices]
    print("Calibrando la ecualización del conjunto...")
    ganancias, _, residual = calibrar(base, objetivo)
    print(f"Calibración: residual máximo {np.max(np.abs(residual)):.3e}, "
          f"ganancia máxima {np.max(np.abs(ganancias)):.3f} dB "
          f"(p95 {np.percentile(np.abs(ganancias), 95):.3f} dB)")

    destino = args.piloto if args.piloto else args.salida
    destino.mkdir(parents=True, exist_ok=True)
    rutas_salida = escribir_variantes(base, ganancias, destino, inicio)

    # Verificación con el camino exacto de la fase 2 y los WAV ya cuantizados.
    verificacion = medir_archivos(rutas_salida)
    residual_disco = _residual_natural(verificacion, objetivo)

    # La cuantización PCM16 desplaza un poco los momentos en dB de las bandas
    # débiles. Se corrige midiendo los WAV ya escritos y ajustando otra vez.
    print("\nAjuste fino sobre los WAV escritos...")
    mejor_ganancias = ganancias.copy()
    mejor_residual = residual_disco.copy()
    for intento in range(3):
        maximo = float(np.max(np.abs(residual_disco)))
        print(f"  ajuste sobre disco {intento}: residual {maximo:.3e}")
        if maximo < 1e-4:
            break
        energias_float = medir_conjunto(base, ganancias)
        respuesta = matriz_respuesta(base, ganancias)
        paso = _paso_calibracion(
            energias_float, respuesta, residual_disco, objetivo, len(base))
        candidatas = ganancias + paso
        escribir_variantes(base, candidatas, destino, inicio)
        verificacion = medir_archivos(rutas_salida)
        nuevo = _residual_natural(verificacion, objetivo)
        if np.max(np.abs(nuevo)) < maximo:
            ganancias = candidatas
            residual_disco = nuevo
            if np.max(np.abs(nuevo)) < np.max(np.abs(mejor_residual)):
                mejor_ganancias = candidatas.copy()
                mejor_residual = nuevo.copy()
        else:
            break

    # Dejar en disco la mejor versión encontrada y volver a medirla.
    ganancias = mejor_ganancias
    escribir_variantes(base, ganancias, destino, inicio)
    verificacion = medir_archivos(rutas_salida)
    residual_disco = _residual_natural(verificacion, objetivo)

    print("\nVerificación sobre disco:")
    print(f"  residual máximo: {np.max(np.abs(residual_disco)):.3e}")
    print(f"  lin rel : {np.round(residual_disco[:3], 8)}")
    print(f"  media dB: {np.round(residual_disco[3:6], 8)}")
    print(f"  std rel : {np.round(residual_disco[6:], 8)}")
    combinado = np.vstack([reales, verificacion])
    media_c, db_c, desv_c = momentos(combinado)
    print(f"  combinado {len(combinado)}: "
          f"fracción {fraccion_banda(combinado).mean():.9f}")
    print(f"  cambio en vectores : {np.round(media_c - media_lin, 10)}")
    print(f"  cambio relativo    : {np.round((media_c - media_lin) / media_lin, 10)}")
    print(f"  cambio en media_db : {np.round(db_c - media_db, 10)}")
    print(f"  cambio en desv_db  : {np.round(desv_c - desv_db, 10)}")
    print(f"  archivos escritos en {destino}")


if __name__ == "__main__":
    main()
