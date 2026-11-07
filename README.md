# Reconocimiento de aves por banco de filtros

Reconoce en tiempo real, con el micrófono, dos especies por su canto: paloma torcaza (*Zenaida auriculata*) y copetón (*Zonotrichia capensis*).

## Algoritmo (tal cual las notas del profesor)

1. Grabaciones de duración fija (5 s). Si hay ruido, se filtran.
2. FFT de cada grabación y magnitud ‖X(k)‖.
3. Espectro promedio por especie ‖C(k)‖ y gráfica de ambas especies en un mismo plano.
4. Se eligen 3 a 8 subbandas donde la energía de las especies sea diferente.
5. Energía por subbanda: E = (1/L) · Σ ‖C_i(k)‖². Estos vectores de umbral son constantes y se guardan.
6. En tiempo real: ‖X(k)‖ de la señal nueva, partido en las mismas subbandas, da E_X.
7. Para cada especie se calcula Σ |E_Xi − E_Ci|. La especie con el valor mínimo es la que se reconoce.

## Estructura

```
src/config.py            parámetros: fs = 22050 Hz, 5 s, pre-filtro 200-8000 Hz, rutas
src/filtros.py           filtros hechos a mano, según las notsa
src/espectro.py          FFT y magnitud
src/captura.py           micrófono
con el mismo formato
src/preprocesamiento.py  acondicionar(): filtro + normalización
app/fase1_captura_filtro.py   demo de la Fase 1
tests/test_contra_pdf.py      pruebas contra los ejercicios del profesor y scipy
datos/banco_aves/        banco de audios (xeno-canto) con créditos
```

## Banco de audios

- Fuente: xeno-canto, con grabaciones de varios autores y países. Los créditos están en `datos/banco_aves/creditos.csv`.
- Formato: WAV mono, 22050 Hz, 5 s.
- Entrenamiento: 66 + 66. Prueba: 12 + 12 (paloma + copetón).

## Decisiones importantes

**Un solo acondicionamiento para todo.** El entrenamiento y el reconocimiento en vivo deben llamar a la misma función, `acondicionar()`. Si el banco se procesa de una forma y el micrófono de otra, los vectores de energía dejan de ser comparables y el reconocimiento falla aunque el algoritmo sea correcto. Si se cambia el filtro, hay que volver a entrenar.

**Entrenar una sola vez.** El banco se procesa una vez, y el resultado (cortes de subbandas, vectores de energía y parámetros del filtro) se guarda en `modelo/vectores_energia.json`. En el reconocimiento solo se carga ese archivo y se procesa la señal nueva. Solo se reentrena si cambia el banco, el filtro o el número de subbandas.

**Filtros escritos a mano, no con scipy.** 
- Butterworth pasabandas de orden 2 por transformación bilineal con prewarping, aplicado con la ecuación en diferencias. Es el pre-filtro de 200 a 8000 Hz: quita lo que no es canto (viento, golpes al micrófono, ruido muy agudo).
- FIR pasabajas con ventana Hamming. Solo se usa en `captura.diezmar()`, cuando el micrófono no abre a 22050 Hz: se graba a 44100 Hz y se reduce a la mitad, filtrando antes para evitar aliasing.

**Normalización de energía: es un agregado nuestro.** No está en las notas. Se hace porque el algoritmo compara energías absolutas contra umbrales fijos, y con el micrófono la energía sube o baja según el volumen y la distancia al ave. Se normaliza por energía total (no por el valor pico), porque un solo golpe al micrófono cambia el pico de toda la grabación pero casi no cambia su energía total. Debe mencionarse en el informe como parte del acondicionamiento, no del método principal.

**FFT.** Se usa `np.fft.rfft` (solo la mitad no negativa del espectro, porque la señal es real).

## Cómo correr

```
pip install -r requirements.txt
python tests/test_contra_pdf.py                     # comprueba los filtros
python app/fase1_captura_filtro.py                  # graba 5 s del micrófono, filtra y grafica
python app/fase1_captura_filtro.py --reproducir     # además suena el original y el filtrado
python app/fase1_captura_filtro.py --wav datos/banco_aves/prueba/paloma/<archivo>.wav
```

## Estado y fases

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Captura por micrófono + filtro | Hecha (micrófono sin probar en vivo) |
| 2 | Entrenamiento: FFT, espectro promedio, cortes de subbandas, energías, JSON | Pendiente |
| 3 | Reconocimiento y evaluación del error (meta: ≤ 5 %) | Pendiente |
| 4 | Interfaz, micrófono en tiempo real, umbral de actividad ("no se detectó canto") | Pendiente |
| 5 | Informe IEEE (12 a 15 páginas) | Pendiente |

## Limitaciones conocidas

- Con 24 señales de prueba, un solo error ya es 4.2 % y dos son 8.3 %. Para sustentar el 5 % hace falta un conjunto de prueba más grande.
- La prueba real es el micrófono: las grabaciones de xeno-canto y el micrófono del computador suenan distinto.
- Una prueba rápida con el banco, con el filtro de la clase y la normalización, clasificó 24/24 con 3 a 8 subbandas. Falta repetirla con el entrenamiento y el reconocimiento finales.
- Falta el umbral de actividad: sin él, el silencio se clasificaría como alguna de las dos especies.