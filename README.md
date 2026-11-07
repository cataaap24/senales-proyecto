# Reconocimiento de aves por banco de filtros

Reconoce tras una captura de cinco segundos, con el micrófono o un archivo, dos especies por su canto: paloma torcaza (*Zenaida auriculata*) y copetón (*Zonotrichia capensis*).

La interfaz no entrena. La fase 2 genera explícitamente un modelo JSON; la fase 3 lo carga para reconocer y evaluar; la fase 4 solo presenta los resultados.

## Cómo funciona

1. Grabaciones de duración fija (5 s). Si hay ruido, se filtran.
2. FFT de cada grabación y magnitud ‖X(k)‖.
3. Espectro promedio por especie ‖C(k)‖ y gráfica de ambas especies en un mismo plano.
4. Se eligen 3 a 8 subbandas donde la energía de las especies sea diferente.
5. Energía por subbanda: E = (1/L) · Σ ‖C_i(k)‖². Estos vectores de referencia se guardan en el modelo.
6. En reconocimiento: ‖X(k)‖ de la señal nueva, partido en las mismas subbandas, da E_X.
7. Para cada especie se calcula Σ |E_Xi − E_Ci|. La especie con el valor mínimo es la que se reconoce.

Esta es la base de trabajo descrita en el repositorio. El PDF del curso exige las dos especies, 3 a 8 subbandas, reconocimiento por micrófono y error máximo del 5 %, pero remite las ecuaciones detalladas del banco de filtros a lo explicado en clase. Antes de declarar oficial la fase matemática hay que cotejarla con esas notas. La normalización y el recorte por máxima energía son decisiones adicionales del proyecto.

## Estructura del proyecto

```
src/config.py                    parámetros y rutas
src/filtros.py                   filtros digitales de la fase 1
src/espectro.py                  FFT y magnitud
src/captura.py                   captura por micrófono
src/preprocesamiento.py          filtro y normalización compartidos
src/banco_filtros.py             operaciones comunes a las fases 2 y 3
app/fase1_captura_filtro.py      captura y filtrado
app/fase2_entrenamiento.py       entrenamiento explícito
app/fase3_reconocimiento.py      reconocimiento y evaluación, sin entrenar
app/fase4_interfaz.py            presentación de resultados
tests/                           pruebas automáticas
datos/banco_aves/                audios de entrenamiento, prueba y créditos
```

## Banco de audios

- Fuente: xeno-canto, con grabaciones de varios autores y países. Los créditos están en `datos/banco_aves/creditos.csv`.
- Formato: WAV mono, 22050 Hz, 5 s.
- Entrenamiento: 66 + 66. Prueba: 12 + 12 (paloma + copetón).

## Decisiones importantes

**Un solo acondicionamiento para todo.** El entrenamiento y el reconocimiento en vivo deben llamar a la misma función, `acondicionar()`. Si el banco se procesa de una forma y el micrófono de otra, los vectores de energía dejan de ser comparables y el reconocimiento falla aunque el algoritmo sea correcto. Si se cambia el filtro, hay que volver a entrenar.

**Entrenar únicamente bajo petición.** `app/fase2_entrenamiento.py` procesa el banco y guarda los bordes, los vectores de energía y los parámetros en `modelo/vectores_energia.json`. La fase 3 lee ese archivo sin modificarlo; la interfaz tampoco entrena. El modelo provisional anterior permanece en esa ruta y también se puede leer. Si se agregan audios o cambian las bandas, se ejecuta de nuevo la fase 2 para reemplazarlo.

**Filtros escritos a mano, no con scipy.** 
- Butterworth pasabandas de orden 2 por transformación bilineal con prewarping, aplicado con la ecuación en diferencias. Es el pre-filtro de 200 a 8000 Hz: quita lo que no es canto (viento, golpes al micrófono, ruido muy agudo).
- FIR pasabajas con ventana Hamming. Solo se usa en `captura.diezmar()`, cuando el micrófono no abre a 22050 Hz: se graba a 44100 Hz y se reduce a la mitad, filtrando antes para evitar aliasing.

**Normalización de energía: es un agregado nuestro.** No aparece en el enunciado del proyecto. Se hace porque el algoritmo compara energías absolutas contra vectores de referencia, y con el micrófono la energía sube o baja según el volumen y la distancia al ave. Se normaliza por energía total (no por el valor pico), porque un solo golpe al micrófono cambia el pico de toda la grabación pero casi no cambia su energía total. Debe mencionarse en el informe como parte del acondicionamiento, no del método principal.

**FFT.** Se usa `np.fft.rfft` (solo la mitad no negativa del espectro, porque la señal es real).

## Preparación y uso

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/fase1_captura_filtro.py --wav datos/banco_aves/prueba/paloma/paloma_008_XC272805_r2.wav --no-mostrar
```

En Windows, activa el entorno con `.venv\Scripts\activate`. Para el micrófono se requiere un dispositivo de entrada compatible con PortAudio. La fase 1 también admite `--reproducir` y captura directamente si se omite `--wav`.

## Ejemplo de uso de las fases 2 a 4

Los bordes iniciales son una base ajustable, no una selección de bandas ya validada. La opción `--grafica` ayuda a inspeccionar los espectros promedio antes de fijarlos.

```bash
python app/fase2_entrenamiento.py --bordes 0,2000,4000,6000,8000,10000 --grafica resultados/figuras/espectros_promedio.png
python app/fase3_reconocimiento.py --audio datos/banco_aves/prueba/paloma/paloma_008_XC272805_r2.wav
python app/fase3_reconocimiento.py --evaluar datos/banco_aves
python app/fase4_interfaz.py
python -m unittest discover -s tests -p 'test_*.py'
```

La fase 2 es la única que crea el modelo. La evaluación imprime aciertos y porcentaje de error. La interfaz vuelve a leer el modelo al analizar un audio: los nuevos vectores, nombres y bordes se muestran sin cambiar el código de la pantalla. Si falta un modelo válido, la interfaz lo informa y no intenta entrenarlo.

## Estado y fases

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Captura por micrófono + filtro | Hecha |
| 2 | Entrenamiento: FFT, espectro promedio, subbandas, energías, JSON | Base implementada; falta validar las bandas y la fidelidad matemática |
| 3 | Reconocimiento y evaluación del error (meta: ≤ 5 %) | Base implementada; faltan más pruebas |
| 4 | Interfaz, micrófono y aviso de señal insuficiente | Prototipo separado del entrenamiento; falta validar con micrófono físico |
| 5 | Informe IEEE (12 a 15 páginas) | Pendiente |

## Limitaciones conocidas

- Con 24 señales de prueba, un solo error ya es 4.2 % y dos son 8.3 %. Para sustentar el 5 % hace falta un conjunto de prueba más grande.
- La prueba real es el micrófono: las grabaciones de xeno-canto y el micrófono del computador suenan distinto.
- El aviso de señal insuficiente usa un umbral provisional de amplitud en la interfaz; no forma parte de la decisión matemática entre especies.
- La selección de subbandas y las expresiones matemáticas deben cotejarse con las notas completas de clase antes de considerarse definitivas.
