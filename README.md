# Reconocimiento de aves por banco de filtros

Reconoce tras una captura de cinco segundos, con el micrófono o un archivo, dos especies por su canto: paloma torcaza (*Zenaida auriculata*) y copetón (*Zonotrichia capensis*).

La interfaz no entrena. La fase 2 genera explícitamente un modelo JSON; la fase 3 lo carga para reconocer y evaluar; la fase 4 solo presenta los resultados.

## Cómo funciona

1. Grabaciones de duración fija (5 s). Si hay ruido, se filtran.
2. FFT completa de N puntos de cada grabación (k = 0 … N−1) y magnitud ‖X(k)‖.
3. Espectro promedio por especie ‖C(k)‖ₘ y gráfica de ambas especies en un mismo plano, solo para escoger los cortes.
4. Se eligen 3 a 8 subbandas donde la energía de las especies sea diferente.
5. Energía de cada grabación en cada subbanda: E_i = (1/L_i) · Σ ‖X(k)‖². Luego se promedian los vectores de todas las grabaciones de la especie (E_A, E_B). Estos vectores se guardan en el modelo.
6. En reconocimiento: ‖X(k)‖ de la señal nueva, partido en las mismas subbandas, da E_X.
7. Para cada especie se calcula d = Σ |E_Xi − E_Ci|. La especie con el valor mínimo es la que se reconoce.

Así lo resolvió el profesor en clase (ejercicio A = {(1,0,0,2), (2,1,0,0)}, 2 subbandas: E_mA = [7 3]; reconocimiento con d(E_x, E_A) = 5 + 1 + 3 = 9 y d(E_x, E_B) = 7). `tests/test_fases.py` reproduce ese ejercicio. El filtro, la normalización y el recorte por máxima energía son acondicionamiento adicional, aceptado por el monitor.

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

- Fuentes: xeno-canto e iNaturalist (solo grado de investigación), de muchos autores y países. Los créditos están en `datos/banco_aves/creditos.csv`.
- Formato: WAV mono, 22050 Hz, 5 s.
- Entrenamiento: 85 paloma + 129 copetón. Prueba: 32 paloma + 56 copetón.
- Entrenamiento y prueba se separan por grabador: nadie que grabó un audio de prueba aparece en el entrenamiento de esa especie, y una misma grabación nunca está en los dos conjuntos.
- `datos/descartados/` guarda audios retirados del banco (duplicados, un llamado en vez de canto, un audio sin fuente). No se usan.

## Decisiones importantes

**Un solo acondicionamiento para todo.** El entrenamiento y el reconocimiento en vivo deben llamar a la misma función, `acondicionar()`. Si el banco se procesa de una forma y el micrófono de otra, los vectores de energía dejan de ser comparables y el reconocimiento falla aunque el algoritmo sea correcto. Si se cambia el filtro, hay que volver a entrenar.

**Entrenar únicamente bajo petición.** `app/fase2_entrenamiento.py` procesa el banco y guarda los bordes, los vectores de energía y los parámetros en `modelo/vectores_energia.json`. La fase 3 lee ese archivo sin modificarlo; la interfaz tampoco entrena. El modelo provisional anterior permanece en esa ruta y también se puede leer. Si se agregan audios o cambian las bandas, se ejecuta de nuevo la fase 2 para reemplazarlo.

**Filtros escritos a mano, no con scipy.** 
- Butterworth pasabandas de orden 2 por transformación bilineal con prewarping, aplicado con la ecuación en diferencias. Es el pre-filtro de 200 a 8000 Hz: quita lo que no es canto (viento, golpes al micrófono, ruido muy agudo).
- FIR pasabajas con ventana Hamming. Solo se usa en `captura.diezmar()`, cuando el micrófono no abre a 22050 Hz: se graba a 44100 Hz y se reduce a la mitad, filtrando antes para evitar aliasing.

**Normalización de energía: es un agregado nuestro.** No aparece en el enunciado del proyecto. Se hace porque el algoritmo compara energías absolutas contra vectores de referencia, y con el micrófono la energía sube o baja según el volumen y la distancia al ave. Se normaliza por energía total (no por el valor pico), porque un solo golpe al micrófono cambia el pico de toda la grabación pero casi no cambia su energía total. Debe mencionarse en el informe como parte del acondicionamiento, no del método principal.

**FFT.** Se usa la FFT completa de N puntos (`np.fft.fft`), como `y = fft(x)` en clase. La frecuencia del punto k es k·fs/N; la segunda mitad (de fs/2 a fs) es el espejo de la primera porque la señal es real. Las subbandas se definen en la primera mitad, donde está la información.

**Subbandas: 3, escogidas sobre los espectros promedio.** B1 = 200–1500 Hz (arrullo de la paloma, ~300–1200 Hz), B2 = 1500–3000 Hz (transición), B3 = 3000–8000 Hz (canto del copetón, ~2500–7000 Hz). Empiezan en 200 Hz y terminan en 8000 Hz porque el pre-filtro elimina lo que está fuera. La figura es `resultados/figuras/espectros_promedio.png`.

**Rechazo: «sin canto» y «no reconocido».** No cambia la decisión del profesor (menor suma de diferencias absolutas); solo evita dar una especie cuando no corresponde.
- *Sin canto* (agregado nuestro): si el rango dinámico de la señal filtrada (energía de cuadros de 50 ms, percentil 95 frente a percentil 10) es menor a 6 dB, no hay sílabas y pausas, así que es ruido constante, zumbido o silencio. En el banco, el canto más plano tiene 11 dB; los ruidos de prueba dan menos de 2 dB.
- *No reconocido* (nota del profesor, pág. 45: agregar la desviación estándar para tener intervalos de energía): la fase 2 guarda, por especie y subbanda, la media y la desviación de las energías de cada grabación, en dB. Si el audio cae fuera de media ± 3·desviación en alguna subbanda de la especie más cercana, se responde «no reconocido». Se usa dB y una desviación mínima de 1 dB porque, al normalizar la energía, la subbanda dominante casi no varía en el banco (0.1–0.2 dB) y el intervalo saldría demasiado estrecho.

## Preparación y uso

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/fase1_captura_filtro.py --wav datos/banco_aves/prueba/paloma/paloma_008_XC272805_r2.wav --no-mostrar
```

En Windows, activa el entorno con `.venv\Scripts\activate`. Para el micrófono se requiere un dispositivo de entrada compatible con PortAudio. La fase 1 también admite `--reproducir` y captura directamente si se omite `--wav`.

## Ejemplo de uso de las fases 2 a 4

Los bordes iniciales son una base ajustable, no una selección de bandas ya validada. La previsualización dibuja los dos espectros promedio y las líneas divisorias de las subbandas propuestas, sin reemplazar el modelo. Ajusta `--bordes` donde las curvas y sus energías se distingan; usa en el informe la gráfica correspondiente a los bordes finalmente elegidos.

```bash
python app/fase2_entrenamiento.py --solo-grafica --bordes 200,1500,3000,8000 --grafica resultados/figuras/espectros_promedio.png
python app/fase3_reconocimiento.py --audio datos/banco_aves/prueba/paloma/paloma_008_XC272805_r2.wav
python app/fase3_reconocimiento.py --evaluar datos/banco_aves
python app/fase4_interfaz.py
python -m unittest discover -s tests -p 'test_*.py'
```

Cuando los cortes estén revisados, ejecuta la misma orden sin `--solo-grafica` para guardar el modelo. La fase 2 es la única que lo crea. La evaluación imprime aciertos y porcentaje de error. La interfaz vuelve a leer el modelo al analizar un audio: los nuevos vectores, nombres y bordes se muestran sin cambiar el código de la pantalla. Si falta un modelo válido, la interfaz lo informa y no intenta entrenarlo. El icono de sol o luna del encabezado alterna entre modo claro y oscuro; la elección se conserva al volver a abrir la aplicación.

## Estado y fases

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Captura por micrófono + filtro | Hecha |
| 2 | Entrenamiento: FFT, espectro promedio, subbandas, energías, JSON | Hecha: 3 subbandas escogidas sobre los espectros, con intervalos de desviación |
| 3 | Reconocimiento y evaluación del error (meta: ≤ 5 %) | Hecha: 88/88 aciertos en prueba (0 %), con «sin canto» y «no reconocido» |
| 4 | Interfaz, micrófono y aviso de señal insuficiente | Hecha; falta validar con micrófono físico |
| 5 | Informe IEEE (12 a 15 páginas) | Pendiente |

## Limitaciones conocidas

- 0 errores en 88 pruebas: con 95 % de confianza el error real del banco es menor a ~3.4 % (regla del tres, 3/88). Los audios del banco pasaron un filtro de calidad, así que son cantos limpios.
- La prueba real es el micrófono: las grabaciones de xeno-canto y el micrófono del computador suenan distinto. Los umbrales de rechazo (6 dB, k = 3) se calibraron con el banco y ruidos sintéticos; hay que revisarlos con grabaciones reales.
- Otra ave que cante en la misma banda que el copetón (por ejemplo el sirirí) se reconoce como copetón: con 3 subbandas de energía no se puede distinguir. Solo hay dos clases.
