# Repository Guidelines

## Project Structure & Module Organization

`src/` contains the shared audio pipeline; `src/banco_filtros.py` holds the math used by both training and recognition. `app/fase2_entrenamiento.py` writes the model, `app/fase3_reconocimiento.py` loads it, and `app/fase4_interfaz.py` presents results. Keep phase 1 stable. Tests live in `tests/`; `experimentos/piloto_aves.py` is exploratory. WAV recordings live in `datos/banco_aves/entrenamiento/` and `datos/banco_aves/prueba/`, organized by species. Keep attribution in `datos/banco_aves/creditos.csv`. Generated plots belong in `resultados/figuras/`; `modelo/vectores_energia.json` is the current provisional model.

## Build, Test, and Development Commands

Run these commands from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/fase1_captura_filtro.py --wav datos/banco_aves/prueba/paloma/paloma_008_XC272805_r2.wav --no-mostrar
python app/fase2_entrenamiento.py --grafica resultados/figuras/espectros_promedio.png
python app/fase3_reconocimiento.py --evaluar datos/banco_aves
python app/fase4_interfaz.py
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q src app experimentos
```

The phase 2 command explicitly creates the model; phase 3 evaluates without training. The WAV command checks phase 1 without a microphone. `compileall` checks Python syntax. There is no separate build step.

## Coding Style & Naming Conventions

Use Python with four-space indentation, `snake_case` for modules and functions, and uppercase names for constants such as `FS` and `DURACION`. Keep reusable signal processing in `src/` and command-line or UI code in `app/`. Follow existing NumPy-based array handling. No formatter or linter is configured; keep imports grouped and changes consistent with nearby code.

## Testing Guidelines

Use `unittest` with focused `tests/test_*.py` files. No coverage threshold is configured. Test the phase 2/3 model contract and keep both phases on the same `acondicionar()` path. Use held-out `prueba/` recordings for evaluation; do not train on them.

## Commit & Pull Request Guidelines

Recent commits use short descriptive subjects in Spanish or English, without a fixed prefix convention. Write an imperative subject that names the change, such as `Add subband energy tests`. In pull requests, describe the behavior changed, commands run, and any effect on audio data or model parameters. Include a relevant figure or screenshot for plot or UI changes, and link an issue when one exists.

## Data & Model Consistency

Keep audio at the configured 22,050 Hz and five-second duration when adding bank recordings. Preserve source credits. If the bank, preprocessing filter, or subband boundaries change, explicitly rerun phase 2. Never train from phase 3 or the interface; keep the JSON schema stable so the interface can display revised models.
