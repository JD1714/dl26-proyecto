# Plantilla del proyecto final — Transfer learning y fine-tuning con Hugging Face

Este repo trae **hecha la parte de ingeniería** del proyecto, para que el trabajo de ustedes se concentre en el
notebook: elegir el problema, adaptar un modelo del Hub y medir con rigor cuánto aporta cada etapa.

```
dl26-proyecto-template/
├── notebook/plantilla_proyecto.ipynb   # esqueleto del entregable principal (sus experimentos)
├── api/                                # API con FastAPI: carga SU modelo desde el Hub y sirve predicciones
│   ├── main.py                         #   GET /health · POST /predict/text · POST /predict/image
│   ├── inference.py                    #   carga del modelo (una sola vez) e inferencia
│   ├── schemas.py                      #   validación de entrada y estructura de error
│   └── tests/test_api.py               #   pruebas con el modelo simulado (corren sin descargar nada)
├── cliente/                            # aplicación web con Reflex que consume el API
│   ├── rxconfig.py
│   └── cliente/cliente.py              #   la app: estado, llamadas al API, interfaz
├── requirements.txt
├── .env.example                        # MODEL_ID, TASK, API_URL
└── Dockerfile                          # API lista para desplegar (opcional)
```

## Qué tienen que hacer ustedes

1. **El notebook.** Es el 60 % de la nota y es todo suyo: `notebook/plantilla_proyecto.ipynb` trae las secciones del
   enunciado y tres ayudas (semilla = cédula, `registrar()`/`tabla_resultados()` para la tabla de regímenes, y la celda
   de publicación con prueba de recarga). Al terminar, el modelo óptimo queda en el Hub como `usuario/mi-modelo`.
2. **Conectar su modelo.** Copien `.env.example` a `.env` y pongan su `MODEL_ID` y la `TASK` (`text` o `image`).
   Con eso el API y la app ya funcionan con su modelo: no hay que tocar código.
3. **Adaptar la app.** En `cliente/cliente/cliente.py` cambien `TITULO` y `DESCRIPCION` (qué hace el modelo, qué
   entradas acepta y qué no debe esperarse de él, en lenguaje llano, como en su model card).
4. **Al menos una extensión** (cuenta para la nota de integración y para α). Ideas, de menor a mayor:
   - mostrar ejemplos de entrada de prueba con un botón;
   - un interruptor para comparar la predicción del **modelo base sin adaptar** con la del modelo afinado
     (requiere un segundo `MODEL_ID` en el API);
   - un endpoint `/predict/batch`;
   - desplegar el API con el `Dockerfile` o la app en un Space de Hugging Face.
5. **Declarar el uso de IA generativa** al final de este README (ver el enunciado).

## Cómo correrlo

Requiere Python 3.11 o 3.12. Para la app, Reflex descarga por su cuenta lo que necesita del frontend la primera vez.

```bash
pip install -r requirements.txt
cp .env.example .env            # y editar MODEL_ID / TASK

# 1) pruebas del API (no descargan el modelo: lo simulan)
pytest api/tests -q

# 2) el API, desde la raíz del repositorio
uvicorn api.main:app --reload   # http://localhost:8000/docs

# 3) la app, en otra terminal
cd cliente
reflex run                      # http://localhost:3000
```

La primera llamada a `/health` tarda lo que tarde en descargarse el modelo del Hub; después queda en memoria.
Si `/health` responde `"cargado": false`, el campo `detalle` dice por qué (identificador mal escrito, modelo privado
sin token, `TASK` que no corresponde al modelo).

## Cómo está hecho (para que puedan modificarlo)

- **API.** `inference.py` crea un `pipeline` de `transformers` con `MODEL_ID` la primera vez que hace falta y lo
  deja en caché. `main.py` expone `/health` (modelo, tarea, dispositivo, estado de carga) y dos rutas de predicción,
  una por modalidad; solo responde la que coincide con `TASK`. **Toda** respuesta de error tiene la misma forma,
  `{"error": {"codigo": ..., "mensaje": ...}}`, sea por validación (422), formato (400), modelo no cargado (503) o
  fallo de inferencia (500): la app la interpreta sin caerse.
- **App.** `Estado` guarda la entrada del usuario y el resultado; `comprobar_api` se ejecuta al cargar la página y
  decide qué widget mostrar (área de texto o carga de imagen) según la tarea que sirve el API. Las predicciones se
  piden por HTTP con `httpx`; el modelo nunca se carga en la app.
- **Probar sin modelo.** `api/tests/test_api.py` reemplaza el pipeline por uno falso: úsenlo como ejemplo para
  probar sus extensiones.

## Declaración de uso de IA generativa

_Indiquen aquí, por componente, si usaron alguna herramienta de IA generativa, cuál y para qué. Si no usaron
ninguna, escríbanlo explícitamente._
