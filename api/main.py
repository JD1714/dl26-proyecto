"""API de predicción para el proyecto final.

Correr desde la raíz del repositorio:
    uvicorn api.main:app --reload
Documentación interactiva: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import inference
from api.schemas import EntradaTexto, Prediccion, RespuestaError, Salud

TIPOS_IMAGEN = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES_IMAGEN = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.precargar()        # carga el modelo una sola vez, al arrancar
    yield


app = FastAPI(
    title="Proyecto final: clasificador afinado con Hugging Face",
    description="Sirve el modelo publicado en el Hub. Configure MODEL_ID y TASK en .env",
    version="1.0",
    lifespan=lifespan,
    responses={400: {"model": RespuestaError}, 422: {"model": RespuestaError}, 500: {"model": RespuestaError}, 503: {"model": RespuestaError}},
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


#  errores con estructura fija
def _error(status: int, codigo: str, mensaje: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"codigo": codigo, "mensaje": mensaje}})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detalle = exc.detail if isinstance(exc.detail, dict) else {"codigo": "http", "mensaje": str(exc.detail)}
    return _error(exc.status_code, detalle["codigo"], detalle["mensaje"])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    campos = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    return _error(422, "validacion", f"Entrada inválida: {campos}")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return _error(500, "interno", f"{type(exc).__name__}: {exc}")


def _exigir_modelo(tarea: str):
    if inference.TASK != tarea:
        raise HTTPException(400, detail={"codigo": "tarea", "mensaje": f"Este API está configurado para TASK={inference.TASK}, no para '{tarea}'"})
    if not inference.ESTADO["cargado"]:
        raise HTTPException(503, detail={"codigo": "modelo", "mensaje": f"El modelo no está cargado: {inference.ESTADO['detalle']}"})


# ----------------------------------------------------------------- rutas
@app.get("/health", response_model=Salud)
def health():
    return Salud(ok=True, cargado=inference.ESTADO["cargado"], modelo=inference.MODEL_ID, tarea=inference.TASK,
                 dispositivo=inference.ESTADO["dispositivo"], detalle=inference.ESTADO["detalle"])


@app.post("/predict/text", response_model=Prediccion)
def predict_text(entrada: EntradaTexto):
    _exigir_modelo("text")
    try:
        res = inference.predecir_texto(entrada.texto)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, detail={"codigo": "inferencia", "mensaje": f"{type(e).__name__}: {e}"})
    return Prediccion(**res, modelo=inference.MODEL_ID, tarea=inference.TASK)


@app.post("/predict/image", response_model=Prediccion)
async def predict_image(archivo: UploadFile = File(..., description="Imagen JPEG, PNG o WebP")):
    _exigir_modelo("image")
    if archivo.content_type not in TIPOS_IMAGEN:
        raise HTTPException(400, detail={"codigo": "formato", "mensaje": f"Tipo de archivo no soportado: {archivo.content_type}. Use JPEG, PNG o WebP"})
    contenido = await archivo.read()
    if len(contenido) > MAX_BYTES_IMAGEN:
        raise HTTPException(400, detail={"codigo": "tamano", "mensaje": "La imagen supera los 10 MB"})
    try:
        res = inference.predecir_imagen(contenido)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, detail={"codigo": "inferencia", "mensaje": f"{type(e).__name__}: {e}"})
    return Prediccion(**res, modelo=inference.MODEL_ID, tarea=inference.TASK)
