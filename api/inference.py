"""Carga del modelo desde el Hub de Hugging Face e inferencia.

Se configura con dos variables de entorno (ver .env.example):
  MODEL_ID  identificador del modelo en el Hub, p. ej. "usuario/mi-modelo"
  TASK      "text" (clasificación de texto) o "image" (clasificación de imágenes)

El modelo se carga UNA sola vez (la primera vez que hace falta) y queda en memoria.
"""
import os
import time
from functools import lru_cache
from io import BytesIO

from dotenv import find_dotenv, load_dotenv
from PIL import Image

load_dotenv(find_dotenv())   # lee MODEL_ID y TASK del archivo .env de la raíz del repositorio, si existe

MODEL_ID = os.getenv("MODEL_ID", "usuario/mi-modelo")
TASK = os.getenv("TASK", "text").lower()

_TAREA_PIPELINE = {"text": "text-classification", "image": "image-classification"}

if TASK not in _TAREA_PIPELINE:
    raise ValueError(f"TASK debe ser 'text' o 'image', no '{TASK}'")

# Estado del modelo, lo consulta /health
ESTADO = {"cargado": False, "detalle": "", "dispositivo": "", "segundos_carga": None}


@lru_cache(maxsize=1)
def get_pipeline():
    """Crea el pipeline de transformers para MODEL_ID. Lento la primera vez; después queda en caché."""
    import torch
    from transformers import pipeline

    dispositivo = 0 if torch.cuda.is_available() else -1
    t0 = time.time()
    clf = pipeline(_TAREA_PIPELINE[TASK], model=MODEL_ID, top_k=None, device=dispositivo)
    ESTADO.update(cargado=True, detalle="", dispositivo="cuda" if dispositivo == 0 else "cpu", segundos_carga=round(time.time() - t0, 1))
    return clf


def precargar():
    """Intenta cargar el modelo al arrancar el API. Si falla, el API arranca igual y /health explica el error."""
    try:
        get_pipeline()
    except Exception as e:  # noqa: BLE001 - queremos reportar cualquier causa en /health
        ESTADO.update(cargado=False, detalle=f"{type(e).__name__}: {e}")


def _formatear(salidas) -> dict:
    """Normaliza la salida del pipeline a {"clase": ..., "probabilidades": {...}} ordenada de mayor a menor."""
    # text-classification con top_k=None puede devolver [[{...}]] para una sola entrada; image-classification devuelve [{...}]
    if salidas and isinstance(salidas[0], list):
        salidas = salidas[0]
    salidas = sorted(salidas, key=lambda r: r["score"], reverse=True)
    return {
        "clase": salidas[0]["label"],
        "probabilidades": {r["label"]: round(float(r["score"]), 4) for r in salidas},
    }


def predecir_texto(texto: str) -> dict:
    return _formatear(get_pipeline()(texto))


def predecir_imagen(contenido: bytes) -> dict:
    imagen = Image.open(BytesIO(contenido)).convert("RGB")
    return _formatear(get_pipeline()(imagen))
