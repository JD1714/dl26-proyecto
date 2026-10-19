"""Esquemas de entrada y salida del API (validación con pydantic)."""
from pydantic import BaseModel, Field


class EntradaTexto(BaseModel):
    texto: str = Field(min_length=1, max_length=2000, description="Texto a clasificar")


class Prediccion(BaseModel):
    clase: str
    probabilidades: dict[str, float]
    modelo: str
    tarea: str


class Salud(BaseModel):
    ok: bool
    cargado: bool
    modelo: str
    tarea: str
    dispositivo: str
    detalle: str = ""


class Error(BaseModel):
    """Estructura de error que la aplicación sabe interpretar."""
    codigo: str
    mensaje: str


class RespuestaError(BaseModel):
    error: Error
