"""Pruebas del API con el modelo simulado: corren en segundos y sin descargar nada.

    pytest api/tests -q
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import inference
from api.main import app


class PipelineFalso:
    """Imita la salida de un pipeline de transformers con top_k=None."""
    def __call__(self, entrada):
        return [[{"label": "positivo", "score": 0.91}, {"label": "negativo", "score": 0.09}]]


@pytest.fixture
def cliente(monkeypatch):
    inference.get_pipeline.cache_clear()
    monkeypatch.setattr(inference, "get_pipeline", lambda: PipelineFalso())
    inference.ESTADO.update(cargado=True, detalle="", dispositivo="cpu")
    with TestClient(app) as c:
        yield c


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] and d["cargado"] and d["tarea"] in ("text", "image")


def test_predict_text_ok(cliente, monkeypatch):
    monkeypatch.setattr(inference, "TASK", "text")
    r = cliente.post("/predict/text", json={"texto": "me encantó el curso"})
    assert r.status_code == 200
    d = r.json()
    assert d["clase"] == "positivo"
    assert set(d["probabilidades"]) == {"positivo", "negativo"}
    assert abs(sum(d["probabilidades"].values()) - 1) < 1e-6


def test_predict_text_entrada_invalida(cliente, monkeypatch):
    monkeypatch.setattr(inference, "TASK", "text")
    r = cliente.post("/predict/text", json={"texto": ""})
    assert r.status_code == 422
    assert r.json()["error"]["codigo"] == "validacion"


def test_tarea_equivocada(cliente, monkeypatch):
    monkeypatch.setattr(inference, "TASK", "image")
    r = cliente.post("/predict/text", json={"texto": "hola"})
    assert r.status_code == 400
    assert r.json()["error"]["codigo"] == "tarea"


def test_predict_image_ok(cliente, monkeypatch):
    monkeypatch.setattr(inference, "TASK", "image")
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, format="PNG")
    r = cliente.post("/predict/image", files={"archivo": ("hoja.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    assert r.json()["clase"] == "positivo"


def test_predict_image_formato(cliente, monkeypatch):
    monkeypatch.setattr(inference, "TASK", "image")
    r = cliente.post("/predict/image", files={"archivo": ("x.txt", b"no soy una imagen", "text/plain")})
    assert r.status_code == 400
    assert r.json()["error"]["codigo"] == "formato"


def test_modelo_no_cargado(cliente, monkeypatch):
    monkeypatch.setattr(inference, "TASK", "text")
    inference.ESTADO.update(cargado=False, detalle="OSError: modelo inexistente")
    r = cliente.post("/predict/text", json={"texto": "hola"})
    assert r.status_code == 503
    assert r.json()["error"]["codigo"] == "modelo"
