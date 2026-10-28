"""Aplicación web del proyecto final (Reflex).

Correr desde la carpeta `cliente/` (con el API ya levantado):
    reflex run
La app NO carga el modelo: le pide las predicciones al API por HTTP.
Configure la URL del API con la variable de entorno API_URL (ver .env.example).
"""
import base64
import os

import httpx
import reflex as rx
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())   # lee API_URL del archivo .env de la raíz del repositorio, si existe

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# TEXTOS QUE LOS ESTUDIANTES DEBEN ADAPTAR
TITULO = "Clasificador de Toxicidad"
DESCRIPCION = (
    "Modelo afinado para detectar y clasificar toxicidad en texto. "
    "Recibe texto en lenguaje natural y predice la clase con sus probabilidades."
)

# identidad visual del curso (colores tomados del logo)
LOGO_URL = "https://ml.ucv.ai/logo.png"
CURSO_URL = "https://docencia.ucv.ai"
AZUL = "#2D55D4"
AMARILLO = "#F7B500"
CARMESI = "#9B1B30"


class Estado(rx.State):
    # estado del API
    api_ok: bool = False
    api_contactado: bool = False
    tarea: str = ""
    modelo: str = ""
    # entrada del usuario
    texto: str = ""
    imagen_preview: str = ""
    # resultado
    clase: str = ""
    confianza: float = 0.0
    probabilidades: list[dict] = []     # [{"clase": ..., "prob": ...}], de mayor a menor
    error: str = ""
    cargando: bool = False

    #  utilidades
    def set_texto(self, valor: str):
        self.texto = valor

    def _limpiar(self):
        self.clase, self.confianza, self.probabilidades, self.error = "", 0.0, [], ""

    def _procesar_respuesta(self, r: httpx.Response):
        """Convierte la respuesta del API (predicción o estructura de error) en estado de la app."""
        try:
            datos = r.json()
        except ValueError:
            self.error = f"El API respondió algo que no es JSON (HTTP {r.status_code})"
            return
        if "error" in datos:                                   # estructura de error del API
            self.error = f"{datos['error']['codigo']}: {datos['error']['mensaje']}"
            return
        self.clase = datos["clase"]
        self.confianza = round(100 * datos["probabilidades"][self.clase], 1)
        self.probabilidades = [
            {"clase": etiqueta, "prob": round(100 * p, 1)}
            for etiqueta, p in datos["probabilidades"].items()
        ]

    #  eventos
    async def comprobar_api(self):
        """Se ejecuta al cargar la página: pregunta al API qué modelo y qué tarea sirve."""
        self._limpiar()
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{API_URL}/health")
            d = r.json()
            self.api_contactado = True
            self.tarea, self.modelo, self.api_ok = d["tarea"], d["modelo"], bool(d["cargado"])
            if not self.api_ok:
                self.error = f"El API está arriba pero el modelo no cargó: {d.get('detalle', '')}"
        except Exception as e:  # noqa: BLE001
            self.api_ok, self.api_contactado = False, False
            self.error = f"No se pudo contactar el API en {API_URL} ({type(e).__name__}). Levántelo con: uvicorn api.main:app"

    async def predecir_texto(self):
        if not self.texto.strip():
            self.error = "Escriba un texto antes de predecir"
            return
        self._limpiar()
        self.cargando = True
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(f"{API_URL}/predict/text", json={"texto": self.texto})
            self._procesar_respuesta(r)
        except Exception as e:  # noqa: BLE001
            self.error = f"Fallo al llamar al API: {type(e).__name__}: {e}"
        finally:
            self.cargando = False

    async def subir_imagen(self, archivos: list[rx.UploadFile]):
        """Recibe la imagen del navegador, la muestra y se la envía al API."""
        if not archivos:
            self.error = "Seleccione una imagen"
            return
        self._limpiar()
        self.cargando = True
        try:
            contenido = await archivos[0].read()
            tipo = archivos[0].content_type or "image/png"
            self.imagen_preview = f"data:{tipo};base64,{base64.b64encode(contenido).decode()}"
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(f"{API_URL}/predict/image", files={"archivo": ("imagen", contenido, tipo)})
            self._procesar_respuesta(r)
        except Exception as e:  # noqa: BLE001
            self.error = f"Fallo al llamar al API: {type(e).__name__}: {e}"
        finally:
            self.cargando = False


# componentes
def encabezado() -> rx.Component:
    """Franja superior con el logo del curso, el título y la descripción del modelo."""
    return rx.box(
        rx.vstack(
            rx.image(src=LOGO_URL, width="120px", height="120px", border_radius="16px",
                     box_shadow="0 6px 24px rgba(0,0,0,0.25)", alt="UCVIA"),
            rx.heading(TITULO, size="8", color="white", text_align="center"),
            rx.text(DESCRIPCION, size="3", color="rgba(255,255,255,0.88)", text_align="center", max_width="640px"),
            align="center",
            spacing="4",
        ),
        background=f"linear-gradient(160deg, {AZUL} 0%, #1E3FA8 100%)",
        border_bottom=f"4px solid {AMARILLO}",
        padding_y="3em",
        padding_x="1.5em",
        width="100%",
    )


def estado_api() -> rx.Component:
    """Línea de estado: qué modelo y qué tarea sirve el API."""
    return rx.cond(
        Estado.api_ok,
        rx.hstack(
            rx.badge(rx.icon("circle_check", size=14), "API conectado", color_scheme="green", variant="soft", size="2"),
            rx.badge("modelo: ", rx.code(Estado.modelo, variant="ghost"), variant="outline", size="2"),
            rx.badge("tarea: ", Estado.tarea, variant="outline", size="2"),
            wrap="wrap",
            spacing="2",
            justify="center",
        ),
        rx.cond(
            Estado.api_contactado,
            rx.badge(rx.icon("circle_alert", size=14), "Modelo no cargado", color_scheme="amber", variant="soft", size="2"),
            rx.badge(rx.icon("circle_x", size=14), "API no disponible", color_scheme="red", variant="soft", size="2"),
        ),
    )


def entrada_texto() -> rx.Component:
    return rx.vstack(
        rx.text_area(
            placeholder="Escriba el texto a clasificar...",
            value=Estado.texto,
            on_change=Estado.set_texto,
            width="100%",
            min_height="140px",
            size="3",
        ),
        rx.button(rx.icon("sparkles", size=16), "Predecir", on_click=Estado.predecir_texto,
                  loading=Estado.cargando, size="3", width="100%"),
        width="100%",
        spacing="3",
    )


def entrada_imagen() -> rx.Component:
    return rx.vstack(
        rx.upload(
            rx.vstack(
                rx.icon("image_up", size=36, color="var(--accent-10)"),
                rx.text("Arrastre una imagen o haga clic para elegirla", weight="medium"),
                rx.text("JPEG, PNG o WebP · hasta 10 MB", size="2", color_scheme="gray"),
                rx.foreach(rx.selected_files("subida"), lambda f: rx.badge(f, variant="soft", size="2")),
                align="center",
                spacing="2",
            ),
            id="subida",
            accept={"image/jpeg": [".jpg", ".jpeg"], "image/png": [".png"], "image/webp": [".webp"]},
            max_files=1,
            border="2px dashed var(--gray-7)",
            border_radius="12px",
            padding="2.5em 1em",
            width="100%",
            cursor="pointer",
            _hover={"border_color": AZUL, "background": "var(--accent-2)"},
        ),
        rx.button(
            rx.icon("sparkles", size=16),
            "Predecir",
            on_click=Estado.subir_imagen(rx.upload_files(upload_id="subida")),
            loading=Estado.cargando,
            size="3",
            width="100%",
        ),
        width="100%",
        spacing="3",
    )


def barra_probabilidad(fila: rx.Var) -> rx.Component:
    """Una fila por clase: etiqueta, barra y porcentaje. La clase ganadora se resalta."""
    ganadora = fila["clase"] == Estado.clase
    return rx.vstack(
        rx.hstack(
            rx.text(fila["clase"], weight=rx.cond(ganadora, "bold", "regular"), size="2"),
            rx.spacer(),
            rx.text(fila["prob"], "%", size="2", color_scheme="gray"),
            width="100%",
        ),
        rx.box(
            rx.box(
                width=fila["prob"].to_string() + "%",
                height="100%",
                background=rx.cond(ganadora, AZUL, "var(--gray-8)"),
                border_radius="999px",
                transition="width 0.4s ease",
            ),
            width="100%",
            height="10px",
            background="var(--gray-4)",
            border_radius="999px",
            overflow="hidden",
        ),
        width="100%",
        spacing="1",
    )


def resultado() -> rx.Component:
    return rx.vstack(
        rx.cond(Estado.error != "", rx.callout(Estado.error, icon="triangle_alert", color_scheme="red", width="100%")),
        rx.cond(
            Estado.clase != "",
            rx.card(
                rx.vstack(
                    rx.text("Predicción", size="2", color_scheme="gray", weight="medium"),
                    rx.hstack(
                        rx.heading(Estado.clase, size="7", color="var(--accent-11)"),
                        rx.badge(Estado.confianza, "% de confianza", size="2", variant="soft", color_scheme="amber"),
                        align="center",
                        wrap="wrap",
                        spacing="3",
                    ),
                    rx.cond(
                        (Estado.tarea == "image") & (Estado.imagen_preview != ""),
                        rx.image(src=Estado.imagen_preview, max_height="260px", border_radius="10px", margin_y="0.5em"),
                    ),
                    rx.divider(),
                    rx.text("Probabilidad por clase", size="2", color_scheme="gray", weight="medium"),
                    rx.foreach(Estado.probabilidades, barra_probabilidad),
                    width="100%",
                    spacing="3",
                ),
                width="100%",
                size="3",
                border_top=f"4px solid {AMARILLO}",
            ),
        ),
        width="100%",
        spacing="4",
    )


def pie() -> rx.Component:
    """Franja inferior con el logo, el nombre del curso y el enlace al sitio de docencia."""
    return rx.box(
        rx.flex(
            rx.hstack(
                rx.image(src=LOGO_URL, width="40px", height="40px", border_radius="8px", alt="UCVIA"),
                rx.vstack(
                    rx.text("Proyecto final", size="2", weight="bold", color="white"),
                    rx.text("Redes Neuronales Profundas · UCV", size="2", color="rgba(255,255,255,0.75)"),
                    spacing="0",
                ),
                align="center",
                spacing="3",
            ),
            rx.link(
                rx.hstack(rx.text(CURSO_URL.removeprefix("https://")), rx.icon("external_link", size=14),
                          align="center", spacing="1"),
                href=CURSO_URL,
                is_external=True,
                size="2",
                weight="medium",
                color=AMARILLO,
                underline="hover",
            ),
            justify="between",
            align="center",
            wrap="wrap",
            gap="1em",
            max_width="688px",
            margin_x="auto",
        ),
        background="linear-gradient(160deg, #1E3FA8 0%, #172F80 100%)",
        border_top=f"4px solid {CARMESI}",
        padding_y="1.5em",
        padding_x="1.5em",
        width="100%",
        margin_top="3em",
    )


def index() -> rx.Component:
    return rx.box(
        encabezado(),
        rx.container(
            rx.vstack(
                estado_api(),
                rx.cond(
                    (Estado.tarea == "text") | (Estado.tarea == "image"),
                    rx.card(
                        rx.cond(Estado.tarea == "text", entrada_texto(), entrada_imagen()),
                        width="100%",
                        size="3",
                    ),
                ),
                resultado(),
                spacing="5",
                width="100%",
                align="center",
                padding_top="2em",
            ),
            size="2",
            flex="1",
            width="100%",
        ),
        pie(),
        display="flex",
        flex_direction="column",
        min_height="100vh",
        background="var(--gray-2)",
    )


app = rx.App(
    stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"],
    style={"font_family": "Inter, system-ui, -apple-system, sans-serif"},
    head_components=[rx.el.link(rel="icon", href=LOGO_URL)],
)
app.add_page(index, title=TITULO, on_load=Estado.comprobar_api)
