import reflex as rx

config = rx.Config(
    app_name="cliente",
    backend_port=8001,  # el backend de Reflex no puede usar el 8000: ahí corre el API (uvicorn)
    # El API corre aparte (uvicorn); esta app solo lo consume por HTTP (ver API_URL en cliente/cliente.py)
    plugins=[rx.plugins.RadixThemesPlugin(theme=rx.theme(accent_color="indigo", gray_color="slate", radius="large")), rx.plugins.SitemapPlugin()],
)
