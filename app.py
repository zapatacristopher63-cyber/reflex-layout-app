import os
import base64
import re
import tempfile
import textwrap
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yt_dlp
from ultralytics import YOLO

VIDEO_PRUEBA_URL = "https://drive.google.com/file/d/1qiK0plB-cUAJBcdLZd61bBHzvmySdlKK/view?usp=drivesdk"

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL Y BRANDING
# ──────────────────────────────────────────────────────────────────────────
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

st.set_page_config(
    page_title="Reflex Layout 360",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "⬛",
    layout="wide",
)


def _logo_base64() -> str:
    if not os.path.exists(LOGO_PATH):
        return ""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


LOGO_B64 = _logo_base64()

# Paleta extravagante construida a partir de los colores reales del logo
# (naranja #EF6824 y verde lima #7FB02B), con un dorado de acento para dar
# un toque llamativo y de alto contraste.
NARANJA = "#FF6B1A"
NARANJA_2 = "#FF9142"
VERDE_LIMA = "#8FCB2E"
VERDE_OSCURO = "#5C8A1E"
DORADO = "#FFC93C"
GRADIENTE = f"linear-gradient(115deg, {NARANJA} 0%, {DORADO} 45%, {VERDE_LIMA} 100%)"
GRADIENTE_SUAVE = f"linear-gradient(115deg, {NARANJA_2}22, {DORADO}22, {VERDE_LIMA}22)"

TEMAS = {
    "🌙 Oscuro": {
        "bg_a": "#1c1206",
        "bg_b": "#0c0904",
        "blob_1": f"{NARANJA}33",
        "blob_2": f"{VERDE_LIMA}2b",
        "texto": "#fbf3e7",
        "subtexto": "#e8c9a0",
        "card": "rgba(255,255,255,0.06)",
        "card_borde": "rgba(255,177,72,0.28)",
        "sidebar": "#170f07",
        "input_bg": "rgba(255,255,255,0.08)",
    },
    "☀️ Claro": {
        "bg_a": "#fff8ec",
        "bg_b": "#f2f9e3",
        "blob_1": f"{NARANJA}22",
        "blob_2": f"{VERDE_LIMA}25",
        "texto": "#2e2410",
        "subtexto": "#7a5a24",
        "card": "rgba(255,255,255,0.8)",
        "card_borde": "rgba(143,203,46,0.3)",
        "sidebar": "#fffaf0",
        "input_bg": "rgba(143,203,46,0.08)",
    },
}

if "tema" not in st.session_state:
    st.session_state.tema = "🌙 Oscuro"

# ──────────────────────────────────────────────────────────────────────────
# PANEL LATERAL
# ──────────────────────────────────────────────────────────────────────────
if LOGO_B64:
    st.sidebar.image(LOGO_PATH, use_container_width=True)

st.sidebar.selectbox("🎨 Apariencia", list(TEMAS.keys()), key="tema")

st.sidebar.header("⚙️ Motor de Procesamiento")
salto_frames = st.sidebar.slider(
    "Salto de frames", 1, 5, 2, help="Acelera el análisis saltando frames del video."
)
opacidad = st.sidebar.slider("Opacidad del Termógrafo", 0.1, 1.0, 0.55)

T = TEMAS[st.session_state.tema]

# NOTA: st.markdown convierte en "bloque de código" cualquier línea con 4+
# espacios de sangría al inicio. Por eso el CSS/HTML se pasa por
# textwrap.dedent() y arranca en la columna 0, para que se renderice como
# HTML real y no como texto plano.
CUSTOM_CSS = textwrap.dedent(
    f"""\
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700;900&family=Manrope:wght@400;500;600;700&display=swap');

    #MainMenu, footer, header {{visibility: hidden;}}

    html, body, [class*="css"] {{
        font-family: 'Manrope', sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 8% 8%, {T['blob_1']} 0%, transparent 40%),
            radial-gradient(circle at 92% 18%, {T['blob_2']} 0%, transparent 38%),
            radial-gradient(circle at 20% 95%, {T['blob_2']} 0%, transparent 35%),
            linear-gradient(160deg, {T['bg_a']} 0%, {T['bg_b']} 100%);
        color: {T['texto']};
    }}

    .rl360-header {{
        display: flex;
        align-items: center;
        gap: 22px;
        padding: 4px 0 10px 0;
        position: relative;
    }}
    .rl360-header img {{
        height: 70px;
        filter: drop-shadow(0 0 18px {NARANJA}55);
    }}
    .rl360-header h1 {{
        font-family: 'Unbounded', sans-serif;
        font-weight: 900;
        font-size: 2.5rem;
        letter-spacing: -0.5px;
        margin: 0;
        background: {GRADIENTE};
        background-size: 280% 280%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: rl360-glow 7s ease infinite;
        text-transform: uppercase;
    }}
    @keyframes rl360-glow {{
        0% {{background-position: 0% 50%;}}
        50% {{background-position: 100% 50%;}}
        100% {{background-position: 0% 50%;}}
    }}
    .rl360-subtitle {{
        color: {T['subtexto']};
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 2px;
        letter-spacing: 0.2px;
    }}

    h2, h3, h4 {{
        font-family: 'Unbounded', sans-serif;
        font-weight: 700;
    }}

    .rl360-card {{
        background: {T['card']};
        border: 1.5px solid {T['card_borde']};
        border-radius: 20px;
        padding: 18px 22px;
        margin-bottom: 14px;
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 28px rgba(0,0,0,0.18);
    }}

    div[data-testid="stMetric"] {{
        background: {T['card']};
        border: 1.5px solid {T['card_borde']};
        border-radius: 16px;
        padding: 10px 14px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Unbounded', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        border-radius: 14px 14px 0 0;
    }}
    .stTabs [aria-selected="true"] {{
        background: {GRADIENTE_SUAVE};
        border-bottom: 3px solid transparent;
        border-image: {GRADIENTE};
        border-image-slice: 1;
    }}

    .stButton>button, .stDownloadButton>button {{
        font-family: 'Unbounded', sans-serif;
        background: {GRADIENTE};
        background-size: 250% 250%;
        color: #1c1206;
        border: none;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.55rem 1.5rem;
        box-shadow: 0 6px 18px {NARANJA}44;
        transition: 0.25s ease;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        background-position: 100% 0%;
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 24px {NARANJA}66;
    }}

    div[data-testid="stTextInput"] input, div[data-testid="stFileUploaderDropzone"] {{
        background: {T['input_bg']} !important;
        border-radius: 14px !important;
        border: 1.5px solid {T['card_borde']} !important;
    }}

    [data-testid="stSidebar"] {{
        background: {T['sidebar']};
        border-right: 1.5px solid {T['card_borde']};
    }}

    div[data-testid="stExpander"] {{
        background: {T['card']};
        border-radius: 16px;
        border: 1.5px solid {T['card_borde']};
    }}

    .rl360-badge {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        background: {GRADIENTE};
        color: #1c1206;
        font-family: 'Unbounded', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
    }}

    hr {{
        border: none;
        height: 3px;
        border-radius: 3px;
        background: {GRADIENTE};
        opacity: 0.6;
    }}
    </style>
    """
)

_logo_img_tag = f'<img src="data:image/png;base64,{LOGO_B64}">' if LOGO_B64 else ""
HEADER_HTML = textwrap.dedent(
    f"""\
    <div class="rl360-header">
    {_logo_img_tag}
    <div>
    <h1>Reflex Layout 360</h1>
    <div class="rl360-subtitle">🛒 Inteligencia Espacial y Gemelos Digitales para Retail</div>
    </div>
    </div>
    """
)

st.markdown(CUSTOM_CSS + HEADER_HTML, unsafe_allow_html=True)

st.write(
    "Sube el metraje de tus cámaras de seguridad. Nuestro motor de IA mapeará "
    "el flujo peatonal y generará decisiones estratégicas de layout al instante 🛍️🔥"
)
st.divider()


# ──────────────────────────────────────────────────────────────────────────
# MODELO (cacheado)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando motor de visión computacional...")
def cargar_modelo():
    return YOLO("yolov8n.pt")


model = cargar_modelo()

# ──────────────────────────────────────────────────────────────────────────
# DESCARGA DE VIDEO DESDE ENLACE
# ──────────────────────────────────────────────────────────────────────────
# YouTube (y otras plataformas de streaming) bloquean de forma activa las IPs
# de datacenter exigiendo verificación "no soy un robot" — no hay forma 100%
# garantizada de evitarlo desde un servidor en la nube.
#
# Google Drive, Dropbox y OneDrive son distintos: son solo almacenamiento de
# archivos, NO hacen detección anti-bot, así que una descarga directa por
# HTTP funciona de forma mucho más confiable. Por eso el flujo es:
#   1) Si el enlace es de Drive/Dropbox/OneDrive -> descarga directa (requests)
#   2) Si no, se asume plataforma de video -> yt-dlp (YouTube, Vimeo, etc.)
CLIENTES_YT = ["android", "ios", "web"]


def _es_html(resp: requests.Response) -> bool:
    """Detecta si la respuesta es una página de error/login en vez del archivo."""
    return "text/html" in resp.headers.get("Content-Type", "")


def _guardar_stream(resp: requests.Response, destino: str) -> None:
    with open(destino, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def descargar_google_drive(url: str, destino: str) -> None:
    match = re.search(r"/d/([a-zA-Z0-9_-]+)|[?&]id=([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("No se pudo extraer el ID del archivo de Google Drive.")
    file_id = match.group(1) or match.group(2)

    session = requests.Session()
    base = "https://drive.google.com/uc?export=download"
    resp = session.get(base, params={"id": file_id}, stream=True, timeout=30)

    # Google Drive interpone una página de confirmación en archivos grandes.
    token = next((v for k, v in resp.cookies.items() if k.startswith("download_warning")), None)
    if token:
        resp = session.get(base, params={"id": file_id, "confirm": token}, stream=True, timeout=30)

    if _es_html(resp):
        raise ValueError("El archivo de Drive no es público o no se pudo confirmar la descarga.")
    _guardar_stream(resp, destino)


def descargar_dropbox(url: str, destino: str) -> None:
    url_directo = re.sub(r"([?&])dl=0", r"\1dl=1", url)
    if "dl=1" not in url_directo:
        url_directo += ("&" if "?" in url_directo else "?") + "dl=1"
    resp = requests.get(url_directo, stream=True, timeout=30, allow_redirects=True)
    if _es_html(resp):
        raise ValueError("El enlace de Dropbox no permite descarga directa.")
    _guardar_stream(resp, destino)


def descargar_onedrive(url: str, destino: str) -> None:
    url_directo = url
    if "1drv.ms" in url:
        # Los enlaces cortos redirigen a la URL real de onedrive.live.com
        redir = requests.get(url, allow_redirects=True, timeout=30)
        url_directo = redir.url
    if "download=1" not in url_directo:
        url_directo += ("&" if "?" in url_directo else "?") + "download=1"
    resp = requests.get(url_directo, stream=True, timeout=30, allow_redirects=True)
    if _es_html(resp):
        raise ValueError("El enlace de OneDrive no permite descarga directa (revisa que sea público).")
    _guardar_stream(resp, destino)


def descargar_youtube(url: str, destino: str, cookies_path: Optional[str]) -> None:
    ultimo_error = None
    for cliente in CLIENTES_YT:
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": destino,
            "quiet": True,
            "noplaylist": True,
            "geo_bypass": True,
            "retries": 3,
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
            "extractor_args": {"youtube": {"player_client": [cliente]}},
        }
        if cookies_path:
            ydl_opts["cookiefile"] = cookies_path
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return
        except Exception as e:  # noqa: BLE001
            ultimo_error = e
            continue
    raise RuntimeError(str(ultimo_error))


def descargar_medio(url: str, destino: str, cookies_path: Optional[str]) -> str:
    """Detecta el proveedor y descarga el video. Devuelve el nombre del proveedor usado."""
    dominio = urlparse(url).netloc.lower()

    if "drive.google.com" in dominio:
        descargar_google_drive(url, destino)
        return "Google Drive"
    if "dropbox.com" in dominio:
        descargar_dropbox(url, destino)
        return "Dropbox"
    if "1drv.ms" in dominio or "onedrive.live.com" in dominio or "sharepoint.com" in dominio:
        descargar_onedrive(url, destino)
        return "OneDrive"

    descargar_youtube(url, destino, cookies_path)
    return "YouTube / otro"


# ──────────────────────────────────────────────────────────────────────────
# ZONA DE ENTRADA DE VIDEO
# ──────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📁 Subir Archivo Local", "🔗 Pegar Enlace Público"])

origen_video = None

with tab1:
    archivo_video = st.file_uploader(
        "Arrastra tu archivo de video aquí (.mp4)", type=["mp4"]
    )
    if archivo_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(archivo_video.read())
        origen_video = tfile.name

with tab2:
    st.caption(
        "Compatibles: **Google Drive**, **Dropbox** y **OneDrive** (recomendado, "
        "sin bloqueos) y YouTube / otras plataformas de video (con posibles "
        "restricciones anti-bot). En Drive/Dropbox/OneDrive asegúrate de que "
        "el enlace tenga permiso público o 'cualquiera con el enlace'."
    )
    st.caption(
        f"🎬 ¿Quieres probar la app sin subir nada? Usa este video de prueba: {VIDEO_PRUEBA_URL}"
    )
    url_video = st.text_input(
        "Pega el enlace del video:",
        placeholder="https://drive.google.com/file/d/... o https://www.dropbox.com/s/...",
    )

    with st.expander("🍪 Opcional: subir cookies.txt (solo para YouTube, si bloquea la descarga)"):
        st.caption(
            "Exporta las cookies de una sesión activa de YouTube en tu navegador "
            "(extensión 'Get cookies.txt') y súbelas aquí. Esto suele resolver "
            "el bloqueo de verificación anti-bot. No aplica a Drive/Dropbox/OneDrive."
        )
        archivo_cookies = st.file_uploader("cookies.txt", type=["txt"], key="cookies")

    cookies_path = None
    if archivo_cookies is not None:
        tcookies = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tcookies.write(archivo_cookies.read())
        cookies_path = tcookies.name

    if url_video:
        with st.spinner("Descargando video desde el enlace..."):
            tfile_ext = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            try:
                proveedor = descargar_medio(url_video, tfile_ext.name, cookies_path)
                origen_video = tfile_ext.name
                st.toast(f"Descarga completada desde {proveedor} ✅")
            except Exception:
                st.warning(
                    "No se pudo descargar el video de ese enlace. Verifica que "
                    "el archivo/carpeta sea **público** ('cualquiera con el "
                    "enlace puede ver'). Si es YouTube, prueba subiendo un "
                    "cookies.txt (panel de arriba), o usa 'Subir Archivo Local'."
                )


# ──────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────
if origen_video is not None:
    st.success("Metraje recibido. Iniciando motor de visión computacional y cuadrícula analítica...")

    col1, col2 = st.columns([1.4, 1])

    cap = cv2.VideoCapture(origen_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, primer_frame = cap.read() if cap.isOpened() else (False, None)

    # El archivo puede llegar corrupto o incompleto (p. ej. una descarga de
    # YouTube truncada por el bloqueo anti-bot). En ese caso cv2 no logra
    # abrir el video y devuelve dimensiones inválidas (0 o -1), lo que antes
    # rompía np.zeros(). Ahora lo detectamos y avisamos con un mensaje claro.
    if not cap.isOpened() or not ret or width <= 0 or height <= 0:
        cap.release()
        st.error(
            "⚠️ No se pudo leer el video correctamente: el archivo llegó "
            "corrupto o incompleto (frecuente cuando la descarga del enlace "
            "fue bloqueada o interrumpida a mitad de camino). "
            "Vuelve a intentar la descarga (revisa el cookies.txt) o sube "
            "el archivo .mp4 directamente en la pestaña 'Subir Archivo Local'."
        )
        st.stop()

    with st.spinner("Procesando Gemelo Digital y extrayendo analítica espacial..."):
        primer_frame = cv2.cvtColor(primer_frame, cv2.COLOR_BGR2RGB)
        mapa_calor = np.zeros((height, width), dtype=np.float32)

        barra_progreso = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 1

        # --- CAPA DE INFERENCIA (YOLO) ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % salto_frames == 0:
                results = model.track(frame, persist=True, classes=[0], verbose=False)
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        cx, cy = int((x1 + x2) / 2), int(y2)
                        if 0 <= cy < height and 0 <= cx < width:
                            mapa_calor[cy - 15 : cy + 15, cx - 15 : cx + 15] += 1

            frame_count += 1
            if total_frames > 0 and frame_count % 10 == 0:
                barra_progreso.progress(min(frame_count / total_frames, 1.0))

        cap.release()

        # --- CAPA DE PROCESAMIENTO (Alpha Blending) ---
        mapa_suavizado = cv2.GaussianBlur(mapa_calor, (0, 0), sigmaX=21, sigmaY=21)
        mapa_norm = cv2.normalize(mapa_suavizado, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mapa_color = cv2.cvtColor(cv2.applyColorMap(mapa_norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

        mezcla = cv2.addWeighted(primer_frame, 1.0 - opacidad, mapa_color, opacidad, 0)
        mascara_3d = np.repeat((mapa_norm > 5)[:, :, np.newaxis], 3, axis=2)
        frame_final = np.where(mascara_3d, mezcla, primer_frame).astype(np.uint8)

        # --- MOTOR ANALÍTICO (Cuadrícula Espacial) ---
        zonas_x, zonas_y = width // 3, height // 3
        nombres_zonas = [
            "Noroeste", "Norte", "Noreste",
            "Oeste", "Centro", "Este",
            "Suroeste", "Sur", "Sureste",
        ]

        analisis_cuadricula = []
        idx = 0
        for i in range(3):
            for j in range(3):
                y0, y1 = i * zonas_y, (i + 1) * zonas_y
                x0, x1 = j * zonas_x, (j + 1) * zonas_x
                intensidad = np.sum(mapa_calor[y0:y1, x0:x1])
                analisis_cuadricula.append({"Sector": nombres_zonas[idx], "Intensidad": intensidad})
                idx += 1

        df_zonas = pd.DataFrame(analisis_cuadricula)
        max_int = df_zonas["Intensidad"].max()

        def clasificar_zona(intensidad: float) -> str:
            if max_int == 0 or intensidad == 0:
                return "Sin Datos"
            ratio = intensidad / max_int
            if ratio > 0.65:
                return "🔥 Caliente (Alta Permanencia)"
            if ratio > 0.25:
                return "🚶 Transición (Flujo Medio)"
            return "🧊 Fría (Bajo Tráfico)"

        def accion_estrategica(clasificacion: str) -> str:
            if "Caliente" in clasificacion:
                return "⚠️ Ubicar productos de alto margen (Impulso)"
            if "Fría" in clasificacion:
                return "💡 Trasladar productos destino para forzar tráfico"
            return "✅ Zona estable - Mantener layout actual"

        df_zonas["Estado del Flujo"] = df_zonas["Intensidad"].apply(clasificar_zona)
        df_zonas["Directriz de Layout (RA)"] = df_zonas["Estado del Flujo"].apply(accion_estrategica)

        df_final = (
            df_zonas[df_zonas["Estado del Flujo"] != "Sin Datos"]
            .sort_values(by="Intensidad", ascending=False)
            .drop(columns=["Intensidad"])
            .reset_index(drop=True)
        )

    # ──────────────────────────────────────────────────────────────────
    # RENDERIZADO
    # ──────────────────────────────────────────────────────────────────
    with col1:
        st.markdown("#### 🔥 Simulación Híbrida: Termógrafo sobre Gemelo Digital")
        st.image(
            frame_final,
            use_container_width=True,
            caption="El mapa de calor respeta la visibilidad del mobiliario real gracias al Alpha Blending.",
        )

    with col2:
        st.markdown("#### 📊 Decisiones Automatizadas de Merchandising")
        st.dataframe(df_final, use_container_width=True, hide_index=True)
        st.caption(
            "Nota: Las directrices señaladas con ⚠️ y 💡 indican dónde reforzar o "
            "reubicar productos según el flujo peatonal detectado."
        )
