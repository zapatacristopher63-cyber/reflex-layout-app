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
import streamlit.components.v1 as components
import yt_dlp
from ultralytics import YOLO

VIDEO_PRUEBA_URL = "https://drive.google.com/file/d/1NQUjiRgYCCktPAsAS7-HHlEbr5q0z46v/view?usp=drivesdk"

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
LOGO_UNIVERSIDAD_PATH = os.path.join(ASSETS_DIR, "logo_universidad.png")

LETRAS_ZONA = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

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

def _logo_universidad_base64() -> str:
    if not os.path.exists(LOGO_UNIVERSIDAD_PATH):
        return ""
    with open(LOGO_UNIVERSIDAD_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()

LOGO_B64 = _logo_base64()
LOGO_UNI_B64 = _logo_universidad_base64()

NARANJA = "#FF6B1A"
NARANJA_2 = "#FF9142"
VERDE_LIMA = "#8FCB2E"
DORADO = "#FFC93C"
GRADIENTE = f"linear-gradient(115deg, {NARANJA} 0%, {DORADO} 45%, {VERDE_LIMA} 100%)"
GRADIENTE_SUAVE = f"linear-gradient(115deg, {NARANJA_2}22, {DORADO}22, {VERDE_LIMA}22)"

TEMAS = {
    "🌙 Oscuro": {
        "bg_a": "#1c1206", "bg_b": "#0c0904", "blob_1": f"{NARANJA}33", "blob_2": f"{VERDE_LIMA}2b",
        "texto": "#fbf3e7", "subtexto": "#e8c9a0", "card": "rgba(255,255,255,0.06)",
        "card_borde": "rgba(255,177,72,0.28)", "sidebar": "#170f07", "input_bg": "rgba(255,255,255,0.08)",
    },
    "☀️ Claro": {
        "bg_a": "#fff8ec", "bg_b": "#f2f9e3", "blob_1": f"{NARANJA}22", "blob_2": f"{VERDE_LIMA}25",
        "texto": "#2e2410", "subtexto": "#7a5a24", "card": "rgba(255,255,255,0.8)",
        "card_borde": "rgba(143,203,46,0.3)", "sidebar": "#fffaf0", "input_bg": "rgba(143,203,46,0.08)",
    },
}

if "tema" not in st.session_state:
    st.session_state.tema = "🌙 Oscuro"

if LOGO_B64:
    st.sidebar.image(LOGO_PATH, use_container_width=True)

st.sidebar.selectbox("🎨 Apariencia", list(TEMAS.keys()), key="tema")
st.sidebar.header("⚙️ Motor de Procesamiento")
salto_frames = st.sidebar.slider("Salto de frames", 1, 5, 2)
opacidad = st.sidebar.slider("Opacidad del Termógrafo", 0.1, 1.0, 0.55)

T = TEMAS[st.session_state.tema]

CUSTOM_CSS = textwrap.dedent(
    f"""\
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700;900&family=Manrope:wght@400;500;600;700&family=Alex+Brush&display=swap');
    #MainMenu, footer, header {{visibility: hidden;}}
    html, body, [class*="css"] {{ font-family: 'Manrope', sans-serif; }}
    .stApp {{
        background: radial-gradient(circle at 8% 8%, {T['blob_1']} 0%, transparent 40%),
                    radial-gradient(circle at 92% 18%, {T['blob_2']} 0%, transparent 38%),
                    linear-gradient(160deg, {T['bg_a']} 0%, {T['bg_b']} 100%);
        color: {T['texto']};
    }}
    .rl360-header {{ display: flex; align-items: center; gap: 22px; padding: 4px 0 10px 0; }}
    .rl360-header img {{ height: 70px; filter: drop-shadow(0 0 18px {NARANJA}55); }}
    .rl360-header h1 {{
        font-family: 'Unbounded', sans-serif; font-weight: 900; font-size: 2.5rem; margin: 0;
        background: {GRADIENTE}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-transform: uppercase;
    }}
    .rl360-subtitle {{ color: {T['subtexto']}; font-size: 1.05rem; font-weight: 600; margin-top: 2px; }}
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

def reproducir_sonido(tipo: str = "exito") -> None:
    perfiles = {"exito": [523, 659, 784], "click": [740], "aviso": [392, 330]}
    notas_js = ",".join(str(f) for f in perfiles.get(tipo, perfiles["exito"]))
    components.html(
        f"""
        <script>
        (function() {{
            try {{
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const ahora = ctx.currentTime;
                const frecuencias = [{notas_js}];
                frecuencias.forEach((f, i) => {{
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = 'sine'; osc.frequency.value = f;
                    const inicio = ahora + i * 0.09;
                    gain.gain.setValueAtTime(0.0001, inicio);
                    gain.gain.exponentialRampToValueAtTime(0.06, inicio + 0.02);
                    gain.gain.exponentialRampToValueAtTime(0.0001, inicio + 0.28);
                    osc.connect(gain).connect(ctx.destination);
                    osc.start(inicio); osc.stop(inicio + 0.3);
                }});
            }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
    )

st.write("Sube el metraje de tus cámaras de seguridad. Nuestro motor de IA mapeará el flujo peatonal y generará decisiones estratégicas de layout al instante 🛍️🔥")
st.divider()

@st.cache_resource(show_spinner="Cargando motor de visión computacional...")
def cargar_modelo():
    return YOLO("yolov8n.pt")

model = cargar_modelo()

CLIENTES_YT = ["android", "ios", "web"]

def _es_html(resp: requests.Response) -> bool:
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
    token = next((v for k, v in resp.cookies.items() if k.startswith("download_warning")), None)
    if token:
        resp = session.get(base, params={"id": file_id, "confirm": token}, stream=True, timeout=30)
    if _es_html(resp):
        raise ValueError("El archivo de Drive no es público.")
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
        redir = requests.get(url, allow_redirects=True, timeout=30)
        url_directo = redir.url
    if "download=1" not in url_directo:
        url_directo += ("&" if "?" in url_directo else "?") + "download=1"
    resp = requests.get(url_directo, stream=True, timeout=30, allow_redirects=True)
    if _es_html(resp):
        raise ValueError("El enlace de OneDrive no permite descarga directa.")
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
            "extractor_args": {"youtube": {"player_client": [cliente]}},
        }
        if cookies_path:
            ydl_opts["cookiefile"] = cookies_path
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return
        except Exception as e:
            ultimo_error = e
            continue
    raise RuntimeError(str(ultimo_error))

def descargar_medio(url: str, destino: str, cookies_path: Optional[str]) -> str:
    dominio = urlparse(url).netloc.lower()
    if "drive.google.com" in dominio:
        descargar_google_drive(url, destino)
        return "Google Drive"
    if "dropbox.com" in dominio:
        descargar_dropbox(url, destino)
        return "Dropbox"
    if "1drv.ms" in dominio or "onedrive.live.com" in dominio:
        descargar_onedrive(url, destino)
        return "OneDrive"
    descargar_youtube(url, destino, cookies_path)
    return "YouTube / otro"

tab1, tab2 = st.tabs(["📁 Subir Archivo Local", "🔗 Pegar Enlace Público"])
origen_video = None

with tab1:
    archivo_video = st.file_uploader("Arrastra tu archivo de video aquí (.mp4)", type=["mp4"])
    if archivo_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(archivo_video.read())
        origen_video = tfile.name
        reproducir_sonido("click")

with tab2:
    st.caption("Compatibles: **Google Drive**, **Dropbox** y **OneDrive** (recomendado).")
    url_video = st.text_input("Pega el enlace del video:")
    archivo_cookies = st.file_uploader("cookies.txt (Opcional para YouTube)", type=["txt"], key="cookies")
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
                reproducir_sonido("exito")
            except Exception:
                reproducir_sonido("aviso")
                st.warning("No se pudo descargar el video de ese enlace. Asegúrate de que sea público o usa la subida local.")

if origen_video is not None:
    st.success("Metraje recibido. Iniciando motor de visión computacional y cuadrícula analítica...")
    col1, col2 = st.columns([1.4, 1])

    cap = cv2.VideoCapture(origen_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, primer_frame = cap.read() if cap.isOpened() else (False, None)

    if not cap.isOpened() or not ret or width <= 0 or height <= 0:
        cap.release()
        st.error("⚠️ No se pudo leer el video correctamente: el archivo llegó corrupto o incompleto.")
        st.stop()

    with st.spinner("Procesando Gemelo Digital y extrayendo analítica espacial..."):
        primer_frame = cv2.cvtColor(primer_frame, cv2.COLOR_BGR2RGB)
        mapa_calor = np.zeros((height, width), dtype=np.float32)
        barra_progreso = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 1

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

        mapa_suavizado = cv2.GaussianBlur(mapa_calor, (0, 0), sigmaX=21, sigmaY=21)
        mapa_norm = cv2.normalize(mapa_suavizado, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mapa_color = cv2.cvtColor(cv2.applyColorMap(mapa_norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

        mezcla = cv2.addWeighted(primer_frame, 1.0 - opacidad, mapa_color, opacidad, 0)
        mascara_3d = np.repeat((mapa_norm > 5)[:, :, np.newaxis], 3, axis=2)
        frame_final = np.where(mascara_3d, mezcla, primer_frame).astype(np.uint8)

        zonas_x, zonas_y = width // 3, height // 3
        nombres_zonas = ["Noroeste", "Norte", "Noreste", "Oeste", "Centro", "Este", "Suroeste", "Sur", "Sureste"]
        analisis_cuadricula = []
        idx = 0
        for i in range(3):
            for j in range(3):
                y0, y1 = i * zonas_y, (i + 1) * zonas_y
                x0, x1 = j * zonas_x, (j + 1) * zonas_x
                intensidad = np.sum(mapa_calor[y0:y1, x0:x1])
                analisis_cuadricula.append({"Zona": LETRAS_ZONA[idx], "Sector": nombres_zonas[idx], "Intensidad": intensidad})
                idx += 1

        df_zonas = pd.DataFrame(analisis_cuadricula)
        max_int = df_zonas["Intensidad"].max()

        def clasificar_zona(intensidad: float) -> str:
            if max_int == 0 or intensidad == 0: return "Sin Datos"
            ratio = intensidad / max_int
            if ratio > 0.65: return "🔥 Caliente (Alta Permanencia)"
            if ratio > 0.25: return "🚶 Transición (Flujo Medio)"
            return "🧊 Fría (Bajo Tráfico)"

        def accion_estrategica(clasificacion: str) -> str:
            if "Caliente" in clasificacion: return "⚠️ Ubicar productos de alto margen (Impulso)"
            if "Fría" in clasificacion: return "💡 Trasladar productos destino para forzar tráfico"
            return "✅ Zona estable - Mantener layout actual"

        df_zonas["Estado del Flujo"] = df_zonas["Intensidad"].apply(clasificar_zona)
        df_zonas["Directriz de Layout (RA)"] = df_zonas["Estado del Flujo"].apply(accion_estrategica)
        df_zonas["Zona"] = df_zonas["Zona"] + " · " + df_zonas["Sector"]

        df_final = df_zonas[df_zonas["Estado del Flujo"] != "Sin Datos"].sort_values(by="Intensidad", ascending=False).drop(columns=["Intensidad", "Sector"]).reset_index(drop=True)

        frame_etiquetado = frame_final.copy()
        for k in range(1, 3):
            cv2.line(frame_etiquetado, (k * zonas_x, 0), (k * zonas_x, height), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame_etiquetado, (0, k * zonas_y), (width, k * zonas_y), (255, 255, 255), 1, cv2.LINE_AA)

        idx = 0
        for i in range(3):
            for j in range(3):
                letra = LETRAS_ZONA[idx]
                cx = j * zonas_x + zonas_x // 2
                cy = i * zonas_y + max(30, zonas_y // 8)
                cv2.circle(frame_etiquetado, (cx, cy), 20, (15, 15, 15), -1, cv2.LINE_AA)
                cv2.circle(frame_etiquetado, (cx, cy), 20, (255, 201, 60), 2, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(letra, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
                cv2.putText(frame_etiquetado, letra, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                idx += 1

    reproducir_sonido("exito")

    with col1:
        st.markdown("#### 🔥 Simulación Híbrida: Termógrafo sobre Gemelo Digital")
        st.image(frame_etiquetado, use_container_width=True, caption="Cada letra (A-I) ubica una zona de la cuadrícula analítica.")

    with col2:
        st.markdown("#### 📊 Decisiones Automatizadas de Merchandising")
        st.dataframe(df_final, use_container_width=True, hide_index=True)
        st.caption("Nota: Las directrices señaladas con ⚠️ y 💡 indican dónde reforzar o reubicar productos.")

FOOTER_HTML = textwrap.dedent(
    f"""\
    <div style="text-align:center; margin-top:30px; padding:22px; border-top:1.5px solid {T['card_borde']};">
    <div style="font-size:0.9rem; color:{T['subtexto']};">Proyecto académico de Inteligencia Espacial · Reflex Layout 360</div>
    </div>
    """
)
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
