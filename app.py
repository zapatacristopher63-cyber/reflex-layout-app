import os
import base64
import tempfile
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import qrcode
import streamlit as st
import yt_dlp
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL Y BRANDING
# ──────────────────────────────────────────────────────────────────────────
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

NARANJA = "#F7941D"
VERDE = "#8DC63F"
FONDO = "#0E1117"
TARJETA = "#161B22"

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

st.markdown(
    f"""
    <style>
        #MainMenu, footer, header {{visibility: hidden;}}

        .stApp {{
            background: radial-gradient(circle at top left, #141a24 0%, {FONDO} 45%);
        }}

        /* Encabezado con logo */
        .rl360-header {{
            display: flex;
            align-items: center;
            gap: 18px;
            padding-bottom: 6px;
        }}
        .rl360-header img {{
            height: 62px;
        }}
        .rl360-header h1 {{
            font-size: 2.1rem;
            margin: 0;
            background: linear-gradient(90deg, {NARANJA}, {VERDE});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .rl360-subtitle {{
            color: #b8bfc9;
            font-size: 1.05rem;
            margin-top: -4px;
        }}

        /* Tarjetas */
        .rl360-card {{
            background: {TARJETA};
            border: 1px solid #262c36;
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
        }}

        div[data-testid="stMetric"] {{
            background: {TARJETA};
            border: 1px solid #262c36;
            border-radius: 12px;
            padding: 10px 14px;
        }}

        .stTabs [data-baseweb="tab"] {{
            font-weight: 600;
        }}

        .stButton>button, .stDownloadButton>button {{
            background: linear-gradient(90deg, {NARANJA}, {VERDE});
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }}

        [data-testid="stSidebar"] {{
            background: #10141c;
            border-right: 1px solid #262c36;
        }}
    </style>

    <div class="rl360-header">
        {f'<img src="data:image/png;base64,{LOGO_B64}">' if LOGO_B64 else ''}
        <div>
            <h1>Reflex Layout 360</h1>
            <div class="rl360-subtitle">Inteligencia Espacial y Gemelos Digitales para Retail</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Sube el metraje de tus cámaras de seguridad. Nuestro motor de IA mapeará "
    "el flujo peatonal y generará decisiones estratégicas de layout al instante."
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
# PANEL LATERAL
# ──────────────────────────────────────────────────────────────────────────
if LOGO_B64:
    st.sidebar.image(LOGO_PATH, use_container_width=True)

st.sidebar.header("⚙️ Motor de Procesamiento")
salto_frames = st.sidebar.slider(
    "Salto de frames", 1, 5, 2, help="Acelera el análisis saltando frames del video."
)
opacidad = st.sidebar.slider("Opacidad del Termógrafo", 0.1, 1.0, 0.55)

# ──────────────────────────────────────────────────────────────────────────
# DESCARGA DE VIDEO DESDE ENLACE (YouTube y similares)
# ──────────────────────────────────────────────────────────────────────────
# YouTube bloquea de forma activa las IPs de datacenter (Streamlit Cloud
# incluida) exigiendo verificación "no soy un robot". No existe una forma
# 100% garantizada de evitarlo desde un servidor en la nube, pero estas
# estrategias reducen mucho la probabilidad de bloqueo:
#   1) Simular el cliente oficial de Android/iOS en vez del navegador web.
#   2) Reintentar automáticamente con distintos "clientes".
#   3) Permitir subir cookies.txt exportadas de una sesión real de YouTube
#      (la solución más efectiva cuando el bloqueo persiste).
CLIENTES_YT = ["android", "ios", "web"]


def descargar_video(url: str, destino: str, cookies_path: Optional[str]) -> None:
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
    url_video = st.text_input("Pega el enlace del video (Ej. YouTube):")

    with st.expander("🍪 Opcional: subir cookies.txt (recomendado si YouTube bloquea la descarga)"):
        st.caption(
            "Exporta las cookies de una sesión activa de YouTube en tu navegador "
            "(extensión 'Get cookies.txt') y súbelas aquí. Esto suele resolver "
            "el bloqueo de verificación anti-bot."
        )
        archivo_cookies = st.file_uploader("cookies.txt", type=["txt"], key="cookies")

    cookies_path = None
    if archivo_cookies is not None:
        tcookies = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tcookies.write(archivo_cookies.read())
        cookies_path = tcookies.name

    if url_video:
        with st.spinner("Procesando enlace externo..."):
            tfile_yt = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            try:
                descargar_video(url_video, tfile_yt.name, cookies_path)
                origen_video = tfile_yt.name
            except Exception:
                st.warning(
                    "YouTube restringió esta descarga automatizada. "
                    "Prueba subiendo un archivo cookies.txt (panel de arriba) "
                    "o usa la pestaña 'Subir Archivo Local' con un video ya descargado."
                )

# ──────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────
if origen_video is not None:
    st.success("Metraje recibido. Iniciando motor de visión computacional y cuadrícula analítica...")

    col1, col2 = st.columns([1.4, 1])

    with st.spinner("Procesando Gemelo Digital y extrayendo analítica espacial..."):
        cap = cv2.VideoCapture(origen_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ret, primer_frame = cap.read()
        if ret:
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
            "Nota: Las directrices señaladas con ⚠️ y 💡 se proyectarán mediante "
            "Realidad Aumentada directamente en los estantes físicos del establecimiento."
        )

    # --- PUENTE DE REALIDAD AUMENTADA (RA) ---
    st.divider()
    st.markdown("### 📱 Despliegue en Espacio Físico (AR)")
    st.write(
        "Escanea el código QR con un dispositivo móvil para proyectar las "
        "directrices de merchandising sobre el entorno físico mediante Realidad Aumentada."
    )

    col_qr, col_info = st.columns([1, 4])

    with col_qr:
        qr = qrcode.QRCode(version=1, box_size=10, border=1)
        qr.add_data("https://ejemplo-webar-layout.com/demo")
        qr.make(fit=True)
        img_qr = qr.make_image(fill_color="white", back_color=FONDO)
        st.image(img_qr.get_image(), use_container_width=True)

    with col_info:
        st.markdown(
            f"""
            <div class="rl360-card">
            <b>Protocolo de Ejecución en Piso:</b><br>
            1. Escanee el código desde su dispositivo móvil.<br>
            2. Enfoque la cámara hacia los estantes de las <i>Zonas Calientes</i>.<br>
            3. Siga la interfaz holográfica para reubicar los productos de alto margen.<br>
            4. Valide la nueva distribución en el sistema.
            </div>
            """,
            unsafe_allow_html=True,
        )
