import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
from ultralytics import YOLO
import yt_dlp

# 1. DISEÑO MINIMALISTA DE LA PÁGINA
st.set_page_config(page_title="Reflex Layout 360", page_icon="⬛", layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Cachear el modelo para no saturar la RAM del servidor
@st.cache_resource
def cargar_modelo():
    return YOLO('yolov8n.pt')

model = cargar_modelo()

# 2. PANEL LATERAL DE CONFIGURACIÓN
st.sidebar.header("⚙️ Motor de Procesamiento")
salto_frames = st.sidebar.slider("Salto de frames", 1, 5, 2, help="Acelera el análisis saltando frames del video.")
opacidad = st.sidebar.slider("Opacidad del Termógrafo", 0.1, 1.0, 0.55)

# 3. ENCABEZADO PROFESIONAL
st.title("⬛ Reflex Layout 360")
st.markdown("### Inteligencia Espacial y Gemelos Digitales para Retail")
st.write("Sube el metraje de tus cámaras de seguridad. Nuestro motor de IA mapeará el flujo peatonal y generará decisiones estratégicas de layout al instante.")
st.divider()

# 4. ZONA DE ENTRADA DE VIDEO
tab1, tab2 = st.tabs(["📁 Subir Archivo Local", "🔗 Pegar Enlace Público"])

origen_video = None

with tab1:
    archivo_video = st.file_uploader("Arrastra tu archivo de video aquí (.mp4)", type=["mp4"])
    if archivo_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(archivo_video.read())
        origen_video = tfile.name

with tab2:
    url_video = st.text_input("Pega el enlace del video (Ej. YouTube):")
    if url_video:
        with st.spinner("Extrayendo flujo de video del enlace..."):
            try:
                ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True, 'noplaylist': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url_video, download=False)
                    origen_video = info['url'] # Obtenemos el link directo al stream
            except Exception as e:
                st.error("No se pudo procesar el enlace. Asegúrate de que el video sea público.")

if origen_video is not None:
    st.success("Metraje recibido. Iniciando motor de visión computacional y cuadrícula analítica...")
    
    col1, col2 = st.columns([1.4, 1])
    
    with st.spinner('Procesando Gemelo Digital y extrayendo analítica espacial...'):
        # OpenCV ahora leerá indistintamente el archivo temporal o el link directo
        cap = cv2.VideoCapture(origen_video)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Extraer el primer frame para usarlo como base visual del local
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
                            mapa_calor[cy-15:cy+15, cx-15:cx+15] += 1
            
            frame_count += 1
            if total_frames > 0 and frame_count % 10 == 0:
                barra_progreso.progress(min(frame_count / total_frames, 1.0))
                
        cap.release()
        
      # --- CAPA DE PROCESAMIENTO (Alpha Blending) ---
        mapa_suavizado = cv2.GaussianBlur(mapa_calor, (0, 0), sigmaX=21, sigmaY=21)
        mapa_norm = cv2.normalize(mapa_suavizado, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mapa_color = cv2.applyColorMap(mapa_norm, cv2.COLORMAP_JET)
        mapa_color = cv2.cvtColor(mapa_color, cv2.COLOR_BGR2RGB)
        
        # 1. Crear la mezcla translúcida completa
        mezcla = cv2.addWeighted(primer_frame, 1.0 - opacidad, mapa_color, opacidad, 0)
        
        # 2. Crear una máscara de 3 canales explícita (Evita el error de PyArrow/Streamlit)
        mascara_1d = mapa_norm > 5
        mascara_3d = np.repeat(mascara_1d[:, :, np.newaxis], 3, axis=2)
        
        # 3. Combinar y forzar el formato de memoria correcto (uint8)
        frame_final = np.where(mascara_3d, mezcla, primer_frame).astype(np.uint8)

        # --- MOTOR ANALÍTICO (Cuadrícula Espacial) ---
        zonas_x, zonas_y = width // 3, height // 3
        analisis_cuadricula = []
        nombres_zonas = ["Noroeste", "Norte", "Noreste", "Oeste", "Centro", "Este", "Suroeste", "Sur", "Sureste"]
        
        idx = 0
        for i in range(3):
            for j in range(3):
                y_inicio, y_fin = i * zonas_y, (i + 1) * zonas_y
                x_inicio, x_fin = j * zonas_x, (j + 1) * zonas_x
                intensidad = np.sum(mapa_calor[y_inicio:y_fin, x_inicio:x_fin])
                analisis_cuadricula.append({'Sector': nombres_zonas[idx], 'Intensidad': intensidad})
                idx += 1
                
        df_zonas = pd.DataFrame(analisis_cuadricula)
        max_int = df_zonas['Intensidad'].max()
        
        def clasificar_zona(intensidad):
            if max_int == 0 or intensidad == 0: return "Sin Datos"
            ratio = intensidad / max_int
            if ratio > 0.65: return "🔥 Caliente (Alta Permanencia)"
            elif ratio > 0.25: return "🚶 Transición (Flujo Medio)"
            else: return "🧊 Fría (Bajo Tráfico)"

        def accion_estrategica(clasificacion):
            if "Caliente" in clasificacion: return "⚠️ Ubicar productos de alto margen (Impulso)"
            elif "Fría" in clasificacion: return "💡 Trasladar productos destino para forzar tráfico"
            else: return "✅ Zona estable - Mantener layout actual"

        df_zonas['Estado del Flujo'] = df_zonas['Intensidad'].apply(clasificar_zona)
        df_zonas['Directriz de Layout (RA)'] = df_zonas['Estado del Flujo'].apply(accion_estrategica)
        
        df_final = df_zonas[df_zonas['Estado del Flujo'] != "Sin Datos"].sort_values(by='Intensidad', ascending=False).drop(columns=['Intensidad']).reset_index(drop=True)

        # 5. RENDERIZADO DE LA INTERFAZ
        with col1:
            st.markdown("#### 🔥 Simulación Híbrida: Termógrafo sobre Gemelo Digital")
            st.image(frame_final, use_container_width=True, caption="El mapa de calor respeta la visibilidad del mobiliario real gracias al Alpha Blending.")
            
        with col2:
            st.markdown("#### 📊 Decisiones Automatizadas de Merchandising")
            st.dataframe(df_final, use_container_width=True)
            st.caption("Nota: Las directrices señaladas con ⚠️ y 💡 se proyectarán mediante Realidad Aumentada directamente en los estantes físicos del establecimiento.")
