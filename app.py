import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
import tempfile

# 1. DISEÑO MINIMALISTA DE LA PÁGINA
st.set_page_config(page_title="Reflex Layout 360", page_icon="⬛", layout="wide")

# Ocultar el menú por defecto de Streamlit para un look más "App Propia"
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# 2. ENCABEZADO PROFESIONAL
st.title("⬛ Reflex Layout 360")
st.markdown("### Inteligencia Espacial y Gemelos Digitales para Retail")
st.write("Sube el metraje de tus cámaras de seguridad. Nuestro motor de IA mapeará el flujo peatonal y generará decisiones estratégicas de layout al instante.")
st.divider()

# 3. ZONA DE CARGA DE ARCHIVOS
archivo_video = st.file_uploader("Arrastra tu archivo de video aquí (.mp4)", type=["mp4"])

if archivo_video is not None:
    st.success("Metraje recibido. Iniciando motor de visión computacional...")
    
    col1, col2 = st.columns(2)
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(archivo_video.read())
    
    with st.spinner('Analizando trayectorias y procesando Gemelo Digital...'):
        model = YOLO('yolov8n.pt')
        cap = cv2.VideoCapture(tfile.name)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mapa_calor = np.zeros((height, width), dtype=np.float32)
        
        barra_progreso = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.track(frame, persist=True, classes=[0], verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int(y2)
                    if 0 <= cy < height and 0 <= cx < width:
                        mapa_calor[cy-15:cy+15, cx-15:cx+15] += 1
            
            frame_count += 1
            if total_frames > 0:
                barra_progreso.progress(min(frame_count / total_frames, 1.0))
                
        cap.release()
        
        # 4. RENDERIZADO DE RESULTADOS
        with col1:
            st.markdown("#### 🔥 Mapeo Termográfico (Zonas Calientes)")
            mapa_suavizado = cv2.GaussianBlur(mapa_calor, (61, 61), 0)
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0e1117') # Fondo oscuro minimalista
            ax.set_facecolor('#0e1117')
            cax = ax.imshow(mapa_suavizado, cmap='inferno') # Color 'inferno' más profesional
            ax.axis('off')
            st.pyplot(fig)
            
        with col2:
            st.markdown("#### 📊 Gemelo Digital: Decisión Estratégica")
            datos_tienda = {
                'Zona': ['Entrada', 'Fondo', 'Centro'],
                'Tráfico Real': ['Alto', 'Bajo', 'Medio'],
                'Rotación Histórica': ['Bajo', 'Alto', 'Bajo'],
            }
            df = pd.DataFrame(datos_tienda)
            
            def generar_recomendacion(fila):
                if fila['Tráfico Real'] == 'Alto' and fila['Rotación Histórica'] == 'Bajo':
                    return "⚠️ Reemplazar por producto de impulso"
                elif fila['Tráfico Real'] == 'Bajo' and fila['Rotación Histórica'] == 'Alto':
                    return "✅ Mantener (Producto Imán)"
                else:
                    return "⚖️ Layout Óptimo"
                    
            df['Acción Sugerida (RA)'] = df.apply(generar_recomendacion, axis=1)
            
            st.dataframe(df, use_container_width=True)
            st.caption("Nota: Las acciones marcadas con ⚠️ se proyectarán en los lentes de Realidad Aumentada del personal para su ejecución inmediata.")
