import os
import shutil
import subprocess
import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

# --- CONFIGURACIÓN DE FFMPEG PARA STREAMLIT CLOUD ---
try:
    import imageio_ffmpeg
    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
except Exception as e:
    pass
# ---------------------------------------------------

st.set_page_config(page_title="IA DJ Mix Generator", page_icon="🎧", layout="centered")

st.title("🎧 Generador de Mixes con IA Local")
st.write("Mezcla canciones desde la web o desde tu equipo con análisis de BPM, normalización y crossfade ajustables.")

# --- BARRA LATERAL: CONFIGURACIÓN PRO ---
st.sidebar.header("🎛️ Configuración del Mix")

duracion_crossfade_seg = st.sidebar.slider("Tiempo de Crossfade (segundos)", min_value=1, max_value=15, value=6)
umbral_silencio = st.sidebar.slider("Sensibilidad de Silencio (dBFS)", min_value=-60, max_value=-20, value=-40)

# FUNCIÓN 1: Normalizador de volumen
normalizar_audio = st.sidebar.checkbox("🔊 Normalizar volumen (mismo nivel para todas)", value=True)

# FUNCIÓN 2: Selector de orden de mezcla
criterio_orden = st.sidebar.selectbox(
    "📊 Orden de las canciones:",
    ["Ascendente por BPM (de menor a mayor)", "Descendente por BPM (de mayor a menor)", "Orden original (sin reordenar)"]
)

# Selección del método de entrada
st.subheader("🎵 Selección de Canciones")
metodo_entrada = st.radio(
    "¿De dónde deseas obtener las canciones?",
    ["Subir desde mi PC (.mp3)", "Escribir nombres para descargar", "Cargar desde canciones.txt"]
)

archivos_para_mezclar = []
lista_descarga = []

# Opción 1: Subir desde la PC
if metodo_entrada == "Subir desde mi PC (.mp3)":
    archivos_subidos = st.file_uploader(
        "Selecciona 2 o más archivos de audio MP3 desde tu equipo:",
        type=["mp3", "wav", "m4a"],
        accept_multiple_files=True
    )
    if archivos_subidos:
        carpeta_subidas = "uploads_locales"
        if os.path.exists(carpeta_subidas):
            shutil.rmtree(carpeta_subidas)
        os.makedirs(carpeta_subidas, exist_ok=True)
        
        for archivo in archivos_subidos:
            ruta_guardado = os.path.join(carpeta_subidas, archivo.name)
            with open(ruta_guardado, "wb") as f:
                f.write(archivo.getbuffer())
            archivos_para_mezclar.append(ruta_guardado)
        st.success(f"Se cargaron {len(archivos_para_mezclar)} archivos locales listos para procesar.")

# Opción 2: Escribir lista para descargar
elif metodo_entrada == "Escribir nombres para descargar":
    texto_canciones = st.text_area("Escribe una canción por línea:", "Daft Punk - One More Time\nStardust - Music Sounds Better With You")
    if texto_canciones:
        lista_descarga = [c.strip() for c in texto_canciones.split("\n") if c.strip()]

# Opción 3: Cargar desde canciones.txt
else:
    if os.path.exists("canciones.txt"):
        with open("canciones.txt", "r", encoding="utf-8") as f:
            lista_descarga = [line.strip() for line in f if line.strip()]
        st.success(f"Se cargaron {len(lista_descarga)} búsquedas desde canciones.txt")
    else:
        st.warning("No se encontró el archivo 'canciones.txt'. Se creará uno por defecto.")
        with open("canciones.txt", "w", encoding="utf-8") as f:
            f.write("Daft Punk - One More Time\nStardust - Music Sounds Better With You")
        lista_descarga = ["Daft Punk - One More Time", "Stardust - Music Sounds Better With You"]

def cargar_audio_seguro(ruta):
    """Carga audio con librosa evitando dependencias de ffprobe"""
    y, sr = librosa.load(ruta, sr=None, mono=False)
    if y.ndim == 2:
        y = y.T
        channels = 2
    else:
        channels = 1
    
    y_int16 = np.clip(y * 32767, -32768, 32767).astype(np.int16)
    return AudioSegment(
        y_int16.tobytes(),
        frame_rate=int(sr),
        sample_width=2,
        channels=channels
    )

def recortar_silencios(audio, silence_threshold=-40):
    """Elimina silencios al inicio y al final"""
    def trim_leading(sound):
        leading_silence = detect_leading_silence(sound, silence_threshold=silence_threshold)
        return sound[leading_silence:]

    start_trimmed = trim_leading(audio)
    end_trimmed = trim_leading(start_trimmed.reverse()).reverse()
    return end_trimmed

def ajustar_volumen(audio, target_dbfs=-14.0):
    """Ajusta el volumen del audio a un nivel estandarizado"""
    change_in_dbfs = target_dbfs - audio.dbfs
    return audio.apply_gain(change_in_dbfs)

if st.button("🚀 Generar Mix Automático"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Descarga web si corresponde
    if metodo_entrada in ["Escribir nombres para descargar", "Cargar desde canciones.txt"]:
        if not lista_descarga:
            st.error("Por favor, ingresa al menos una canción.")
            st.stop()
            
        carpeta_destino = "downloads"
        if os.path.exists(carpeta_destino):
            shutil.rmtree(carpeta_destino)
        os.makedirs(carpeta_destino, exist_ok=True)

        status_text.text("📥 Descargando canciones actuales de la lista...")
        for i, cancion in enumerate(lista_descarga):
            nombre_base = f"song_{i+1:02d}"
            plantilla_salida = os.path.join(carpeta_destino, f"{nombre_base}.%(ext)s")
            
            comando = [
                "yt-dlp",
                f"ytsearch1:{cancion}",
                "-x", "--audio-format", "mp3",
                "-o", plantilla_salida
            ]
            subprocess.run(comando)
            
            archivo_mp3 = os.path.join(carpeta_destino, f"{nombre_base}.mp3")
            if os.path.exists(archivo_mp3):
                archivos_para_mezclar.append(archivo_mp3)
                
            progress_bar.progress(int((i + 1) / len(lista_descarga) * 30))

    if len(archivos_para_mezclar) < 2:
        st.error("Necesitas al menos 2 canciones para realizar una mezcla.")
    else:
        # 1. Análisis de BPM con librosa
        status_text.text("🎧 Analizando tempos (BPM) con librosa...")
        analisis = []
        for archivo in archivos_para_mezclar:
            y, sr = librosa.load(archivo)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0]
            
            nombre_pista = os.path.basename(archivo)
            analisis.append({
                "ruta": archivo,
                "nombre": nombre_pista,
                "bpm": float(tempo)
            })
        
        # FUNCIÓN 2: Aplicar orden seleccionado
        if criterio_orden == "Ascendente por BPM (de menor a mayor)":
            analisis.sort(key=lambda x: x["bpm"])
        elif criterio_orden == "Descendente por BPM (de mayor a menor)":
            analisis.sort(key=lambda x: x["bpm"], reverse=True)
            
        progress_bar.progress(60)

        # 2. Recorte, Normalización y Mezcla
        status_text.text("🎚️ Procesando audios y aplicando crossfade...")
        os.makedirs("output", exist_ok=True)
        archivo_salida = "output/mix_final.mp3"

        segmentos_procesados = []
        duracion_total_ms = 0

        for item in analisis:
            audio = cargar_audio_seguro(item["ruta"])
            audio = recortar_silencios(audio, silence_threshold=umbral_silencio)
            
            # FUNCIÓN 1: Normalizar volumen opcional
            if normalizar_audio:
                audio = ajustar_volumen(audio)
                
            segmentos_procesados.append(audio)

        ms_crossfade = duracion_crossfade_seg * 1000

        # Construcción del Mix
        mix = segmentos_procesados[0]
        duracion_total_ms += len(segmentos_procesados[0])

        for audio_sig in segmentos_procesados[1:]:
            mix = mix.append(audio_sig, crossfade=ms_crossfade)
            duracion_total_ms += len(audio_sig) - ms_crossfade

        mix.export(archivo_salida, format="mp3")
        progress_bar.progress(100)
        status_text.text("✨ ¡Mix completado con éxito!")

        st.success(f"¡Tu mix de {len(analisis)} canciones está listo!")

        # FUNCIÓN 3: Panel de estadísticas y visualización previa
        minutos = int(duracion_total_ms // 60000)
        segundos = int((duracion_total_ms % 60000) // 1000)
        
        st.markdown("### 📊 Detalles del Set")
        st.info(f"⏱️ **Duración total estimada:** {minutos} min {segundos} seg")
        
        with st.expander("📋 Ver orden final de canciones y BPMs", expanded=True):
            for idx, pista in enumerate(analisis, start=1):
                st.write(f"**{idx}.** `{pista['nombre']}` — **{pista['bpm']:.1f} BPM**")

        st.audio(archivo_salida)
        
        with open(archivo_salida, "rb") as file:
            st.download_button(
                label="📥 Descargar MP3 Final",
                data=file,
                file_name="mix_final.mp3",
                mime="audio/mp3"
            )
