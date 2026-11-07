import os
import shutil
import subprocess
import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

# --- CONFIGURACIÓN DE FFMEG Y FFPROBE PARA STREAMLIT CLOUD ---
try:
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Asignar el ejecutable de ffmpeg
    AudioSegment.converter = ffmpeg_bin
    
    # Asignar ffmpeg como fallback para ffprobe en pydub
    AudioSegment.ffprobe = ffmpeg_bin
except Exception as e:
    pass
# -------------------------------------------------------------

st.set_page_config(page_title="IA DJ Mix Generator", page_icon="🎧", layout="centered")

st.title("🎧 Generador de Mixes con IA Local")
st.write("Mezcla canciones desde la web o directamente desde tu computadora de forma 100% gratuita.")

# Configuración de opciones en la barra lateral
st.sidebar.header("🎛️ Configuración de Mezcla")
duracion_crossfade_seg = st.sidebar.slider("Tiempo de Crossfade (segundos)", min_value=1, max_value=15, value=6)
umbral_silencio = st.sidebar.slider("Sensibilidad de Silencio (dBFS)", min_value=-60, max_value=-20, value=-40)

# Selección del método de entrada
st.subheader("🎵 Selección de Canciones")
metodo_entrada = st.radio(
    "¿De dónde deseas obtener las canciones?",
    ["Subir desde mi PC (.mp3)", "Escribir nombres para descargar", "Cargar desde canciones.txt"]
)

archivos_para_mezclar = []

# Opción 1: Subir directamente desde la PC
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

def recortar_silencios(audio, silence_threshold=-40):
    """Elimina silencios al inicio y al final de la canción"""
    def trim_leading(sound):
        leading_silence = detect_leading_silence(sound, silence_threshold=silence_threshold)
        return sound[leading_silence:]

    start_trimmed = trim_leading(audio)
    end_trimmed = trim_leading(start_trimmed.reverse()).reverse()
    return end_trimmed

if st.button("🚀 Generar Mix Automático"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Si elegimos descargar de la web
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
                
            progress_bar.progress(int((i + 1) / len(lista_descarga) * 35))

    # Validación de archivos
    if len(archivos_para_mezclar) < 2:
        st.error("Necesitas al menos 2 canciones para realizar una mezcla.")
    else:
        # 1. Análisis de BPM con IA Local
        status_text.text("🎧 Analizando BPMs con IA local (librosa)...")
        analisis = []
        for archivo in archivos_para_mezclar:
            y, sr = librosa.load(archivo)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0]
            analisis.append((archivo, float(tempo)))
        
        # Ordenar canciones por velocidad (BPM)
        analisis.sort(key=lambda x: x[1])
        progress_bar.progress(65)

        # 2. Recorte de silencios y Mezcla
        status_text.text("✂️ Recortando silencios y aplicando crossfade...")
        os.makedirs("output", exist_ok=True)
        archivo_salida = "output/mix_final.mp3"

        audio_primero = AudioSegment.from_file(analisis[0][0])
        mix = recortar_silencios(audio_primero, silence_threshold=umbral_silencio)
        
        ms_crossfade = duracion_crossfade_seg * 1000

        for i in range(1, len(analisis)):
            siguiente_audio = AudioSegment.from_file(analisis[i][0])
            siguiente_recortado = recortar_silencios(siguiente_audio, silence_threshold=umbral_silencio)
            mix = mix.append(siguiente_recortado, crossfade=ms_crossfade)

        mix.export(archivo_salida, format="mp3")
        progress_bar.progress(100)
        status_text.text("✨ ¡Mix completado con éxito!")

        st.success(f"¡Tu mix con {len(archivos_para_mezclar)} canciones está listo!")
        st.audio(archivo_salida)
        
        with open(archivo_salida, "rb") as file:
            st.download_button(
                label="📥 Descargar MP3 Final",
                data=file,
                file_name="mix_final.mp3",
                mime="audio/mp3"
            )
