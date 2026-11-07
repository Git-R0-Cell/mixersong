import os
import shutil
import subprocess
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

# --- CONFIGURACIÓN DE FFMPEG PARA STREAMLIT CLOUD ---
try:
    import imageio_ffmpeg
    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
except Exception as e:
    pass
# ---------------------------------------------------

st.set_page_config(page_title="IA DJ Mix Generator - Ultimate", page_icon="🎧", layout="centered")

st.title("🎧 Generador de Mixes con IA Local (v3.0 Pro)")
st.write("Crea mezclas profesionales con análisis de BPM, tonalidad armónica, efectos sintéticos y gráficas de onda.")

# --- BARRA LATERAL: CONFIGURACIÓN PRO ---
st.sidebar.header("🎛️ Configuración de Transiciones")

duracion_crossfade_seg = st.sidebar.slider("Tiempo de Crossfade (segundos)", min_value=1, max_value=15, value=6)
umbral_silencio = st.sidebar.slider("Sensibilidad de Silencio (dBFS)", min_value=-60, max_value=-20, value=-40)
normalizar_audio = st.sidebar.checkbox("🔊 Normalizar volumen (mismo dBFS)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("🔀 Orden y Armonía Musical")

criterio_orden = st.sidebar.selectbox(
    "📊 Criterio de orden de canciones:",
    [
        "Ascendente por BPM (de menor a mayor)",
        "Descendente por BPM (de mayor a menor)",
        "🎵 Mezcla Armónica (por Tonalidad / Key)",
        "Orden original (sin reordenar)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("✨ Efectos y Marca de Agua")

# OPCIÓN 1: Efectos sintéticos IA
usar_efectos_ia = st.sidebar.checkbox("🚀 Generar efecto sintético de transición (IA)", value=False)
tipo_efecto = st.sidebar.selectbox(
    "Tipo de efecto:",
    ["Laser Synth Drop", "White Noise Sweep (Viento)", "BPM Beeps (Cuenta Regresiva)"],
    disabled=not usar_efectos_ia
)

# OPCIÓN 2: DJ Drop / Jingle
usar_dj_drop = st.sidebar.checkbox("🗣️ Insertar DJ Drop / Jingle personal", value=False)
archivo_jingle = None
if usar_dj_drop:
    archivo_jingle = st.sidebar.file_uploader("Sube tu voz o jingle (.mp3, .wav)", type=["mp3", "wav"])

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
        st.success(f"Se cargaron {len(archivos_para_mezclar)} archivos locales listos.")

# Opción 2: Escribir lista para descargar
elif metodo_entrada == "Escribir nombres para descargar":
    texto_canciones = st.text_area("Escribe una canción por línea:", "Robbie Williams - Feel\Eros Ramazzotti, Tina Turner - Cosas de la Vida")
    if texto_canciones:
        lista_descarga = [c.strip() for c in texto_canciones.splitlines() if c.strip()]

# Opción 3: Cargar desde canciones.txt
else:
    if os.path.exists("canciones.txt"):
        with open("canciones.txt", "r", encoding="utf-8") as f:
            lista_descarga = [line.strip() for line in f if line.strip()]
        st.success(f"Se cargaron {len(lista_descarga)} búsquedas desde canciones.txt")
    else:
        st.warning("No se encontró 'canciones.txt'. Se creará uno por defecto.")
        with open("canciones.txt", "w", encoding="utf-8") as f:
            f.write("Daft Punk - One More Time\nStardust - Music Sounds Better With You")
        lista_descarga = ["Daft Punk - One More Time", "Stardust - Music Sounds Better With You"]

# --- FUNCIONES TÉCNICAS DE AUDIO ---

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
    """Ajusta el volumen de forma segura evitando errores de dbfs en audio sintético"""
    try:
        if hasattr(audio, 'dbfs') and audio.dbfs is not None and not np.isnan(audio.dbfs):
            change_in_dbfs = target_dbfs - audio.dbfs
            return audio.apply_gain(change_in_dbfs)
    except Exception:
        pass
    
    # Si falla el cálculo estándar de pydub, retornamos el audio sin modificar gain para evitar que rompa la app
    return audio

def detectar_tonalidad(y, sr):
    """Detecta la clave/tonalidad armónica de una pista (C, C#, D, etc.)"""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vals = np.sum(chroma, axis=1)
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    tonalidad_idx = np.argmax(chroma_vals)
    return notas[tonalidad_idx]

def generar_efecto_sintetizado(estilo="Laser Synth Drop", duracion_ms=1200, sr=44100):
    """Sintetiza efectos de audio con matemáticas en NumPy"""
    num_samples = int(sr * (duracion_ms / 1000.0))
    t = np.linspace(0, duracion_ms / 1000.0, num_samples)
    
    if estilo == "Laser Synth Drop":
        frecuencia = 1200 * np.exp(-6 * t)
        audio_np = 0.4 * np.sin(2 * np.pi * frecuencia * t * 8)
    elif estilo == "White Noise Sweep (Viento)":
        ruido = np.random.uniform(-0.5, 0.5, num_samples)
        envelope = np.sin(np.pi * t / (duracion_ms / 1000.0))
        audio_np = ruido * envelope * 0.4
    else:  # BPM Beeps
        audio_np = np.zeros(num_samples)
        pasos = 4
        sub_len = num_samples // pasos
        for b in range(pasos):
            t_sub = np.linspace(0, 0.08, int(sr * 0.08))
            beep = 0.3 * np.sin(2 * np.pi * 880 * t_sub)
            inicio = b * sub_len
            fin = inicio + len(beep)
            if fin < num_samples:
                audio_np[inicio:fin] = beep
                
    ventana = np.hanning(len(audio_np))
    audio_np = audio_np * ventana
    y_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
    return AudioSegment(y_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)

def generar_grafico_forma_onda(ruta_mp3):
    """Genera una imagen con la forma de onda del mix procesado"""
    y, sr = librosa.load(ruta_mp3, sr=22050)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#1DB954", alpha=0.8)
    ax.set_title("Onda de Audio del Mix Resultante", fontsize=12, color="white")
    ax.set_xlabel("Tiempo (segundos)", fontsize=10, color="gray")
    ax.set_ylabel("Amplitud", fontsize=10, color="gray")
    ax.tick_params(colors='gray')
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")
    plt.tight_layout()
    return fig

# --- PROCESAMIENTO DEL MIX ---

if st.button("🚀 Generar Mix Automático (v3.0 Pro)"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    if metodo_entrada in ["Escribir nombres para descargar", "Cargar desde canciones.txt"]:
        if not lista_descarga:
            st.error("Por favor, ingresa al menos una canción.")
            st.stop()
            
        carpeta_destino = "downloads"
        if os.path.exists(carpeta_destino):
            shutil.rmtree(carpeta_destino)
        os.makedirs(carpeta_destino, exist_ok=True)

        status_text.text("📥 Descargando canciones actuales...")
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
                
            progress_bar.progress(int((i + 1) / len(lista_descarga) * 25))

    if len(archivos_para_mezclar) < 2:
        st.error("Necesitas al menos 2 canciones para realizar una mezcla.")
    else:
        # 1. Análisis de BPM y Tonalidad (Key)
        status_text.text("🎼 Analizando BPMs y Tonalidad Armónica...")
        analisis = []
        for archivo in archivos_para_mezclar:
            y, sr = librosa.load(archivo)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0]
            
            key_detectada = detectar_tonalidad(y, sr)
            nombre_pista = os.path.basename(archivo)
            
            analisis.append({
                "ruta": archivo,
                "nombre": nombre_pista,
                "bpm": float(tempo),
                "key": key_detectada
            })
        
        # Aplicar orden seleccionado
        if criterio_orden == "Ascendente por BPM (de menor a mayor)":
            analisis.sort(key=lambda x: x["bpm"])
        elif criterio_orden == "Descendente por BPM (de mayor a menor)":
            analisis.sort(key=lambda x: x["bpm"], reverse=True)
        elif criterio_orden == "🎵 Mezcla Armónica (por Tonalidad / Key)":
            analisis.sort(key=lambda x: x["key"])
            
        progress_bar.progress(50)

        # 2. Recorte y Procesamiento de Pistas
        status_text.text("🎚️ Aplicando Gain Matching y procesando silencios...")
        os.makedirs("output", exist_ok=True)
        archivo_salida = "output/mix_final.mp3"

        segmentos_procesados = []
        for item in analisis:
            audio = cargar_audio_seguro(item["ruta"])
            audio = recortar_silencios(audio, silence_threshold=umbral_silencio)
            
            if normalizar_audio:
                audio = ajustar_volumen(audio)
                
            segmentos_procesados.append(audio)

        ms_crossfade = duracion_crossfade_seg * 1000

        # Cargar DJ Drop si fue subido
        jingle_audio = None
        if usar_dj_drop and archivo_jingle is not None:
            ruta_jingle_temp = "temp_jingle.mp3"
            with open(ruta_jingle_temp, "wb") as f:
                f.write(archivo_jingle.getbuffer())
            jingle_audio = cargar_audio_seguro(ruta_jingle_temp)
            if normalizar_audio:
                jingle_audio = ajustar_volumen(jingle_audio)

        # Construcción y Ensamblaje del Mix
        status_text.text("🔀 Mezclando pistas con efectos y transiciones...")
        mix = segmentos_procesados[0]
        duracion_total_ms = len(segmentos_procesados[0])

        for idx in range(1, len(segmentos_procesados)):
            siguiente_pista = segmentos_procesados[idx]
            
            # Si hay efectos sintéticos o DJ drop, los overlay en la transición
            if usar_efectos_ia or jingle_audio:
                efecto_transicion = None
                
                if usar_efectos_ia:
                    efecto_transicion = generar_efecto_sintetizado(estilo=tipo_efecto)
                elif jingle_audio:
                    efecto_transicion = jingle_audio

                if efecto_transicion:
                    posicion_overlay = max(0, len(mix) - ms_crossfade)
                    mix = mix.overlay(efecto_transicion, position=posicion_overlay)

            mix = mix.append(siguiente_pista, crossfade=ms_crossfade)
            duracion_total_ms += len(siguiente_pista) - ms_crossfade

        mix.export(archivo_salida, format="mp3")
        progress_bar.progress(85)

        # 3. Renderear Gráfica de Onda
        status_text.text("📈 Generando previsualización de forma de onda...")
        figura_onda = generar_grafico_forma_onda(archivo_salida)
        progress_bar.progress(100)
        status_text.text("✨ ¡Mix completado con éxito!")

        st.success(f"¡Tu mix de {len(analisis)} canciones está listo!")

        # Muestra de Duración y Lista Armónica
        minutos = int(duracion_total_ms // 60000)
        segundos = int((duracion_total_ms % 60000) // 1000)
        
        st.markdown("### 📊 Detalles del Set Profesionales")
        st.info(f"⏱️ **Duración total estimada:** {minutos} min {segundos} seg")
        
        st.pyplot(figura_onda)
        
        with st.expander("📋 Ver orden final, BPMs y Tonalidad (Key)", expanded=True):
            for idx_p, pista in enumerate(analisis, start=1):
                st.write(f"**{idx_p}.** `{pista['nombre']}` — **{pista['bpm']:.1f} BPM** — Key: **{pista['key']}**")

        st.audio(archivo_salida)
        
        with open(archivo_salida, "rb") as file:
            st.download_button(
                label="📥 Descargar MP3 Final",
                data=file,
                file_name="mix_final.mp3",
                mime="audio/mp3"
            )
