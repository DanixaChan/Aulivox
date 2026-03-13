import asyncio
import edge_tts
import unicodedata

# ==========================
# CONFIGURACIÓN
# ==========================

VOICE = "es-ES-AlvaroNeural"   # Voz masculina española
RATE = "+3%"                   # Velocidad: -20% más lento, +20% más rápido
VOLUME = "+0%"                 # Volumen extra (opcional)

# Otras voces interesantes:
# "es-ES-ElviraNeural"  (España femenina)
# "es-MX-JorgeNeural"   (México masculino)
# "es-MX-DaliaNeural"   (México femenina)


# ==========================
# FUNCIÓN PARA GUARDAR AUDIO
# ==========================

async def save_audio_async(text, filename):
    text = unicodedata.normalize("NFC", text)
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=RATE,
        volume=VOLUME
    )
    await communicate.save(filename)

async def save_audio(text, filename):
    await save_audio_async(text, filename)