import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import unicodedata
from pathlib import Path
import hashlib
from tqdm import tqdm

def postprocess_summary(summary):
    # Eliminar tokens <extra_id_X>
    summary = re.sub(r'<extra_id_\d+>', '', summary)
    # Eliminar espacios múltiples
    summary = re.sub(r' +', ' ', summary)
    # Eliminar saltos de línea múltiples
    summary = re.sub(r'\n{3,}', '\n\n', summary)
    # Eliminar fragmentos repetidos (muy simples)
    lines = summary.split('\n')
    seen = set()
    cleaned_lines = []
    for line in lines:
        line_clean = line.strip()
        if line_clean and line_clean not in seen:
            cleaned_lines.append(line_clean)
            seen.add(line_clean)
    summary = '\n'.join(cleaned_lines)
    return summary
def clean_text(text):
    text = text.replace('\t', ' ')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'^ +', '', text, flags=re.MULTILINE)
    text = re.sub(r' +$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*$\n', '', text, flags=re.MULTILINE)
    # Insertar pausa después de títulos/secciones (líneas en mayúsculas o con número y texto)
    def add_pause(match):
        # Usar punto y salto de línea para simular pausa compatible con edge_tts
        return match.group(0) + '.\n'
    # Ejemplo: "2. ALCANCE", "ALCANCE", "INTRODUCCIÓN", "1. OBJETIVO"
    text = re.sub(r'^(\d+\.\s*)?[A-ZÁÉÍÓÚÑ ]{3,}$', add_pause, text, flags=re.MULTILINE)
    return text

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "aulivoxmodel" / "checkpoint-150"
cache = {}
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Modelo en:", MODEL_PATH)
        print("Existe?", MODEL_PATH.exists())
        # Usar ruta absoluta y formato POSIX
        model_path = MODEL_PATH.resolve().as_posix()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            print(f"Error cargando modelo/tokenizer: {e}")
            raise

def chunk_text(text, max_tokens=900, overlap=100):
    # Chunking inteligente: primero por secciones, luego por tokens si es necesario
    # Detectar saltos de sección (capítulos, títulos, saltos dobles de línea)
    section_pattern = r"(?:\n\s*\n|Cap[ií]tulo|Secci[oó]n|\n#|\n##)"
    sections = re.split(section_pattern, text)
    chunks = []
    for section in sections:
        tokens = tokenizer.encode(section)
        if len(tokens) <= max_tokens:
            chunks.append(section)
        else:
            step = max_tokens - overlap
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i:i+max_tokens]
                chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk)
    return [c for c in chunks if c.strip()]

def generate_summary(prompt, max_len, min_len, temperature, num_beams, length_penalty):
    # Progreso de generación de resumen
    print("Generando resumen...")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_len,
        min_length=min_len,
        num_beams=num_beams,
        temperature=temperature,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        early_stopping=True
    )
    for idx in enumerate(outputs):
        tqdm.write(f"Progreso: {idx[0]+1}/{len(outputs)}")

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_text(text, mode="abstract"):
    load_model()
    text = clean_text(text)

    text_hash = hashlib.md5(text.encode()).hexdigest()

    if text_hash not in cache:
        cache[text_hash] = {}

    if mode in cache[text_hash]:
        return cache[text_hash][mode]

    # =========================
    # PROMPTS + PARAMETROS
    # =========================

    if mode == "abstract":
        prompt = f"""
Eres un experto en resúmenes académicos.

Crea un resumen conciso pero completo del siguiente texto.
Preserva conceptos esenciales, definiciones, procesos y comparaciones.
Escribe en un tono académico neutral.
No uses viñetas ni comentarios personales.

Texto:
{text}
"""
        temperature = 0.3
        num_beams = 4
        max_len = 220
        min_len = 100
        length_penalty = 1.0

    elif mode == "structured":
        prompt = f"""
Eres un experto en organización de contenido.

Reescribe el siguiente texto como un resumen claramente estructurado.
Crea breves encabezados para cada sección o tema.
Bajo cada encabezado, escribe un párrafo conciso y bien desarrollado.
No omitas ideas clave.
Mantén todo limpio y bien organizado.

Texto:
{text}
"""
        temperature = 0.4
        num_beams = 4
        max_len = 260
        min_len = 120
        length_penalty = 1.1

    elif mode == "narrative":
        prompt = f"""
Eres un escritor profesional de audiolibros.

Transforma el texto en un resumen narrativo fluido, detallado y atractivo en español.

Instrucciones:
- Escribe de 3 a 5 párrafos bien desarrollados.
- Usa transiciones naturales entre ideas.
- Expande explicaciones cuando sea útil.
- Evita viñetas o encabezados.
- Haz que suene como un capítulo de documental.
- Revisa y corrige todas las tildes y el uso de la letra 'ñ'; el texto final debe tener ortografía, gramática y acentuación impecables.

Texto:
{text}
"""
        temperature = 0.6
        num_beams = 4
        max_len = 350
        min_len = 150
        length_penalty = 1.2

    elif mode == "flashcard":
        prompt = f"""
Resume el siguiente texto de manera muy concisa.
Captura solo las ideas centrales.
Sé directo y compacto.
No agregues explicaciones extra ni estructura compleja.

Texto:
{text}
"""
        temperature = 0.2
        num_beams = 1
        max_len = 120
        min_len = 40
        length_penalty = 0.8

    else:
        prompt = text
        temperature = 0.3
        num_beams = 3
        max_len = 250
        min_len = 80
        length_penalty = 1.0
    # =========================
    chunk_summaries = []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 900:
        chunks = [text]
    else:
        chunks = chunk_text(text)
    print(f"Generando resumen para fragmento de {sum(len(tokenizer.encode(chunk)) for chunk in chunks)} tokens...")
    for chunk in tqdm(chunks, desc="Procesando fragmentos"):
        tqdm.write(f"Progreso: {len(chunk_summaries)}/{len(chunks)} fragmentos resumidos")
        chunk_prompt = f"{prompt}\n\nTexto:\n{chunk}"
        summary_chunk = generate_summary(
            chunk_prompt,
            max_len,
            min_len,
            temperature,
            num_beams,
            length_penalty
        )

        chunk_summaries.append(summary_chunk)

    combined_summary = "\n\n".join(chunk_summaries)
    final_prompt = f"""
Eres un experto en síntesis de textos largos.
Tienes varios resúmenes parciales de un documento extenso.
Tu tarea es unificarlos en un solo resumen coherente, fluido y completo.
Elimina repeticiones, mejora la conexión entre ideas y asegúrate de que el resultado final sea claro y ordenado.
No omitas conceptos importantes.

Resúmenes parciales:
{combined_summary}
"""
    summary = generate_summary(
        final_prompt,
        max_len,
        min_len,
        temperature,
        num_beams,
        length_penalty
    )

    # NORMALIZAR caracteres especiales
    summary = unicodedata.normalize("NFC", summary)
    # Post-procesamiento para limpiar tokens y fragmentos
    summary = postprocess_summary(summary)
    # Guardar en caché
    cache[text_hash][mode] = summary
    return summary
