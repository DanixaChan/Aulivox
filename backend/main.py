from PIL import ImageOps
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
from fastapi import UploadFile, File
from pdf_parser.import_utils import extract_text_from_word, extract_text_from_pdf, remove_toc_tables
from pdf2image import convert_from_path
import re
from fastapi import FastAPI, Request
from pydantic import BaseModel
from tts.textspeech import save_audio
from summarizer.summarizer import clean_text, summarize_text
import os
import unicodedata
import asyncio
from tqdm import tqdm

# Diccionario de abreviaciones comunes
ABBREVIATIONS = {
    "máx": "máximo",
    "min": "minuto",
    "seg": "segundo",
    "hr": "hora",
    "hrs": "horas",
    "aprox": "aproximadamente",
    "doc": "documento",
    "pág": "página",
    "vol": "volumen",
    "etc": "etcétera",
    "prof": "profesor",
    "sr": "señor",
    "sra": "señora",
    "dr": "doctor",
    "dra": "doctora",
    "lic": "licenciado",
    "tel": "teléfono",
    "dir": "dirección",
    "ed": "edición",
    "adm": "administración",
    "dep": "departamento",
    "univ": "universidad",
    "av": "avenida",
    "dpto": "departamento",
    "mts": "metros",
    "km": "kilómetro",
    "gral": "general",
    "art": "artículo",
    "fig": "figura",
    "num": "número",
    "resp": "respuesta",
    "exp": "explicación",
    "obs": "observación"
    # Puedes agregar más según tus necesidades
}

def detect_best_psm(img):
    """
    Detecta si la imagen tiene una o múltiples columnas/bloques
    y elige el PSM más adecuado.
    """
    import cv2
    import numpy as np

    arr = np.array(img)
    h, w = arr.shape[:2]

    # Proyección horizontal: detecta si hay separación de columnas
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_projection = np.sum(binary, axis=0)
    
    # Busca valles profundos en la proyección (= separación de columnas)
    threshold_valley = col_projection.max() * 0.05
    valleys = np.where(col_projection < threshold_valley)[0]
    
    # Si hay un valle sostenido en el centro → 2 columnas → PSM 3
    mid = w // 2
    center_zone = valleys[(valleys > mid * 0.6) & (valleys < mid * 1.4)]
    
    if len(center_zone) > w * 0.05:  # valle de más del 5% del ancho
        return 3  # Auto (múltiples columnas)
    
    # Detectar si es una sola línea
    row_projection = np.sum(binary, axis=1)
    non_empty_rows = np.sum(row_projection > row_projection.max() * 0.1)
    
    if non_empty_rows < h * 0.15:  # muy pocas filas con texto
        return 7  # Línea única
    
    return 4  # Bloque de texto (default para documentos)

def validate_with_dictionary(text):
    """
    Usa pyspellchecker para corregir palabras con baja confianza en OCR.
    pip install pyspellchecker
    """
    from spellchecker import SpellChecker
    spell = SpellChecker(language='es')

    words = text.split()
    corrected = []
    for word in words:
        # Solo corregir palabras puramente alfabéticas de más de 3 letras
        clean = re.sub(r'[^a-záéíóúñüA-ZÁÉÍÓÚÑÜ]', '', word)
        if len(clean) > 3 and clean.lower() in spell.unknown([clean]):
            suggestion = spell.correction(clean.lower())
            if suggestion and suggestion != clean.lower():
                # Preservar mayúscula inicial si la tenía
                if word[0].isupper():
                    suggestion = suggestion.capitalize()
                corrected.append(suggestion)
            else:
                corrected.append(word)
        else:
            corrected.append(word)
    return ' '.join(corrected)

def smart_ocr_single_column(img, psm_mode=4):
    """
    Optimizado para documentos de una sola columna.
    PSM 4 = columna única de texto de tamaño variable
    PSM 6 = bloque uniforme (útil si el PDF tiene márgenes limpios)
    """
    import pytesseract
    import numpy as np

    def run_ocr(psm):
        # NUEVO: parámetros internos de Tesseract
        custom_config = (
            f'--psm {psm} --oem 1 '
            # Umbral de confianza interno más bajo
            '-c tessedit_char_blacklist=|}{[]<> '   # excluye símbolos inútiles
            '-c preserve_interword_spaces=1 '        # mantiene espacios entre palabras
            '-c textord_heavy_nr=0 '                 # no ignora texto "raro"
            '-c edges_max_children_per_outline=40 '  # mejor para serifas complejas
        )
        return custom_config
    
    tsv_data = pytesseract.image_to_data(
        img, lang="spa",
        output_type=pytesseract.Output.DATAFRAME,
        config=run_ocr(psm_mode)
    )
    df = tsv_data.copy()

    # Filtros básicos
    df = df[df["conf"] >= 30]
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip() != ""]

    if df.empty:
        return ""

    df["center_y"] = df["top"] + df["height"] / 2

    # Filtrar texto vertical (alto >> ancho)
    df = df[df["height"] < df["width"] * 3]

    # --- ORDENAR ESTRICTAMENTE por bloque y luego por Y ---
    # block_num de Tesseract respeta el orden de lectura natural
    df = df.sort_values(["block_num", "par_num", "line_num", "word_num"])

    # Reconstruir líneas respetando el orden de Tesseract
    lines = []
    current_line_words = []
    last_line_key = None

    for _, row in df.iterrows():
        # Clave única por línea según Tesseract
        line_key = (row["block_num"], row["par_num"], row["line_num"])
        
        if last_line_key is None:
            last_line_key = line_key

        if line_key == last_line_key:
            current_line_words.append(str(row["text"]))
        else:
            if current_line_words:
                lines.append(" ".join(current_line_words))
            current_line_words = [str(row["text"])]
            last_line_key = line_key

    if current_line_words:
        lines.append(" ".join(current_line_words))

    return "\n".join(lines)

def preprocess_single_column(pil_img):
    import cv2
    import numpy as np

    img = np.array(pil_img)
    h, w = img.shape[:2]

    # Escalar solo si realmente es pequeña
    if w < 1500:
        scale = 3
    elif w < 2500:
        scale = 2
    else:
        scale = 1  # ← NUEVO: no escalar si ya tiene buena resolución

    if scale > 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    h2, w2 = img.shape[:2]
    ksize = (w2 // 8) | 1  # ~12% del ancho, siempre impar ← CLAVE
    ksize = max(51, min(ksize, 301))  # clamp entre 51 y 301

    # --- Normalización de fondo (Background Subtraction) ---
    background = cv2.GaussianBlur(img, (ksize, ksize), 0)
    img = cv2.divide(img, background, scale=255)

    # --- Unsharp mask suave ---
    blurred = cv2.GaussianBlur(img, (0, 0), 1.5)  # era 2, reducir
    img = cv2.addWeighted(img, 1.3, blurred, -0.3, 0)  # era 1.4/-0.4

    # Denoise: reducir h de 10 a 7 para preservar trazos finos
    img = cv2.fastNlMeansDenoising(
            img,
            h=12,                  # era 6, subir para atacar ruido de fondo
            templateWindowSize=9,  # era 7, ventana más grande
            searchWindowSize=21
        )

    # CLAHE: tileGridSize más grande para fondo gris uniforme
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    img = clahe.apply(img)

    # --- Threshold: ahora con fondo normalizado, Otsu funciona mejor ---
    # Primero intentar Otsu global (más limpio si el fondo ya está normalizado)
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ← CAMBIO PRINCIPAL: blockSize y C ajustados para texto con serifa sobre fondo gris
    adaptive = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=41,
        C=12
    )

    # NUEVO: combinar ambos — Otsu limpia el fondo, adaptativo recupera texto fino
    binary = cv2.bitwise_and(otsu, adaptive)

    # Morfología: erosión mínima para separar letras pegadas
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(clean)

def smart_ocr(img, column_gap_threshold=None, mode="auto", psm_mode=6):
    """
    mode: 'auto' (columnas, clustering) o 'line' (línea por línea)
    """
    """
    Ignora textos girados a 90° (verticales), solo procesa líneas horizontales.
    """
    import pytesseract
    import pandas as pd
    import numpy as np
    from io import StringIO

    custom_config = f'--psm {psm_mode}'
    tsv_data = pytesseract.image_to_data(img, lang="spa", output_type=pytesseract.Output.DATAFRAME, config=custom_config)
    df = tsv_data.copy()

    # Filtrar solo texto real con confianza alta (descarta ruido de íconos)
    df = df[df["conf"] >= 30]
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip() != ""]

    if df.empty:
        return ""

    df["center_x"] = df["left"] + df["width"] / 2
    df["center_y"] = df["top"] + df["height"] / 2

    # Filtrar líneas verticales (textos girados a 90°)
    # Si el alto del bounding box es mucho mayor que el ancho, se considera vertical
    df = df[df["height"] < df["width"] * 2]

    if mode == "line":
        # Agrupar solo por líneas horizontales (ignora columnas)
        row_gap_threshold = img.size[1] * 0.04  # 4% del alto
        df = df.sort_values("center_y")
        lines = []
        current_line = []
        last_y = None
        for _, row in df.iterrows():
            y = row["center_y"]
            text = str(row["text"])
            if last_y is None:
                current_line.append((row["left"], text))
                last_y = y
                continue
            if abs(y - last_y) < row_gap_threshold:
                current_line.append((row["left"], text))
            else:
                current_line.sort(key=lambda t: t[0])
                lines.append(" ".join(w for _, w in current_line))
                current_line = [(row["left"], text)]
            last_y = y
        if current_line:
            current_line.sort(key=lambda t: t[0])
            lines.append(" ".join(w for _, w in current_line))
        return "\n".join(lines)
    # Modo automático (columnas)
    img_width = img.size[0]
    if column_gap_threshold is None:
        column_gap_threshold = img_width * 0.08  # 8% del ancho
    # Clustering de columnas por center_x
    sorted_x = sorted(df["center_x"].tolist())
    clusters = []
    cluster = [sorted_x[0]]
    for x in sorted_x[1:]:
        if x - cluster[-1] > column_gap_threshold:
            clusters.append(cluster)
            cluster = [x]
        else:
            cluster.append(x)
    clusters.append(cluster)
    column_centers = [np.mean(c) for c in clusters]
    def assign_column(cx):
        return min(range(len(column_centers)), key=lambda i: abs(column_centers[i] - cx))
    df["col_idx"] = df["center_x"].apply(assign_column)
    # Clustering de filas por center_y DENTRO de cada columna
    row_gap_threshold = img.size[1] * 0.04  # 4% del alto
    all_texts = []
    for col_idx in sorted(df["col_idx"].unique()):
        col_df = df[df["col_idx"] == col_idx].sort_values("center_y")
        # Agrupar en líneas por proximidad vertical
        lines = []
        current_line = []
        last_y = None
        for _, row in col_df.iterrows():
            y = row["center_y"]
            text = str(row["text"])
            if last_y is None:
                current_line.append((row["left"], text))
                last_y = y
                continue
            if abs(y - last_y) < row_gap_threshold:
                current_line.append((row["left"], text))
            else:
                # Ordenar palabras dentro de la línea por posición X
                current_line.sort(key=lambda t: t[0])
                lines.append(" ".join(w for _, w in current_line))
                current_line = [(row["left"], text)]
            last_y = y
        if current_line:
            current_line.sort(key=lambda t: t[0])
            lines.append(" ".join(w for _, w in current_line))
        all_texts.append("\n".join(lines))
    return "\n\n".join(all_texts)

# Expande abreviaciones en el texto
def expand_abbreviations(text):
    # Reemplazo especial para 'N°' por 'Número'
    text = re.sub(r'N°', 'Número', text, flags=re.IGNORECASE)
    def repl(match):
        abbr = match.group(1).lower()
        return ABBREVIATIONS.get(abbr, match.group(0))
    pattern = r'\b(' + '|'.join(re.escape(k) for k in ABBREVIATIONS.keys()) + r')(\.|\b)'
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

# Expande palabras con barra (/) como "encargado/a" -> "encargado o encargada"
def expand_slash_words(text):
    def repl(match):
        w1 = match.group(1)
        w2 = match.group(2)
        # Caso especial: un/a -> un o una
        if w1 == "un" and w2 == "a":
            return "un o una"
        if w1 == "una" and w2 == "o":
            return "una o uno"
        if w1 == "unos" and w2 == "as":
            return "unos o unas"
        if w1 == "unas" and w2 == "os":
            return "unas o unos"
        # Caso artículos genéricos
        if w1 in ["un", "una", "uno", "unos", "unas"] and w2 in ["un", "una", "uno", "unos", "unas"]:
            return f"{w1} o {w2}"
        # Caso palabras terminadas en o/a, a/o, os/as, as/os
        if w1.endswith("o") and w2 == "a":
            return f"{w1} o {w1[:-1]}a"
        if w1.endswith("a") and w2 == "o":
            return f"{w1} o {w1[:-1]}o"
        if w1.endswith("os") and w2 == "as":
            return f"{w1} o {w1[:-2]}as"
        if w1.endswith("as") and w2 == "os":
            return f"{w1} o {w1[:-2]}os"
        # Caso palabras terminadas en e/es
        if w1.endswith("e") and w2 == "es":
            return f"{w1} o {w1}es"
        if w1.endswith("es") and w2 == "e":
            return f"{w1} o {w1[:-2]}e"
        # Caso genérico: profesor/a -> profesor o profesora
        if len(w2) == 1 and w1.endswith("o") and w2 == "a":
            return f"{w1} o {w1[:-1]}a"
        if len(w2) == 1 and w1.endswith("a") and w2 == "o":
            return f"{w1} o {w1[:-1]}o"
        # Si w2 es una palabra completa, simplemente unir
        return f"{w1} o {w2}"
    # Manejo de (s): la(s) -> la o las, profesional(es) -> profesional o profesionales
    def paren_repl(match):
        base = match.group(1)
        plural = base + "s"
        return f"{base} o {plural}"
    text = re.sub(r'\b([a-zA-ZñÑáéíóúÁÉÍÓÚ]+)\(s\)', paren_repl, text)
    text = re.sub(r'\b([a-zA-ZñÑáéíóúÁÉÍÓÚ]+)\(es\)', lambda m: f"{m.group(1)} o {m.group(1)}es", text)
    pattern = r'\b([a-zA-ZñÑáéíóúÁÉÍÓÚ]+)/(a|o|os|as|e|es|un|una|uno|unas|unos|[a-zA-ZñÑáéíóúÁÉÍÓÚ]+)\b'
    return re.sub(pattern, repl, text)

# Elimina bloques de índice o tabla de contenidos
# def remove_index_sections(text):
#     # Busca patrones típicos de índice o tabla de contenidos SOLO al inicio del texto
#     # Elimina solo el primer bloque que parezca índice, pero nunca borra todo el texto útil
#     pat_index = r'(?is)^.*?\b(índice|tabla de contenidos|contenido|contents|contenidos)\b.*?(\n\s*\n)'
#     match = re.match(pat_index, text)
#     if match:
#         # Solo elimina el bloque de índice si hay contenido después del doble salto de línea
#         idx = match.end(2)
#         text = text[idx:]
#     return text

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Aulivox backend funcionando"}

# Endpoint para importar archivo y dictar automáticamente

from fastapi import Form

@app.post("/import-dictate")
async def import_and_dictate(file: UploadFile = File(...), ocr_mode: str = Form("line"), deskew_enabled: bool = Form(False), psm_mode: int = Form(4)):
    filename = file.filename.lower()
    temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file.filename))
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    if filename.endswith(".docx"):
        pages = extract_text_from_word(temp_path)
    elif filename.endswith(".pdf"):
        pages = extract_text_from_pdf(temp_path)
        # Si no se extrajo texto, intentar OCR página por página
        if not any(p.strip() for p in pages):
            try:
                images = convert_from_path(temp_path)
                ocr_pages = []
                for idx, img in enumerate(images):
                    img = img.convert("L")
                    if deskew_enabled:
                        img = deskew_pil_image(img)
                    img = preprocess_single_column(img)           # 1. Preprocesamiento visual
                    psm = detect_best_psm(img)                    # 2. PSM dinámico
                    text = smart_ocr_single_column(img, psm)      # 3. OCR con config fina
                    text = expand_abbreviations(text)             # 4. Tu lógica existente
                    text = expand_slash_words(text)
                    # text = validate_with_dictionary(text)       # 5. Opcional, más lento
                    cleaned_text = clean_text(text)
                    ocr_pages.append(cleaned_text)
                pages = ocr_pages
            except Exception as e:
                os.remove(temp_path)
                return {"error": f"No se pudo extraer texto del PDF (OCR): {str(e)}"}
    else:
        os.remove(temp_path)
        return {"error": "Formato no soportado. Solo .docx y .pdf"}
    os.remove(temp_path)
    pages = remove_toc_tables(pages)
    texto_final = '\n\n'.join(pages)
    # Expande abreviaciones y palabras con barra
    texto_final = expand_abbreviations(texto_final)
    texto_final = expand_slash_words(texto_final)
    cleaned_text = clean_text(texto_final)
    if not cleaned_text.strip():
        return {"error": "El texto a dictar está vacío después del preprocesamiento."}
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "output.wav"))
    await save_audio(cleaned_text, output_path)
    return {"audio_file": output_path, "text": cleaned_text}

# --- Deskew utility ---
def deskew_pil_image(pil_img):
    # Convierte PIL.Image a array de OpenCV
    img = np.array(pil_img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Binariza
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = 255 - img_bin
    coords = np.column_stack(np.where(img_bin > 0))
    if coords.shape[0] < 10:
        return pil_img  # No hay suficiente texto para deskew
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)

# Endpoint para dictar texto extraído de una imagen
from fastapi import Form

@app.post("/dictate-image")
async def dictate_image(file: UploadFile = File(...), columns: int = Form(1)):
    # Guardar imagen temporalmente
    filename = file.filename.lower()
    temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file.filename))
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        img = Image.open(temp_path)
        img = img.convert("L")
        img = img.resize((img.width*2, img.height*2))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        text = smart_ocr(img)  # detección automática, sin loop manual
    except Exception as e:
        os.remove(temp_path)
        return {"error": f"No se pudo extraer texto de la imagen: {str(e)}"}
    os.remove(temp_path)
    text = expand_abbreviations(text)
    text = expand_slash_words(text)
    cleaned_text = clean_text(text)
    if not cleaned_text.strip():
        return {"error": "No se encontró texto en la imagen para dictar."}
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "output.wav"))
    await save_audio(cleaned_text, output_path)
    return {"audio_file": output_path, "text": cleaned_text}

# Endpoint para importar texto desde Word o PDF

@app.post("/import-text")
async def import_text(file: UploadFile = File(...), ocr_mode: str = Form("auto"), deskew_enabled: bool = Form(True), psm_mode: int = Form(6)):
    filename = file.filename.lower()
    # Guardar archivo temporalmente
    temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file.filename))
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    # Extraer texto según tipo
    if filename.endswith(".docx"):
        pages = extract_text_from_word(temp_path)
    elif filename.endswith(".pdf"):
        pages = extract_text_from_pdf(temp_path)
        # Si no se extrajo texto, intentar OCR página por página
        if not any(p.strip() for p in pages):
            try:
                images = convert_from_path(temp_path)
                ocr_pages = []
                for idx, img in enumerate(images):
                    img = img.convert("L")
                    img = img.resize((img.width*2, img.height*2))
                    if deskew_enabled:
                        img = deskew_pil_image(img)
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(2.0)
                    if idx == 0:
                        img.save(os.path.abspath(os.path.join(os.path.dirname(__file__), "processed_page1.png")))
                    text = smart_ocr(img, mode=ocr_mode, psm_mode=psm_mode)
                    ocr_pages.append(text)
                pages = ocr_pages
            except Exception as e:
                os.remove(temp_path)
                return {"error": f"No se pudo extraer texto del PDF (OCR): {str(e)}"}
    else:
        os.remove(temp_path)
        return {"error": "Formato no soportado. Solo .docx y .pdf"}
    os.remove(temp_path)
    # Omitir páginas de índice/contenidos
    # pages = skip_index_pages(pages)
    texto_final = '\n\n'.join(pages)
    return {"text": texto_final}

@app.post("/dictate")
def dictate_text(text: str):
    # Expande abreviaciones y palabras con barra
    text = expand_abbreviations(text)
    text = expand_slash_words(text)
    cleaned_text = clean_text(text)
    if not cleaned_text.strip():
        return {"error": "El texto a dictar está vacío después del preprocesamiento."}
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "output.wav"))
    asyncio.create_task(save_audio(cleaned_text, output_path))
    return {"audio_file": output_path}

@app.post("/import-summarize")
async def process_text(file: UploadFile = File(...), mode: str = Form("abstract"), ocr_mode: str = Form("line"), deskew_enabled: bool = Form(False), psm_mode: int = Form(4)):
    filename = file.filename.lower()
    temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file.filename))
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    if filename.endswith(".docx"):
        pages = extract_text_from_word(temp_path)
    elif filename.endswith(".pdf"):
        pages = extract_text_from_pdf(temp_path)
        # Si no se extrajo texto, intentar OCR página por página
        if not any(p.strip() for p in pages):
            try:
                images = convert_from_path(temp_path)
                ocr_pages = []
                for idx, img in enumerate(tqdm(images, desc="Procesando páginas")):
                    print(f"Procesando página {idx+1}/{len(images)}...")
                    img = img.convert("L")
                    if deskew_enabled:
                        img = deskew_pil_image(img)
                    img = preprocess_single_column(img)           # 1. Preprocesamiento visual
                    psm = detect_best_psm(img)                    # 2. PSM dinámico
                    text = smart_ocr_single_column(img, psm)      # 3. OCR con config fina
                    text = expand_abbreviations(text)             # 4. Tu lógica existente
                    text = expand_slash_words(text)
                    # text = validate_with_dictionary(text)       # 5. Opcional, más lento
                    cleaned_text = clean_text(text)
                    ocr_pages.append(cleaned_text)
                pages = ocr_pages
            except Exception as e:
                os.remove(temp_path)
                return {"error": f"No se pudo extraer texto del PDF (OCR): {str(e)}"}
    else:
        os.remove(temp_path)
        return {"error": "Formato no soportado. Solo .docx y .pdf"}
    os.remove(temp_path)
    for idx in enumerate(tqdm(pages, desc="Procesando páginas")):
        tqdm.write(f"Procesando página {idx[0]+1}/{len(pages)}...")
    pages = remove_toc_tables(pages)
    texto_final = '\n\n'.join(pages)
    # Expande abreviaciones y palabras con barra
    texto_final = expand_abbreviations(texto_final)
    texto_final = expand_slash_words(texto_final)
    cleaned_text = clean_text(texto_final)
    if not cleaned_text.strip():
        return {"error": "El texto a dictar está vacío después del preprocesamiento."}
    intro = "Saludos, ¿estás listo para escuchar un resumen?"

    summary = summarize_text(cleaned_text, mode)
    summary = summary.replace("•", "")
    summary = summary.replace("-", "")
    summary = summary.replace("\n", " ")
    summary = intro + " " + summary
    summary = unicodedata.normalize("NFC", summary)
    filename = "resumen" + filename[:filename.rfind(".")] + ".wav"
    await save_audio(summary, filename)

    return {
        "summary": summary,
        "mode": mode,
        "audio_file": filename
    }