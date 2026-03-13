import docx
import PyPDF2

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import re

def remove_toc_tables(pages):

    toc_keywords = ["índice", "contenido", "tabla de contenidos", "contents"]

    cleaned_pages = []

    for page in pages:

        lines = page.splitlines()
        cleaned_lines = []

        inside_toc = False

        for line in lines:

            lower = line.lower()

            # Detectar inicio del índice
            if any(k in lower for k in toc_keywords):
                inside_toc = True
                continue

            # Si estamos dentro del índice
            if inside_toc:

                # línea típica de índice
                if re.search(r"\d+\s*$", line):
                    continue
                else:
                    # terminó el índice
                    inside_toc = False

            cleaned_lines.append(line)

        cleaned_pages.append("\n".join(cleaned_lines))

    return cleaned_pages

def iter_block_items(parent):
    """
    Itera párrafos y tablas del documento en el orden real.
    """
    for child in parent.element.body.iterchildren():
        if child.tag.endswith("p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("tbl"):
            yield Table(child, parent)


def extract_text_from_word(path):

    doc = Document(path)

    pages = []
    current_page = []

    for block in iter_block_items(doc):

        # -------------------
        # PÁRRAFOS
        # -------------------
        if isinstance(block, Paragraph):

            text = block.text.strip()

            if text:
                current_page.append(text)

        # -------------------
        # TABLAS
        # -------------------
        elif isinstance(block, Table):

            for row in block.rows:

                row_text = []

                for cell in row.cells:
                    cell_text = cell.text.strip()

                    if cell_text:
                        row_text.append(cell_text)

                if row_text:
                    current_page.append(" | ".join(row_text))

    if current_page:
        pages.append("\n".join(current_page))

    return pages

# ========== PDF ==========
def extract_text_from_pdf(path):
    reader = PyPDF2.PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return pages

# ========== UTILIDAD ==========
def skip_index_pages(pages):
    # Heurística mejorada: omitir sólo la PRIMERA página que parezca una tabla de contenidos (TOC).
    # Criterios de TOC (suficiente para detectar índices, pero evitar tablas normales):
    # 1) Palabras clave ('índice', 'tabla de contenidos', 'contenido') aparecen en las primeras 6 líneas.
    # 2) O muchas líneas que son entradas de índice: líneas con líderes (puntos/guiones) seguidas de un número/página.
    def is_toc_page(page_text):
        if not page_text or not page_text.strip():
            return False

        lines = [l.strip() for l in page_text.splitlines() if l.strip()]
        text_lower = page_text.lower()

        # 1️⃣ Debe contener palabra clave de índice
        skip_keywords = ["índice", "tabla de contenidos", "contenido", "contents", "contenidos"]

        if not any(k in text_lower for k in skip_keywords):
            return False

        # 2️⃣ Debe parecer lista de secciones
        import re
        toc_entry_re = re.compile(r".+\.{2,}\s*\d+$")

        toc_like = sum(1 for ln in lines if toc_entry_re.match(ln))

        if toc_like >= 3:
            return True

        return False

    result = []
    skipped = False
    for page in pages:
        if not skipped and is_toc_page(page):
            skipped = True
            continue
        result.append(page)
    return result

# Ejemplo de uso:
# pages = extract_text_from_word('archivo.docx')
# pages = extract_text_from_pdf('archivo.pdf')
# pages = skip_index_pages(pages)
# texto_final = '\n\n'.join(pages)
