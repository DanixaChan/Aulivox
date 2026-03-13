# Aulivox

## Descripción

¿Alguna vez has necesitado estudiar un libro, artículo o investigación, pero no tienes suficiente tiempo para leerlo?

Aulivox es un sistema basado en inteligencia artificial capaz de transformar documentos como libros, artículos o reportes de investigación en contenido de audio. El sistema analiza el texto, genera resúmenes o explicaciones detalladas y los convierte en un audiolibro que puede escucharse en cualquier momento y lugar.

El objetivo de Aulivox es facilitar el acceso al conocimiento permitiendo que los usuarios puedan aprender mientras realizan otras actividades.

---

## Progreso actual

Actualmente el proyecto se encuentra en fase de desarrollo e investigación.

Progreso del sistema:

- Implementación de APIs para el procesamiento de documentos mediante OCR.
- Desarrollo de funcionalidades principales del backend.
- Entrenamiento del modelo de inteligencia artificial para generación de resúmenes (3% del dataset procesado).
- Desarrollo del sistema de generación de resúmenes y narración detallada a partir de texto procesado por OCR y texto plano.

![Alt Sistema en funcionamiento](https://github.com/DanixaChan/Aulivox/EvidenciaVisual/AULIVOX_V0.1.gif)

---

¿Cómo ejecutarlo?
De momento solo está disponible el backend, por lo que se puede acceder sus funcionalidades en FastApi con el modelo AI local, lo cuál aún no está disponible lo último.

Acceder al repositorio local:
git clone repo
cd aulivox/backend

Crear entorno virtual:
python -m venv myvenv
myvenv/Scripts/activate

Instalar dependencias:
pip install -r requirements.txt

Iniciar servidor local:
python -m uvicorn main:app --reload

---

## Tecnologías
### Backend
- **Python 3**
- **FastAPI** – Framework para APIs web rápidas y asíncronas
- **Pillow (PIL)** – Procesamiento de imágenes
- **OpenCV** – Procesamiento avanzado de imágenes
- **pytesseract** – OCR (Reconocimiento Óptico de Caracteres)
- **pdf2image** – Conversión de PDF a imágenes
- **Pydantic** – Validación de datos
- **asyncio** – Programación asíncrona

### Frontend
- **Vite** – Herramienta de build para frontend moderno
- **React** – Biblioteca para interfaces de usuario
- **TypeScript** – Superset de JavaScript tipado
- **CSS** – Estilos personalizados
- 
---
### Arquitectura
- backend/
    - dataset_factory
    - summarizer
    - train_model
    - tts

- frontend/
    - react app

---

### Proceso del Modelo de IA

El sistema utiliza un modelo de resumen entrenado con datasets educativos generados mediante una fábrica de datos sintéticos.

El modelo aprende a transformar texto en diferentes modos de representación:

- resumen abstracto (mode: abstract)
- resumen estructurado (mode: structured)
- resumen narrativo (mode: narrative)
- flashcards (mode: flashcards)
