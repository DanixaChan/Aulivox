import json
import random
import os

# cargar knowledge base
topics = []

with open("dataset_factory/topics_knowledge_base.txt", encoding="utf8") as f:
    for line in f:
        topic, definition, field = line.strip().split("|")
        topics.append({
            "topic": topic,
            "definition": definition,
            "field": field
        })


# ---------- BASE TEXT VARIATIONS ----------

base_templates = [
    "{topic} es {definition}",
    "Dentro del campo de {field}, {topic} se define como {definition}",
    "{topic} corresponde a {definition}",
    "En {field}, el concepto de {topic} describe {definition}",
    "{topic} puede entenderse como {definition}"
]


def generate_base_text(topic, definition, field):

    template = random.choice(base_templates)

    return template.format(
        topic=topic,
        definition=definition,
        field=field
    )


# ---------- ABSTRACT MODE ----------

abstract_templates = [
    "{topic} es un concepto importante dentro de {field}. {definition}",
    "El estudio de {topic} permite comprender mejor fenómenos dentro de {field}.",
    "{topic} representa un elemento fundamental en el campo de {field}."
]


# ---------- STRUCTURED MODE ----------

structured_schemas = [
    "conceptual",
    "technical",
    "medical",
    "scientific",
    "historical",
    "organizational"
]

structured_templates = {

"conceptual":[
"""### Definición
{definition}

### Componentes
- Elemento principal
- Factor relevante
- Elemento asociado

### Proceso
1. Inicio del fenómeno
2. Desarrollo del proceso
3. Resultado final

### Aplicaciones
- Uso práctico
- Impacto en su campo
"""
],
"technical":[
"""### Descripción técnica
{definition}

### Componentes y especificaciones
- Componente principal: función y características clave
- Especificaciones críticas: valores, tolerancias o versiones
- Interfaces: entradas/salidas, protocolos o conexiones

### Funcionamiento (resumen)
1. Condiciones de entrada / precondiciones
2. Secuencia de operaciones / flujo de datos
3. Resultado esperado / salida

### Requisitos y dependencias
- Hardware / materiales necesarios
- Software / librerías o protocolos
- Condiciones operativas (temperatura, energía, etc.)

### Integración e implementación
- Pasos básicos para montaje o despliegue
- Buenas prácticas y recomendaciones

### Pruebas y verificación
- Prueba funcional (qué verificar)
- Prueba de rendimiento / criterios de aceptación

### Mantenimiento y seguridad
- Tareas de mantenimiento recomendadas
- Riesgos y medidas de seguridad

### Aplicaciones y limitaciones
- Ejemplos de uso típicos
- Restricciones o escenarios no recomendados
"""
],
"medical":[
"""### Definición
{definition}

### Sistemas afectados
- Sistemas/orgánicos implicados: [especificar]

### Epidemiología y factores de riesgo
- Incidencia / prevalencia: [datos o rangos si aplica]
- Factores de riesgo principales: edad, genética, hábitos, comorbilidades

### Etiología / Causas
- Causas más comunes (infecciosas, genéticas, ambientales, iatrogénicas)

### Fisiopatología
- Mecanismos subyacentes: cómo se desarrolla la enfermedad a nivel celular/tisular
- Vías fisiológicas implicadas y consecuencias funcionales

### Presentación clínica
- Signos y síntomas predominantes:
    - Síntoma 1
    - Síntoma 2
- Signos de alarma que requieren atención inmediata

### Evaluación y diagnóstico
- Historia clínica: aspectos clave a indagar
- Exploración física: hallazgos relevantes
- Pruebas complementarias recomendadas: laboratorio, imagen, pruebas funcionales
- Criterios diagnósticos (si existen estándares o guías)

### Diagnóstico diferencial
- Entidades a considerar y cómo distinguirlas

### Tratamiento y manejo
- Objetivos terapéuticos generales
- Opciones de tratamiento (farmacológico y no farmacológico), con indicaciones generales
- Manejo en fases agudas vs crónicas
- Recomendaciones para manejo de complicaciones

### Seguimiento y pronóstico
- Parámetros a monitorizar
- Factores que influyen en el pronóstico
- Expectativas de recuperación o curso clínico

### Prevención
- Medidas preventivas primarias y secundarias aplicables
- Estrategias de educación y promoción de la salud

### Complicaciones y riesgos
- Complicaciones frecuentes y cómo identificarlas

### Consideraciones especiales
- Embarazo, infancia, edad avanzada, comorbilidades y adaptación del manejo

### Recursos y referencias
- Guías, revisiones o fuentes recomendadas para profundizar (indicar fuentes locales o internacionales según disponibilidad)
"""
],

"scientific":[
"""### Definición
{definition}

### Marco teórico / contexto
Breve explicación del marco conceptual, teorías relacionadas y antecedentes relevantes.

### Mecanismos / procesos
- Descripción de los mecanismos subyacentes
- Factores que modulan el fenómeno
- Dinámica temporal o escala relevante

### Metodologías de estudio
- Técnicas experimentales, observacionales o analíticas típicas
- Variables clave y cómo se cuantifican

### Evidencia y ejemplos
- Resultados empíricos representativos
- Estudios de caso o experimentos ilustrativos

### Implicaciones científicas
- Consecuencias para la disciplina y posibles aplicaciones
- Conexiones con otras áreas del conocimiento

### Limitaciones y preguntas abiertas
- Supuestos, incertidumbres y limitaciones de los enfoques actuales
- Líneas de investigación futuras

### Referencias sugeridas
- Artículos, autores o recursos para profundizar
"""
],

"historical":[
"""### Causas
- Factor político
- Factor social
- Factor económico

### Detonante
Evento que desencadena el conflicto.

### Fechas clave
- Inicio del conflicto: [Fecha de inicio]
- [Evento intermedio]: [Fecha del evento intermedio]
- Fin del conflicto: [Fecha de fin]

### Desarrollo
- Primera fase
- Expansión del conflicto
- Punto de inflexión

### Consecuencias
- Cambio político
- Impacto social
"""
],

"organizational":[
"""### Definición
{definition}

### Propósito y objetivos
- Objetivo principal: qué busca lograr la organización/unidad
- Objetivos secundarios: metas a corto y mediano plazo

### Estructura y organigrama
- Niveles (dirección, gestión, operaciones)
- Unidades clave y su relación jerárquica
- Canales de comunicación internos

### Roles y responsabilidades
1. Rol principal: funciones y responsabilidades críticas
2. Roles de soporte: tareas y dependencias
3. Líderes responsables: decisiones y rendición de cuentas

### Procesos clave y flujos de trabajo
- Procesos críticos (entrada → transformación → salida)
- Interacciones entre unidades
- Puntos de coordinación y transferencia de información

### Gobernanza y toma de decisiones
- Mecanismos de decisión (quién decide y cómo)
- Políticas, normas y procedimientos
- Control interno y cumplimiento

### Indicadores y evaluación (KPIs)
- Métricas principales para medir desempeño
- Frecuencia de revisión y responsables de seguimiento

### Recursos y capacidades
- Recursos humanos: habilidades necesarias
- Recursos materiales y financieros
- Capacitación y desarrollo

### Riesgos y mitigación
- Principales riesgos organizacionales
- Estrategias de mitigación y contingencia

### Buenas prácticas e integración
- Recomendaciones operativas y de coordinación
- Ejemplos de integración con otras unidades o proyectos

### Ejemplo / caso práctico (opcional)
- Breve ilustración de cómo se aplica la estructura en un caso real
"""
]
}


# ---------- NARRATIVE MODE ----------

narrative_templates = [

"""### Introducción
El concepto de {topic} aparece con frecuencia en el estudio de {field}.

### Explicación
{definition}

### Conclusión
Comprender este concepto ayuda a interpretar distintos fenómenos dentro de su campo.""",

"""### Introducción
En muchas áreas del conocimiento, {topic} juega un papel importante.

### Desarrollo
{definition}

### Reflexión
Su estudio ha permitido avanzar en la comprensión de {field}.
"""
]


# ---------- FLASHMODE ----------

flash_templates = [

"""Pregunta: ¿Qué es {topic}?
Respuesta: {definition}

Pregunta: ¿En qué campo se estudia {topic}?
Respuesta: Principalmente en {field}.
""",

"""Pregunta: ¿Cómo se define {topic}?
Respuesta: {definition}

Pregunta: ¿Por qué es importante {topic}?
Respuesta: Porque ayuda a comprender conceptos dentro de {field}.
"""
]


# ---------- INPUT VARIATIONS ----------

input_templates = [
"Modo: {mode}\nTexto: {text}"
# "Tarea: generar salida en modo {mode}\nContenido: {text}",
# "Instrucción: modo {mode}\nEntrada: {text}",
# "Formato solicitado: {mode}\nTexto base: {text}"
]


dataset = []

# Mapeo de campo a esquema structured
field_to_schema = {
    # Ciencias biológicas
    "biología": "scientific",
    "medicina": "medical",
    "salud": "medical",
    "genética": "scientific",
    "bioquímica": "scientific",

    # Ciencias físicas y exactas
    "física": "scientific",
    "química": "scientific",
    "matemáticas": "conceptual",
    "astronomía": "historical",
    "cosmología": "historical",

    # Ciencias sociales y humanidades
    "historia": "historical",
    "derecho": "organizational",
    "sociología": "organizational",
    "psicología": "organizational",
    "filosofía": "conceptual",

    # Ciencias ambientales
    "ecología": "scientific",
    "medio ambiente": "scientific",
    "geografía": "historical",

    # Tecnología e ingeniería
    "tecnología": "technical",
    "informática": "technical",
    "ingeniería": "technical",
    "energía": "technical",
    "electrónica": "technical",
    "robótica": "technical",

    # Economía y administración
    "economía": "organizational",
    "administración": "organizational",
    "negocios": "organizational",
    "finanzas": "organizational",

    # Otros campos generales
    "educación": "organizational",
    "arte": "conceptual",
    "literatura": "conceptual",
    "comunicación": "organizational",
    "política": "organizational",
    # Si el campo no está definido, usar conceptual
}

for item in topics:
    topic = item["topic"]
    definition = item["definition"]
    field = item["field"]

    for _ in range(5):  # variaciones por topic
        base_text = generate_base_text(topic, definition, field)

        # elegir esquema structured según el campo
        schema = field_to_schema.get(field, "conceptual")  # default: conceptual
        structured_output = random.choice(structured_templates[schema]).format(
            definition=definition
        )

        outputs = {
            "abstract": random.choice(abstract_templates).format(
                topic=topic,
                field=field,
                definition=definition
            ),
            "structured": structured_output,
            "narrative": random.choice(narrative_templates).format(
                topic=topic,
                definition=definition,
                field=field
            ),
            "flashmode": random.choice(flash_templates).format(
                topic=topic,
                definition=definition,
                field=field
            )
        }

        for mode, output in outputs.items():
            input_text = random.choice(input_templates).format(
                mode=mode,
                text=base_text
            )
            dataset.append({
                "input": input_text,
                "target": output
            })

# guardar dataset
output_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "train_model",
    "dataset_train.jsonl"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print("Dataset generado:", len(dataset))