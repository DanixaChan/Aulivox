# Formato actual de Dataset
## Se formaliza los siguientes formatos para la entrada y target de cada ejemplo y cada modo implementado en el dataset de entrenamiento para el modelo IA "aulivoxmodel":
### Input
En el dataset, para cada modo en cada ejemplo, se debe definir el dato de entrada de la sgte. manera, siendo "Modo: " la modalidad seleccionada y "Texto: " el texto de entrada recibida como tema/información/dato:

Ejemplo .jsonl :

    {"input": "Modo: abstract\nTexto: **Tema**",...}

### Modo "Abstract":
**Objetivo: Resumen corto y directo, dividido en concepto, idea clave y importancia del tema/información/dato.**
Ejemplo .jsonl :

    {"target": "### Concepto:\n**ConceptodelTema**\n\n### Idea Clave:\n**IdeaClavedelTema**\n\n### Importancia:\n**ImportanciadelTema** "}

### Modo "Structured":
**Objetivo: Apuntes ordenados tipo cuaderno, dividido en definición, componentes, proceso y aplicaciones del tema/información/dato, su división (exceptuando definición) puede variar según el tema a tratar (por ej. formación, estructura, propiedades físicas, tipos, efectos, causas, consecuencias, reglas, ingredientes...)**	

Ejemplo .jsonl :

    {"target":"### Definición: **DefinicióndelTema**\n\n### Componentes:\n- **Componente1**\n- **Componente2**\n- **Componente3**\n- **Componente4**\n\n### Proceso:\n1. **Proceso1**\n2. **Proceso2**\n3. **Proceso3**\n\n### Aplicaciones:\n- **Aplicación1**\n- **Aplicación2**"}

### Modo "Narrative":
**Objetivo: Explicación fluida tipo profesor, dividido en introducción, explicación y conclusión del tema/información/dato.**	

Ejemplo .jsonl :

    {"target":"### Introducción: **IntroducciónalTema**\n\n### Explicación: **ExplicacióndelTema**\n\n### Conclusión: **ConclusióndelTema**"}

### Modo "flashcard":
**Objetivo: tarjetas de estudio rápidas (flashcards), dividido en pregunta y respuesta del tema/información/dato.**	

Ejemplo .jsonl :

    {"target": "Pregunta: ¿**Pregunta1**?\nRespuesta: **Respuesta1**\n\nPregunta: ¿**Pregunta2**?\nRespuesta: **Respuesta2**\n\nPregunta: ¿**Pregunta3**?\nRespuesta: **Respuesta3**"}

### Consideraciones:
- Respetar puntuación y espacios, tal cuál se le indica en cada sección.
- target debe ser conciso y claro según el modo a considerar.
- Debe haber variación para cada tema y ejemplo , garantizando calidad del dataset.
- El formato de archivo principal es **.jsonl**, por ende respetar sintaxis.
- Se debe filtrar de forma adecuada, no se debe agregar carácteres especiales a parte de los ya implementados en cada indicación.

### Plantilla Copy-Paste:
```
{	"input": "Modo: abstract\nTexto: ", 
	"target": "### Concepto:\n \n\n### Idea Clave:\n \n\n### Importancia:\n "}
{	"input": "Modo: structured\nTexto: ", 
	"target":"### Definición: \n\n### Componentes:\n- \n- \n- \n- \n\n### Proceso:\n1. \n2. \n3. \n\n### Aplicaciones:\n- \n- "}
{	"input": "Modo: narrative\nTexto: ", 
	"target":"### Introducción: \n\n### Explicación: \n\n### Conclusión: "}
{	"input": "Modo: flashmode\nTexto: ", 
	"target": "Pregunta: ¿?\nRespuesta: \n\nPregunta: ¿?\nRespuesta: \n\nPregunta: ¿?\nRespuesta: "}
```
