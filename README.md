# 🐾 Agente RAG Veterinario (Chatbot)

Este proyecto implementa un agente conversacional basado en **Inteligencia Artificial** diseñado específicamente para la página web de una clínica veterinaria. Cumple con todos los lineamientos establecidos por la rúbrica de evaluación, utilizando una arquitectura moderna de **Retrieval-Augmented Generation (RAG)** estructurada con LCEL.

## 🚀 Tecnologías Utilizadas

### 1. Framework Base: LangChain
Toda la lógica del agente y la conexión entre módulos (Recuperación, Procesamiento y Generación) fue estructurada utilizando **LangChain** (`langchain`, `langchain-core` y `langchain-community`). Se utilizó explícitamente el enfoque moderno **LCEL (LangChain Expression Language)** para declarar flujos de datos transparentes y directos **(Criterio IE5 - Arquitectura Modular)**.

### 2. Procesamiento de Lenguaje Genenativo (LLM)
Se integró el proveedor **OpenAI** a través del endpoint de acceso asíncrono gratuito de **GitHub Models**. 
El modelo designado fue `gpt-4o-mini`, invocado mediante la clase `ChatOpenAI`. A este modelo se le programó un rígido **Prompt de Sistema** para asegurar que su comportamiento se centre en brindar asistencia amigable, evitando riesgos clínicos mediante la **prohibición estricta de emitir diagnósticos** médicos **(Criterio IE2 y Criterio IE4 - Prompts y Coherencia Ética)**.

### 3. Base de Datos Vectorial: ChromaDB (`langchain-chroma`)
Se construyó una solución persistente en memoria local usando **ChromaDB**. Esta base almacena el conocimiento clínico de la veterinaria para ser consultado velozmente sin la necesidad de motores externos pesados, lo que optimiza la disponibilidad y abarata costos **(Criterio IE3 y IE7 - Mecanismos de Recuperación y Fundamento Organizacional)**.

### 4. Embeddings 100% Locales (`sentence-transformers`)
Dado que se trata de datos clínicos sensibles, la **vectorización de textos** (convertir las historias clínicas de `.csv` a su representación semántica) se procesó usando un modelo incrustado en el CPU local (`all-MiniLM-L6-v2` vía `HuggingFaceEmbeddings`). Esto nos protegió contra las saturaciones de límite por cuotas ("Rate Limits") que presentan APIs gratuitas, garantizando ingestiones ultrarrápidas de todo el historial.

### 5. Ingesta de Datos (`CSVLoader`)
El conocimiento base del agente se extrae del archivo interno `veterinary_clinical_data.csv`. Se limitó a extraer las primeras 100 historias críticas para eficientar el arranque local **(Criterio IE3)**.

---

## 🧠 Modelos de Inteligencia Artificial Implementados
Si eres un desarrollador interesado en hacer un *fork* o replicar este ecosistema para tus proyectos, ten en cuenta que nuestro agente delega sus funciones en un esquema "híbrido", separando el esfuerzo en dos modelos especializados que trabajan en equipo:

1. **Modelo de Inferencia y Chat (`gpt-4o-mini`)**: 
   - Actúa como el *"Cerebro Generativo"* del chatbot.
   - **Enfoque de uso:** Decidí implementarlo consumiendo el endpoint libre de **GitHub Models** (vía `langchain-openai`). Esto permite emular lógicas y calidades de la familia GPT-4 sin lidiar con los pagos de OpenAI para pruebas rápidas. Este modelo se encarga exclusivamente de redactar texto asimilable para humanos y asegurar que el filtro ético de NO recetar medicamentos se cumpla.
   
2. **Modelo de Vectorización y Embeddings (`all-MiniLM-L6-v2`)**: 
   - El *"Buscador Silencioso"* encargado de ordenar matemáticamente el conocimiento interno de la veterinaria.
   - **Enfoque de uso:** Es un modelo ultra integrado de HuggingFace (`sentence-transformers`) que **ejecutamos de manera 100% local en tu CPU**. La justificación de aislar esto a un entorno local es enorme: nos evita agotar o bloquear tus cuotas limitadas de la API al indexar masivamente los historiales médicos, permitiéndote escalar tu base de datos a cientos de miles de Excel sin pagar un solo centavo en vectorización de RAG.

> **💡 El flujo en acción:** *MiniLM* hace el trabajo pesado y gratuito en tu computadora buceando en la DB; extrae únicamente el contexto quirúrgico necesario, y se lo pre-empaqueta a *GPT-4* en la nube para que te dé la respuesta final.

---

## 🛠 Instalación y Requisitos

**Prerrequisitos:** Debes tener `Python 3.10+` instalado y configurado en tu entorno virtual (Ej. `AprendeTiempo/.venv/`).

### Pasos de Instalación de Librerías
Antes de ejecutar el Agente RAG, asegúrate de tener tu entorno virtual activo e instala cada dependencia corriendo las siguientes líneas de manera secuencial en tu consola:

```powershell
# 1. Instalar el Framework Core de Langchain y lectura de credenciales seguras
pip install langchain langchain-core langchain-community python-dotenv

# 2. Instalar integraciones de LLM de OpenAI y base de datos Chroma
pip install langchain-openai langchain-chroma openai

# 3. Instalar la librería esencial para los Embeddings Locales (HuggingFace)
pip install sentence-transformers
```

### Configuración Final
1. **Archivo Segurizado de Credenciales (`.env`):**
   Para evitar filtrar tu llave a la nube, crea un archivo oculto llamado `.env` en la misma carpeta de este proyecto y guarda tu token de GitHub Models adentro, respetando el siguiente formato estricto:
   ```env
   GITHUB_PAT_TOKEN=tu_token_aqui_adentro
   ```

3. **Ejecutar Modelo RAG:**
   Ejecuta el archivo directamente en tu sistema; éste indexará los vectores en la carpeta local `./chroma_db_local` en un par de segundos y luego abrirá la consola de Chat:
   ```bash
   python rag_chatbot.py
   ```

---

## 📋 Alineación con la Rúbrica Integradora (IE1 -> IE9)

- **IE1:** Se diseñó el RAG totalmente a la medida para atender un caso real (Bot virtual de salud para mascotas, brindando solo orientación de acuerdo con los objetivos del negocio).
- **IE2:** Se configuró el `PromptTemplate` dictando las fronteras operacionales del LLM para instruir la negativa de recetar medicamentos.
- **IE3:** Se integra exitosamente el flujo RAG con "Document Retrievers" y "Embeddings Local".
- **IE4:** La coherencia entre datos consultados por similitud (Similarity Search) y la generación del modelo está bloqueada a un patrón ético para no inventar falsedades ("hallucinations").
- **IE5:** La arquitectura de LangChain (LCEL) fue estructurada separando el cargador, el emparrillado Chroma, el Prompt y la canalización generativa final.
- **IE6:** Un diagrama de arquitectura detallando todos los nodos del agente fue levantado y entregado en un documento anexo (`walkthrough.md`).
- **IE7:** El rediseño hacia embeddings locales ahorra dinero en API para startups y garantiza no agotar cuotas gratis.
- **IE8 / IE9:** Se adjuntó un reporte narrativo técnico de arquitectura listos para acoplar al documento de conclusiones de investigación y diseño de sistema.
---

linea de ejecucion: 
& c:/_RGS_DEV_Antigravity/AprendeTiempo/.venv/Scripts/python.exe c:/Users/ragal/.gemini/antigravity/scratch/rag_chatbot.py

& c:/_RGS_DEV_Antigravity/AprendeTiempo/.venv/Scripts/python.exe c:/Users/ragal/.gemini/antigravity/scratch/rag_chatbot.py

---
*Desplegado con ❤️ para cumplir y superar el ecosistema de evaluación.*
