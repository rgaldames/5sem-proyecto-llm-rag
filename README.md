# Proyecto Académico: Agente RAG Veterinario

> **Asignatura:** Ingeniería de Soluciones con IA  
> **Evaluación:** Desarrollo de un Agente Funcional  
> **Framework:** LangChain 1.2 + LangGraph  
> **Dominio:** Asistente virtual de orientación para clínica veterinaria

---

## Descripción

Sistema de agente funcional que integra herramientas de **consulta, escritura y razonamiento** en un flujo de trabajo veterinario simulado. El agente usa el patrón **Tool-Calling Loop** (equivalente moderno al ReAct) de LangChain 1.2, combinando recuperación de información (RAG), memoria de corto y largo plazo, y toma de decisiones adaptativa.

---

## Requisitos del Sistema

| Requisito | Versión mínima |
|---|---|
| Python | 3.10+ (probado en 3.13.9) |
| Sistema Operativo | Windows / macOS / Linux |
| Conexión a internet | Necesaria (para GitHub Models API) |
| Token de GitHub | Con acceso a GitHub Models Beta |

---

## Instalación Paso a Paso

### 1. Clona el repositorio

```bash
git clone https://github.com/rgaldames/5sem-proyecto-llm-rag.git
cd 5sem-proyecto-llm-rag
```

### 2. Crear y activar un entorno virtual

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias

Instala los paquetes en el orden indicado para evitar conflictos de versiones:

```bash
# LangChain 1.2+ con todas sus integraciones
pip install langchain==1.2.15
pip install langchain-core==1.3.0
pip install langchain-community==0.4.1
pip install langchain-openai==1.1.14
pip install langchain-chroma==1.1.0

# Embeddings locales (HuggingFace  no requiere API key)
pip install sentence-transformers==5.4.1

# Utilidades
pip install python-dotenv==1.2.2
pip install numpy==2.4.4
```

> **Nota:** Si prefieres instalar todo de una vez:
> ```bash
> pip install langchain==1.2.15 langchain-core==1.3.0 langchain-community==0.4.1 langchain-openai==1.1.14 langchain-chroma==1.1.0 sentence-transformers==5.4.1 python-dotenv numpy
> ```

### 4. Configurar las credenciales

Crea un archivo llamado `.env` en la raíz del proyecto:

```
GITHUB_PAT_TOKEN=tu_github_personal_access_token
```

**Cómo obtener el token de GitHub Models:**

1. Ir a [github.com](https://github.com)  Click en tu avatar  **Settings**
2. Menú lateral  **Developer settings**
3. **Personal access tokens**  **Tokens (classic)**
4. Click en **Generate new token (classic)**
5. Dale un nombre (ej: `VetBot-LLM`) y selecciona sin permisos especiales
6. Copia el token generado y pégalo en el `.env`

> **Importante:** El token de GitHub Models da acceso gratuito a `gpt-4o-mini` a través del endpoint de Azure Inference (`https://models.inference.ai.azure.com`). No es lo mismo que una API Key de OpenAI.

---

## Estructura del Proyecto

```
5sem-proyecto-llm-rag/
|
|-- agent.py                     <- Agente principal (Tool-Calling Loop + orquestación)
|-- tools.py                     <- 3 herramientas: consulta, escritura, razonamiento
|-- short_term_memory.py         <- Memoria de corto plazo (buffer deslizante k=5)
|-- memory_store.py              <- Memoria de largo plazo (JSON + embeddings semánticos)
|-- rag_chatbot.py               <- Chatbot RAG original (Evaluación Parcial 1)
|-- read_pdf.py                  <- Utilidad de lectura de PDFs
|
|-- veterinary_clinical_data.csv <- Base de conocimiento clínica (50+ registros usados)
|-- memories.json                <- Memorias persistentes entre sesiones (se genera solo)
|-- visit_summaries.jsonl        <- Registro de consultas guardadas (se genera solo)
|-- chroma_db_local/             <- Base de datos vectorial ChromaDB (se genera sola)
|
|-- .env                         <- Credenciales (NO subir a GitHub  en .gitignore)
|-- .gitignore
+-- README.md
```

> Los archivos `memories.json`, `visit_summaries.jsonl` y la carpeta `chroma_db_local/` se crean automáticamente la primera vez que ejecutas el agente.

---

## Ejecución

### Ejecutar el Agente Funcional

```bash
python agent.py
```

**Primera ejecución:** el agente tarda ~30-60 segundos en iniciar porque descarga el modelo de embeddings `all-MiniLM-L6-v2` desde HuggingFace (solo ocurre una vez).

**Ejemplo de sesión con el Agente Funcional:**
```
=================================================================
  AGENTE VETERINARIO FUNCIONAL
=================================================================
  Framework  : LangChain 1.2 + LangGraph (create_agent)
  Herramientas: Consulta | Escritura | Razonamiento
  Memoria    : Corto Plazo (buffer) + Largo Plazo (semantico)
=================================================================

[1/5] Configurando modelo de lenguaje (gpt-4o-mini)...
[2/5] Cargando modelo de embeddings local (all-MiniLM-L6-v2)...
[3/5] Indexando base de conocimiento clinico en ChromaDB...
     50 registros clinicos indexados.
[4/5] Inicializando sistema de memoria dual...
[5/5] Construyendo agente con herramientas...
     Agente compilado con 3 herramientas.

Agente listo!

Usuario: Mi gato tiene tos y lleva dos dias sin comer
[Agente procesando...]

VetBot responde:
-----------------------------------------------------------------
(respuesta del agente...)
=================================================================

Usuario: _
```

### Comandos especiales dentro del agente

| Comando | Descripción |
|---|---|
| `estado` | Muestra estadísticas de memoria (turno, buffer actual) |
| `historial` | Imprime la conversación reciente del buffer de corto plazo |
| `salir` / `exit` / `quit` | Apaga el agente limpiamente |

### Ejecutar el Chatbot RAG original (Evaluación Parcial 1)

```bash
python rag_chatbot.py
```

---

## Arquitectura del Sistema

```
+------------------------------------------------------------------+
|                  AGENTE VETERINARIO FUNCIONAL                    |
|                                                                  |
|   Usuario                                                        |
|      |                                                           |
|      v                                                           |
|  +---------------------------------------------------------------+|
|  |               SISTEMA DE MEMORIA DUAL                        ||
|  |                                                               ||
|  |  +-----------------------+   +---------------------------+   ||
|  |  | CORTO PLAZO           |   | LARGO PLAZO               |   ||
|  |  | ShortTermMemory       |   | MemoryStore               |   ||
|  |  | HumanMessage/AIMessage|   | JSON persistente          |   ||
|  |  | Buffer deslizante k=5 |   | Embeddings cosine sim.    |   ||
|  |  +-----------------------+   +---------------------------+   ||
|  +---------------------------------------------------------------+|
|      |                                                           |
|      v                                                           |
|  +---------------------------------------------------------------+|
|  |      create_agent()  LangGraph Tool-Calling Loop            ||
|  |                                                               ||
|  |  LLM decide si usar herramienta o responder directamente     ||
|  |                                                               ||
|  |  +-----------------------------------------------------+     ||
|  |  |                   HERRAMIENTAS                      |     ||
|  |  |                                                     |     ||
|  |  | [CONSULTA]    search_clinical_db  -> ChromaDB RAG   |     ||
|  |  | [ESCRITURA]   write_visit_summary -> JSONL file     |     ||
|  |  | [RAZONAMIENTO] analyze_symptoms  -> Reglas urgencia |     ||
|  |  +-----------------------------------------------------+     ||
|  +---------------------------------------------------------------+|
|      |                                                           |
|      v                                                           |
|  +---------------------------------------------------------------+|
|  | LLM: gpt-4o-mini via GitHub Models (Azure Inference)         ||
|  | Embeddings: all-MiniLM-L6-v2 (local, sin API key)            ||
|  | Vector DB: ChromaDB (local, persistente)                      ||
|  +---------------------------------------------------------------+|
|      |                                                           |
|      v                                                           |
|   Respuesta final al Usuario                                     |
+------------------------------------------------------------------+
```

---

## Descripción de Componentes

### `agent.py`  Orquestador Principal

Punto de entrada del sistema. Inicializa todos los módulos y ejecuta el bucle de interacción.

- Usa `create_agent()` de LangChain 1.2 (basado en LangGraph internamente)
- El LLM llama herramientas en loop automático hasta tener suficiente información
- Integra memoria dual en cada invocación mediante `run_agent()`
- Inyecta el contexto de largo plazo directamente en el mensaje de entrada

### `tools.py`  Herramientas del Agente

Define las tres herramientas que el agente puede invocar de forma autónoma:

| Herramienta | Tipo | Descripción |
|---|---|---|
| `search_clinical_db` | **CONSULTA** | Búsqueda semántica en ChromaDB sobre 50 historiales clínicos |
| `write_visit_summary` | **ESCRITURA** | Persiste consultas en `visit_summaries.jsonl` para auditoría |
| `analyze_symptoms` | **RAZONAMIENTO** | Clasifica síntomas como CRÍTICOS / MODERADOS / LEVES por reglas |

### `short_term_memory.py`  Memoria de Corto Plazo

Buffer de ventana deslizante implementado con `HumanMessage` / `AIMessage` de `langchain_core`.

- Mantiene las últimas `k=5` interacciones (pares pregunta + respuesta)
- Se resetea al terminar cada sesión
- Compatible nativamente con la API de mensajes de LangChain 1.2

### `memory_store.py`  Memoria de Largo Plazo

Almacenamiento persistente entre sesiones usando embeddings semánticos y similitud coseno.

- Guarda cada interacción como vector en `memories.json`
- Recupera los 3 recuerdos más relevantes semánticamente para cada nueva consulta
- Usa `all-MiniLM-L6-v2` para la vectorización (100% local, sin API key)

---

## Planificación Adaptativa

El agente **no sigue un flujo rígido**. El LLM decide en tiempo real qué herramienta usar según el tipo de consulta:

```
Input del usuario
      |
      v
  LLM analiza el contenido
      |
      +-------> Sintomas mencionados?        -> analyze_symptoms    -> nivel de urgencia
      |
      +-------> Pregunta clinica especifica? -> search_clinical_db  -> contexto clinico
      |
      +-------> Consulta concluida?          -> write_visit_summary -> registro persistente
      |
      +-------> Puede responder directamente? -> respuesta sin llamar herramientas
      |
      v
  Respuesta final al usuario
```

**Ejemplos de comportamiento adaptativo:**

| Consulta del usuario | Herramienta elegida | Razón |
|---|---|---|
| "Mi perro tuvo una convulsión" | `analyze_symptoms` | Detecta síntoma crítico  alerta urgencia |
| "¿Qué tratamientos usan para otitis?" | `search_clinical_db` | Busca en historiales clínicos |
| "Gracias, mi gato se llama TOM" | `write_visit_summary` | Registra el cierre de la consulta |
| "¿Cuántas veces al día debo alimentar a un gatito?" | *(sin herramienta)* | El LLM responde directamente |

---

## Preguntas de Prueba Sugeridas

Puedes probar con las siguientes preguntas o crear las tuyas al ejecutar `agent.py`:

```
1. Mi perro lleva dos dias con tos y sin apetito. ¿Que hago?

2. ¿Que tipo de enfermedades han tenido los gatos en la clinica?

3. Mi gato tuvo una convulsion hace 10 minutos.

4. ¿Cada cuanto debo llevar a mi perro al veterinario para chequeo preventivo?

5. Mi mascota se llama Max, es un labrador de 4 anios y consulte por perdida de pelo.
   Guarda un resumen de esta consulta.
```

---

## Solución de Problemas Frecuentes

### Error: `cannot import name 'AgentExecutor' from 'langchain.agents'`

Este error ocurre si tienes LangChain 0.x instalado. Este proyecto usa **LangChain 1.2+** que reemplazó `AgentExecutor` / `create_react_agent` por `create_agent`. Solución:

```bash
pip install --upgrade langchain==1.2.15 langchain-core==1.3.0
```

### Error: `No module named 'langchain.memory'`

El módulo `langchain.memory` fue eliminado en LangChain 1.x. El proyecto ya implementa su propia memoria en `short_term_memory.py` sin depender de ese módulo. Verifica que estés ejecutando `agent.py` (no `rag_chatbot.py`).

### Error: `No se encontro GITHUB_PAT_TOKEN`

Verifica que el archivo `.env` existe en la misma carpeta que `agent.py` con el formato exacto:
```
GITHUB_PAT_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```
Sin comillas, sin espacios alrededor del `=`.

### Error: `looping content` del modelo

El proyecto ya tiene este comportamiento corregido en el system prompt de `agent.py`. El LLM decide autónomamente si llamar o no una herramienta, evitando el loop.

### El modelo de embeddings tarda mucho en cargar

Normal en la primera ejecución  descarga `all-MiniLM-L6-v2` (~90 MB) desde HuggingFace. En ejecuciones siguientes se carga desde caché local en segundos.

### Warning: `HuggingFaceEmbeddings was deprecated`

Es solo un aviso, no un error. El proyecto funciona correctamente.

---

## Versiones Exactas Utilizadas

| Paquete | Versión |
|---|---|
| Python | 3.13.9 |
| langchain | 1.2.15 |
| langchain-core | 1.3.0 |
| langchain-community | 0.4.1 |
| langchain-openai | 1.1.14 |
| langchain-chroma | 1.1.0 |
| sentence-transformers | 5.4.1 |
| chromadb | 1.5.8 |
| openai | 2.32.0 |
| numpy | 2.4.4 |
| python-dotenv | 1.2.2 |
| huggingface-hub | 1.11.0 |

---

## Indicadores de Logro

| IL | Indicador | Implementación |
|---|---|---|
| **IL2.1** | Herramientas de consulta, escritura y razonamiento | `search_clinical_db`, `write_visit_summary`, `analyze_symptoms` en `tools.py` |
| **IL2.2** | Memoria de corto y largo plazo | `ShortTermMemory` (buffer HumanMessage/AIMessage k=5) + `MemoryStore` (embeddings JSON persistentes) |
| **IL2.3** | Planificación y toma de decisiones adaptativas | LangGraph tool-calling loop: el LLM decide autónomamente qué herramienta invocar y cuándo |
| **IL2.4** | Documentación técnica y orquestación | Diagramas ASCII en código fuente + este README con arquitectura, flujos y ejemplos |

---

## Referencias Bibliográficas (APA)

Chase, H. (2022). *LangChain* [Software]. GitHub. https://github.com/langchain-ai/langchain

LangChain AI. (2024). *LangChain 1.2 documentation: Agents*. https://python.langchain.com/docs/modules/agents/

LangGraph. (2024). *LangGraph: Build stateful, multi-actor applications with LLMs*. https://langchain-ai.github.io/langgraph/

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 94599474. https://doi.org/10.48550/arXiv.2005.11401

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. https://doi.org/10.48550/arXiv.1908.10084

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *International Conference on Learning Representations (ICLR 2023)*. https://doi.org/10.48550/arXiv.2210.03629

Chroma. (2023). *Chroma: The open-source embedding database*. https://docs.trychroma.com/

Hugging Face. (2023). *Sentence Transformers: all-MiniLM-L6-v2*. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

---

*Desplegado con proposito academico  Ingenieria de Soluciones con IA, 2026.*

*por Ricardo A. Galdames Soto*
