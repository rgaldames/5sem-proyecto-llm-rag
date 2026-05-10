"""
agent.py  Agente Funcional Veterinario (Evaluación Parcial 2)
=============================================================
Implementa un agente funcional usando la API moderna de LangChain 1.2+
basada en LangGraph (create_agent con tool-calling loop).

Arquitectura:
  - LangChain 1.2.15 / LangGraph 1.1.8
  - Patrón: Tool-Calling Loop (equivalente moderno al ReAct)
  - Herramientas: consulta (RAG), escritura (JSONL), razonamiento (reglas)
  - Memoria: corto plazo (buffer de mensajes) + largo plazo (embeddings JSON)

IL2.1  Herramientas de consulta, escritura y razonamiento
IL2.2  Memoria de corto y largo plazo
IL2.3  Planificación y toma de decisiones adaptativas
IL2.4  Documentación técnica y orquestación de componentes

Flujo de Orquestación:
  Usuario
     Recuperar contexto largo plazo (MemoryStore)
     Construir historial + system prompt contextual
     Agente (LLM + herramientas en loop)
         LLM decide si llamar herramienta
         [search_clinical_db | write_visit_summary | analyze_symptoms]
         Observación  LLM decide continuar o responder
     Respuesta final
     Guardar en corto plazo (buffer) + largo plazo (embeddings)
"""

import os
import sys
from dotenv import load_dotenv

# -- 1. CONFIGURACIÓN DE ENTORNO ---------------------------------------------
load_dotenv()
GITHUB_PAT_TOKEN = os.getenv("GITHUB_PAT_TOKEN")
if not GITHUB_PAT_TOKEN:
    raise ValueError("Error: No se encontro GITHUB_PAT_TOKEN en el archivo .env")

os.environ["OPENAI_API_KEY"] = GITHUB_PAT_TOKEN

# -- 2. IMPORTACIONES --------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain.agents import create_agent          # API nueva LangChain 1.2+
from langchain_core.messages import HumanMessage, AIMessage

from tools import make_search_tool, write_visit_summary, analyze_symptoms
from memory_store import MemoryStore
from short_term_memory import ShortTermMemory

# -- 3. RUTAS DE ARCHIVOS ----------------------------------------------------
CSV_PATH      = "./veterinary_clinical_data.csv"
CHROMA_DB_DIR = "./chroma_db_local"
MEMORY_PATH   = "./memories.json"

# -- 4. SYSTEM PROMPT DEL AGENTE ---------------------------------------------
# Prompt simplificado para evitar que el LLM entre en loops de tool-calling.
# El modelo decide POR SÍ MISMO si necesita o no llamar una herramienta.
SYSTEM_PROMPT = """Eres VetBot, un asistente de inteligencia artificial para una clinica veterinaria.
Ayudas a duenos de mascotas con consejos de cuidado, informacion clinica y orientacion.

REGLAS:
- Jamas recetes medicamentos ni emitas diagnosticos medicos definitivos.
- Si hay sintomas graves, recomienda ir a urgencias veterinarias.
- Usa las herramientas disponibles SOLO cuando sea necesario para responder mejor.
- Si puedes responder directamente sin herramientas, hazlo sin llamarlas.
- Responde siempre en espanol, de forma clara y amigable.

Herramientas disponibles (usarlas solo si aportan valor real a la respuesta):
- search_clinical_db: busca en historiales clinicos de la BD interna
- analyze_symptoms: evalua nivel de urgencia de sintomas descritos
- write_visit_summary: guarda un resumen al FINAL de una consulta importante"""


def initialize_components():
    """
    Inicializa todos los componentes del agente de forma modular.

    Returns:
        Tupla con (agente_compilado, memoria_corto_plazo, memoria_largo_plazo)
    """
    print("[1/5] Configurando modelo de lenguaje (gpt-4o-mini)...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        base_url="https://models.inference.ai.azure.com"
    )

    print("[2/5] Cargando modelo de embeddings local (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("[3/5] Indexando base de conocimiento clinico en ChromaDB...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: No se encontro el archivo CSV: {CSV_PATH}")
        sys.exit(1)

    loader = CSVLoader(file_path=CSV_PATH, encoding="utf-8")
    documents = loader.load()
    documents = documents[:50]
    print(f"     {len(documents)} registros clinicos indexados.")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print("[4/5] Inicializando sistema de memoria dual...")
    # Corto plazo: buffer de ventana deslizante (últimas 5 interacciones)
    short_term = ShortTermMemory(window_size=5)
    # Largo plazo: almacenamiento semántico persistente entre sesiones
    long_term = MemoryStore(embeddings, path=MEMORY_PATH)
    print("     Corto plazo: ConversationBufferWindowMemory (k=5)")
    print("     Largo plazo: MemoryStore JSON + embeddings cosine similarity")

    print("[5/5] Construyendo agente con herramientas (LangChain 1.2 create_agent)...")
    # Herramienta de consulta vinculada al vectorstore activo
    search_tool = make_search_tool(vectorstore)

    # Las 3 herramientas del agente (IL2.1)
    tools = [
        search_tool,          # CONSULTA: RAG sobre historiales clinicos
        analyze_symptoms,     # RAZONAMIENTO: deteccion de urgencia por reglas
        write_visit_summary,  # ESCRITURA: persistencia de consultas
    ]

    # Agente moderno LangChain 1.2+ (equivalente a ReAct, basado en LangGraph)
    # El LLM llama herramientas en loop hasta que no necesita mas informacion
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    print("     Agente compilado con 3 herramientas.")

    return agent, short_term, long_term


def run_agent(agent, short_term: ShortTermMemory, long_term: MemoryStore,
              user_input: str) -> str:
    """
    Ejecuta el agente integrando memoria dual en cada invocación.

    Estrategia de planificación adaptativa (IL2.3):
    1. Recuperar memorias de largo plazo relevantes para el contexto actual
    2. Construir lista de mensajes con historial de corto plazo
    3. Invocar el agente (el LLM decide qué herramientas usar y cuándo)
    4. Persistir la interacción en ambos niveles de memoria

    Args:
        agent: El agente compilado (LangGraph CompiledStateGraph).
        short_term: Buffer de memoria de corto plazo (sesión actual).
        long_term: Store de memoria de largo plazo (persistente).
        user_input: Mensaje del usuario.

    Returns:
        Respuesta final del agente como string.
    """
    # Paso 1: Recuperar memorias relevantes de largo plazo (IL2.2)
    past_memories = long_term.retrieve(user_input, top_k=3)
    long_term_context = ""
    if past_memories:
        long_term_context = "\n".join(f"- {m['text']}" for m in past_memories)

    # Paso 2: Construir el input para el agente.
    # Se pasa SOLO el mensaje actual del usuario (el agente gestiona su propio estado).
    # El contexto de largo plazo se inyecta dentro del texto para evitar
    # conflictos con el loop interno de LangGraph.
    if long_term_context:
        enriched_input = (
            f"{user_input}\n\n"
            f"[Contexto de sesiones anteriores relevante:]\n{long_term_context}"
        )
    else:
        enriched_input = user_input

    # Paso 3: Invocar el agente con un unico HumanMessage
    try:
        result = agent.invoke({"messages": [HumanMessage(content=enriched_input)]})
        # La respuesta final esta en el ultimo AIMessage sin tool_calls
        output_messages = result.get("messages", [])
        response = ""
        for msg in reversed(output_messages):
            msg_type = msg.__class__.__name__
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", [])
            if msg_type == "AIMessage" and content and not tool_calls:
                response = content
                break
        if not response:
            # Fallback: cualquier mensaje con contenido
            for msg in reversed(output_messages):
                if getattr(msg, "content", ""):
                    response = str(msg.content)
                    break
        if not response:
            response = "No pude generar una respuesta. Por favor intenta de nuevo."

    except Exception as e:
        response = (
            f"Ocurrio un error al procesar tu consulta: {str(e)}\n"
            "Por favor, intenta reformular tu pregunta."
        )

    # Paso 4: Guardar en memoria de corto plazo (sesion actual)
    short_term.save_turn(user_input, response)

    # Paso 5: Guardar resumen en memoria de largo plazo (persistente entre sesiones)
    try:
        combined = f"Consulta: {user_input[:150]} | Respuesta: {response[:150]}"
        long_term.add_memory(
            combined,
            metadata={"role": "interaction", "turn": short_term.turn_count}
        )
    except Exception:
        pass  # No interrumpir el flujo por errores de memoria

    return response


def print_status(short_term: ShortTermMemory):
    """Imprime el estado actual de la memoria del agente."""
    stats = short_term.get_summary_stats()
    print(
        f"\n>> Estado de Memoria: "
        f"Turno #{stats['total_turns_session']} | "
        f"Buffer: {stats['messages_in_buffer']}/{stats['window_size']*2} msgs | "
        f"{'[LLENO]' if stats['buffer_full'] else '[ACTIVO]'}"
    )


def main():
    """Punto de entrada principal del agente funcional veterinario."""
    print("=" * 65)
    print("  AGENTE VETERINARIO FUNCIONAL - Evaluacion Parcial 2")
    print("=" * 65)
    print("  Framework  : LangChain 1.2 + LangGraph (create_agent)")
    print("  Herramientas: Consulta | Escritura | Razonamiento")
    print("  Memoria    : Corto Plazo (buffer) + Largo Plazo (semantico)")
    print("=" * 65)
    print()

    agent, short_term, long_term = initialize_components()

    print()
    print("Agente listo! (Escribe 'salir' para terminar | 'estado' para ver memoria)\n")
    print("-" * 65)

    # -- Bucle principal de interaccion --------------------------------------
    while True:
        try:
            user_input = input("\nUsuario: ").strip()

            # Comandos especiales
            if not user_input:
                continue
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("\nAgente apagandose. Hasta pronto!")
                break
            if user_input.lower() == "estado":
                print_status(short_term)
                continue
            if user_input.lower() == "historial":
                hist = short_term.get_history_as_text()
                print(f"\nHistorial reciente:\n{hist if hist else '(vacio)'}")
                continue

            # Ejecutar el agente
            print("\n[Agente procesando...]\n")
            response = run_agent(agent, short_term, long_term, user_input)

            print("\n" + "=" * 65)
            print("VetBot responde:")
            print("-" * 65)
            print(response)
            print("=" * 65)

        except KeyboardInterrupt:
            print("\n\nSesion interrumpida. Hasta luego!")
            break
        except Exception as e:
            print(f"\nError inesperado: {e}")
            print("Intenta nuevamente o escribe 'salir' para terminar.")


if __name__ == "__main__":
    main()
