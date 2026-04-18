import os
import sys

# ==== Configuración de Variables de Entorno ====
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Llave obtenida desde el archivo .env (Token de GitHub Models)
GITHUB_PAT_TOKEN = os.getenv("GITHUB_PAT_TOKEN")
if not GITHUB_PAT_TOKEN:
    raise ValueError("¡Error de seguridad! No se encontró GITHUB_PAT_TOKEN dentro del archivo .env")

os.environ["OPENAI_API_KEY"] = GITHUB_PAT_TOKEN

# La trazabilidad está comentada para evitar mensajes de error de "Failed to ingest..."
# os.environ["LANGCHAIN_TRACING_V2"] = "true" 
# os.environ["LANGCHAIN_PROJECT"] = "Veterinary_RAG_Agent"

# ===============================================
# Importación de Módulos LangChain
# ===============================================
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma 
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Rutas de Archivos
CSV_PATH = "./veterinary_clinical_data.csv"
CHROMA_DB_DIR = "./chroma_db_local"

def main():
    print("=== Inicializando el Agente de Inteligencia Artificial para la Veterinaria ===")

    # 1. Configuración de Modelos e Ingnesta de Datos
    # Usamos la base_url apuntando al servicio de inferencia de GitHub porque se ingresó un GitHub PAT
    # Si vas a usar una API KEY estándar de OpenAI, simplemente elimina el parámetro `base_url`
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3, # Baja temperatura para respuestas consistentes
        base_url="https://models.inference.ai.azure.com" 
    )

    # Reemplazamos la API Limitada por modelos 100% locales
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Cargando datos clñinicos desde {CSV_PATH}...")
    
    if not os.path.exists(CSV_PATH):
        print(f"¡Error! No se encontró el archivo CSV en la ruta especificada: {CSV_PATH}")
        sys.exit(1)

    # 2. Carga del CSV
    loader = CSVLoader(file_path=CSV_PATH, encoding="utf-8")
    documents = loader.load()
    
    # Limitar a las primeras 100 filas para evitar Rate Limits de la API gratuita
    documents = documents[:100]
    print(f"Se tomaron las primeras {len(documents)} filas de historiales médicos para la base de conocimiento.")

    # 3. Creación de la Base de Datos Vectorial (Chroma)
    print("Creando Base Vectorial (ChromaDB) y procesando embeddings...")
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    
    # Configuramos el Retriever para recuperar los 3 contextos más relevantes que alimentarán al LLM
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Diseño del Prompt Ético y Contextual (IE2, IE4)
    # Este prompt actúa como "Sistema" dictando el comportamiento restringido y ético del agente.
    template = """Eres un experto y amigable asistente de inteligencia artificial para una clínica veterinaria. 
Tu objetivo es responder de forma clara y sencilla las preguntas de los dueños de mascotas. 
Utiliza ÚNICAMENTE la siguiente información clínica extraída de la base de datos de la clínica para basar tus respuestas (ignorando si la información no es relevante para la pregunta).

INFORMACIÓN CLÍNICA (Contexto):
{context}

REGLAS ÉTICAS Y DE COMPORTAMIENTO:
1. NUNCA diagnostiques médicamente a un animal ni recetes un tratamiento definitivo o medicamento.
2. Si el usuario te menciona síntomas graves del animal o riesgos a su salud, debes recomendar explícitamente y con urgencia visitar a un profesional veterinario físico. Tú eres sólo un consejero virtual y orientador de cuidado.
3. Brinda consejos de cuidado básico relacionados con perros, gatos y otras mascotas de forma amigable.

Pregunta del Cliente: {question}
Respuesta del Asistente Veterinario:"""
    
    custom_rag_prompt = PromptTemplate.from_template(template)

    # Función auxiliar para combinar el contenido de los documentos recuperados
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Construcción de Arquitectura y Cadena de Ejecución (LCEL - LangChain Expression Language) (IE5)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    print("\n¡Chatbot Veterinario en Línea! (Escribe 'salir' o 'exit' para terminar)\n")
    print("-" * 50)
    
    # 6. Interfaz de Chat Simple en Consola
    while True:
        try:
            user_input = input("\nUsuario: ")
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("Chatbot apagándose... ¡Hasta luego!")
                break
            
            if not user_input.strip():
                continue

            print("Pensando...")
            # Aquí es donde ocurre todo el RAG
            response = rag_chain.invoke(user_input)
            
            print("\n🐶🐾 Chatbot Veterinario dice:")
            print(response)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nChatbot apagándose...")
            break
        except Exception as e:
            print(f"\nOcurrió un error al procesar tu solicitud: {e}")
            print("Verifica si tus cuotas de uso (API Tokens) están activas o si la base de datos está accesible.")

if __name__ == "__main__":
    main()
