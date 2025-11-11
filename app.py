import os
import gradio as gr

# Importaciones de LangChain y componentes
from langchain_community.document_loaders import UnstructuredWordDocumentLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------------
# 1. VERIFICACIÓN DE CLAVE API
# La clave GEMINI_API_KEY se carga automáticamente desde los 'Secrets' del Space.
# -----------------------------------------------------------
if 'GEMINI_API_KEY' not in os.environ:
    # Esto detendrá la ejecución si la clave no está presente en el entorno de HF.
    raise ValueError("GEMINI_API_KEY no encontrada. Por favor, configura tu clave API en Hugging Face Secrets.")
print("✅ Clave API cargada desde el entorno.")


# -----------------------------------------------------------
# 2. CARGA Y PROCESAMIENTO DEL DOCUMENTO DOCX (RAG Setup)
# -----------------------------------------------------------

file_path = "reglamento.docx" # Ruta relativa en el Space

try:
    # Carga robusta de DOCX (UnstructuredWordDocumentLoader)
    loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
    pages = loader.load()
    
    if not pages:
        raise Exception("El cargador no extrajo contenido. El archivo DOCX puede estar vacío o corrupto.")
        
    print(f"✅ DOCX cargado exitosamente. Total de elementos extraídos: {len(pages)}")

except FileNotFoundError:
    print(f"🚨 ERROR CRÍTICO: No se encontró el archivo '{file_path}'. Súbelo a tu Space.")
    raise 
except Exception as e:
    print(f"🚨 ERROR AL CARGAR: {e}")
    raise
    
# División (Chunking) con alta superposición
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], # Separadores robustos
    chunk_size=2000, 
    chunk_overlap=400 # Overlap aumentado
)
docs = text_splitter.split_documents(pages)
print(f"✅ Texto dividido en {len(docs)} trozos con alta superposición.")


# 3. CREACIÓN DE EMBEDDINGS Y BASE DE DATOS FAISS
print("Creando embeddings y Base de Datos FAISS...")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Crear la Base de Datos con FAISS
db = FAISS.from_documents(docs, embedding_model)

# Definir el Retriever - Máxima recuperación
retriever = db.as_retriever(search_kwargs={"k": 20}) # k=20 para máxima recuperación
print("✅ Base de datos FAISS creada y Retriever listo (k=20).")


# 4. CONSTRUCCIÓN DE LA CADENA RAG (LCEL)

# Inicializar el modelo Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.2, 
    google_api_key=os.environ['GEMINI_API_KEY'] 
)

# Definir el Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", 
      "ERES UN EXPERTO LEGAL UNIVERSITARIO y tu ÚNICA fuente de respuesta es el CONTEXTO PROPORCIONADO. Responde formalmente y con precisión. Si el contexto NO contiene la respuesta, DEBES responder textualmente: 'Lo siento, no encuentro esa información específica en el reglamento universitario. Por favor, consulta con la oficina correspondiente.'"),
    ("human", "CONTEXTO RECUPERADO:\n---\n{context}\n---\n\nPregunta del Usuario: {question}"),
])

# Función para formatear documentos
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Construir la Cadena RAG
qa_chain = (
    # 1. Recuperación
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # 2. Generación
    | prompt
    | llm
    | StrOutputParser()
)


# 5. FUNCIÓN Y INTERFAZ DE GRADIO
def rag_chat(user_input, chat_history):
    # Ignoramos chat_history, pero Gradio lo requiere en la firma de la función
    try:
        answer = qa_chain.invoke(user_input)
        return answer
    except Exception as e:
        print(f"Error durante la invocación de la cadena RAG: {e}")
        return "Hubo un error al procesar tu solicitud. Por favor, revisa los logs del servidor."


# Crear y Lanzar la Interfaz de Gradio
# ------------------------------------------------------------------
# CAMBIOS ESTÉTICOS: Aplicamos un tema monocromático más limpio.
# ------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=rag_chat,
    chatbot=gr.Chatbot(
        height=450, 
        label="Asistente de Reglamento Universitario",
        # Añadir un logo o icono de tu universidad (opcional: usando el parámetro avatar_images)
    ),
    # Título más descriptivo y centrado
    title="📚 Asistente de Consulta: Reglamento Oficial", 
    
    # Usamos un tema diferente con un color primario para un toque moderno
    theme=gr.themes.Monochrome(primary_hue="blue", secondary_hue="cool"),
    
    description="Bienvenido al Asistente RAG. Haz preguntas específicas sobre cualquier **artículo, proceso o norma** contenida en el documento oficial (reglamento.docx).",
    
    # Ejemplos de preguntas revisados para ser más representativos
    examples=[
        "¿Cuáles son los requisitos de matrícula para un estudiante nuevo?", 
        "¿Qué establece el reglamento sobre el procedimiento de apelación de notas?", 
        "Detalles sobre las sanciones por fraude académico."
    ],
)

demo.queue()