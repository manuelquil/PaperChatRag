from langchain.document_loaders.pdf import PyPDFDirectoryLoader # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
from langchain.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
#from dotenv import load_dotenv # Importing dotenv to get API key from .env file
import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

import gradio as gr


def load_documents(DOCUMENTS_PATH):
    """
    Load PDF documents from the specified directory using PyPDFDirectoryLoader.
    Returns
    List of Document objects: Loaded PDF documents represented as Langchain
                                                            Document objects.
    """
    # Initialize PDF loader with specified directory
    document_loader = PyPDFDirectoryLoader(DOCUMENTS_PATH) 
    # Load PDF documents and return them as a list of Document objects
    return document_loader.load()

def split_text(documents: list[Document]):
    """
    Split the text content of the given list of Document objects into smaller chunks.
    Args:
      documents (list[Document]): List of Document objects containing text content to split.
    Returns:
      list[Document]: List of Document objects representing the split text chunks.
    """
    # Initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=300, # Size of each chunk in characters
      chunk_overlap=100, # Overlap between consecutive chunks
      length_function=len, # Function to compute the length of the text
      add_start_index=True, # Flag to add start index to each chunk
    )

    # Split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)

    return chunks # Return the list of split text chunks

def save_to_chroma(chunks: list[Document], CHROMA_PATH):
    """
    Save the given list of Document objects to a Chroma database.
    Args:
    chunks (list[Document]): List of Document objects representing text chunks to save.
    Returns:
    None
    """ 
    # Clear out the existing database directory if it exists
    #  if os.path.exists(CHROMA_PATH):
    #    shutil.rmtree(CHROMA_PATH)

    # Create a new Chroma database from the documents using OpenAI embeddings
    db = Chroma.from_documents(
      chunks,
      HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
      persist_directory=CHROMA_PATH
    )

    # Persist the database to disk
    db.persist()
    
def generate_data_store(DOCUMENTS_PATH, CHROMA_PATH):
    """
    Function to generate vector database in chroma from documents.
    """
    documents = load_documents(DOCUMENTS_PATH) # Load documents from a source
    chunks = split_text(documents) # Split documents into manageable chunks
    gr.Info("Start Creating Chroma Knowledge DataBase")
    save_to_chroma(chunks, CHROMA_PATH) # Save the processed data to a data store
    gr.Info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def query_rag(query_text, history, CHROMA_PATH, LANGUAGE, k_SIMILARITY, MODEL):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and Ollama.
    Args:
      - query_text (str): The text to query the RAG system with.
    Returns:
      - formatted_response (str): Formatted response including the generated text and sources.
      - response_text (str): The generated response text.
    """

    MODEL = f"{MODEL}"
    LANGUAGE = f"{LANGUAGE}"
    CHROMA_PATH = f"{CHROMA_PATH}"
    
    # YOU MUST - Use same embedding function as before
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Prepare the database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieving the context from the DB using similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=k_SIMILARITY)  #k=3 (para Español) k=1 Ingles 

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])

    LANGUAGE = LANGUAGE.split("['")[1]
    LANGUAGE = LANGUAGE.split("']")[0]
    LANGUAGE = str(LANGUAGE)

    MODEL = MODEL.split("['")[1]
    MODEL = MODEL.split("']")[0]
    MODEL = str(MODEL)
    
    # Create prompt template using context and query text
    if LANGUAGE == 'Spanish':
        PROMPT_TEMPLATE = """
                            Responder la pregunta basandose unicamente en el siguiente documento y no en otra cosa:
                            {context}
                            - -
                            Responder la pregunta basandose unicamente en el contexto del documento y no en otra cosa: {question}
                          """
    if LANGUAGE=='English':
        PROMPT_TEMPLATE = """
                            Answer the question based only on the following context:
                            {context}
                            - -
                            Answer the question based on the above context: {question}
                          """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize OLlama chat model
    model = ChatOllama(model=MODEL, stream=True)

    full_responce = " "
      
    for chunk in model.stream(prompt):
      responce = chunk.content
      full_responce += responce 
      yield full_responce

def upload_file(file, CHROMA_PATH):    
    
    UPLOAD_FOLDER = f"{CHROMA_PATH}"
    if not os.path.exists(UPLOAD_FOLDER):    
        os.mkdir(UPLOAD_FOLDER)    
    shutil.copy(file, UPLOAD_FOLDER)    
    gr.Info("File Uploaded!!!")

def run_app():
    with gr.Blocks() as demo:
        gr.Markdown('# Q&A RAG Bot Models')   
        with gr.Tab("Knowledge Bot"):
            with gr.Row():
              CHROMA_PATH = gr.Textbox(placeholder=r"e.g. C:\chroma", label="Knowledge Vector DataBase Path")
              DOCUMENTS_PATH = CHROMA_PATH
              upload_button = gr.UploadButton("Click to Upload a File")    
              upload_button.upload(upload_file, inputs = [upload_button, CHROMA_PATH])
              btn = gr.Button("Create Chroma DataBase")
              btn.click(fn=generate_data_store, inputs=[DOCUMENTS_PATH, CHROMA_PATH])
            with gr.Row():
              LANGUAGE = gr.CheckboxGroup(["Spanish", "English"], label="Language")
              MODEL = gr.CheckboxGroup(["mistral", "llama3.1"], label="Model")
              k_SIMILARITY = gr.Slider(1, 5, step=1, label="k Similarity Parameter")
            gr.ChatInterface(query_rag, additional_inputs=[CHROMA_PATH, LANGUAGE, k_SIMILARITY, MODEL])
    demo.queue().launch(debug = True)

if __name__ == "__main__":      
    run_app()
    


    
