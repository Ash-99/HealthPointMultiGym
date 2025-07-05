import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load API key from .env file for security
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- Configuration ---
SOURCE_DOC_PATH = "data/gyminfo.pdf"
VECTOR_STORE_PATH = "vectordatabase" # The folder where the store will be saved

def ingest_data():
    """
    Loads data from a source PDF, processes it, and stores it in a FAISS vector store.
    """
    print("Loading documents...")
    # 1. Load the document
    loader = PyPDFLoader(SOURCE_DOC_PATH)
    documents = loader.load()
    if not documents:
        print("Could not load any documents. Please check the file path.")
        return

    print(f"Loaded {len(documents)} document(s).")

    # 2. Split the document into smaller chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    # 3. Create embeddings
    # This converts the text chunks into numerical vectors
    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Create a FAISS vector store and save it
    # This indexes the document chunks for fast retrieval
    print("Creating and saving vector store...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_STORE_PATH)
    print(f"Vector store created and saved at: {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    ingest_data()