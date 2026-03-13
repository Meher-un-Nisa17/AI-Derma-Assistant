import os
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings # Local MiniLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# 1. API Key for Gemini (The "Brain" still needs internet)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Local MiniLM Embeddings (This runs on your CPU/RAM)
# This model is small (approx 400MB) and very reliable.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Setup Vector Database
DB_PATH = "./chroma_db"

# CRITICAL: If you are switching back from Google, you MUST delete the old folder
if os.path.exists(DB_PATH):
    try:
        # Test if the existing DB matches MiniLM (384 dimensions)
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        # Try a dummy search to trigger a dimension check
        vector_db.similarity_search("test", k=1)
    except Exception:
        print("Dimension mismatch detected (Google vs MiniLM). Deleting old database...")
        shutil.rmtree(DB_PATH)
        # Re-initialize after deletion
        loader = TextLoader("data/Data.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
else:
    loader = TextLoader("data/Data.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)

# 4. Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

class DermaBot:
    def ask(self, query: str):
        # Retrieve context from local MiniLM database
        docs = vector_db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are a professional Dermatology AI Assistant. 
        Answer the question using the context provided. 
        
        CONTEXT:
        {context}

        QUESTION: 
        {query}

        ANSWER:"""

        response = llm.invoke(prompt)
        return response.content

bot = DermaBot()