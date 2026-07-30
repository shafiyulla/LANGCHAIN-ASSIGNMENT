from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load document
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

print("Documents Loaded:", len(documents))

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Chunks Created:", len(chunks))

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector store
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save locally
vectorstore.save_local("faiss_db")

print("✅ FAISS database saved successfully!")