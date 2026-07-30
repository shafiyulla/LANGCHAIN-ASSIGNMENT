from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------------
# STEP 1 : Load PDF
# -----------------------------------

loader = PyPDFLoader("mybook.pdf")
documents = loader.load()

loader = PyPDFLoader("myasm.pdf")
documents = loader.load()

print("=" * 50)
print("PDF Loaded Successfully")
print("Number of Pages:", len(documents))

# -----------------------------------
# STEP 2 : Split Documents
# -----------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("=" * 50)
print("Chunks Created:", len(chunks))

# -----------------------------------
# STEP 3 : Create Embeddings
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("=" * 50)
print("Embedding Model Loaded")

# -----------------------------------
# STEP 4 : Create FAISS Vector Store
# -----------------------------------

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("=" * 50)
print("FAISS Vector Store Created")

# -----------------------------------
# STEP 5 : Save FAISS
# -----------------------------------

vector_store.save_local("mystore")

print("Vector Store Saved in mystore/")

# -----------------------------------
# STEP 6 : Load FAISS
# -----------------------------------

db = FAISS.load_local(
    "mystore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("=" * 50)
print("Vector Store Loaded")

# -----------------------------------
# STEP 7 : Create Retriever
# -----------------------------------

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("=" * 50)
print("Retriever Ready")

# -----------------------------------
# STEP 8 : Ask Question
# -----------------------------------

question = input("\nEnter your question: ")

# -----------------------------------
# STEP 9 : Retrieve Documents
# -----------------------------------

results = retriever.invoke(question)

print("\n" + "=" * 50)
print("Retrieved Documents")
print("=" * 50)

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("-" * 50)
    print(doc.page_content)