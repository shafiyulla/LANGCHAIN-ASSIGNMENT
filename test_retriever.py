from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Sample documents
texts = [
    "Python is a programming language used for AI.",
    "Streamlit is used to build web applications quickly.",
    "LangChain helps connect AI models with your own data.",
    "Dogs are friendly and loyal animals.",
    "FORD car is hihghly reliable and efficient."
]

# Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector store
vector_store = FAISS.from_texts(
    texts=texts,
    embedding=embeddings
)

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

# User question
question = "High speed car?"

# Retrieve relevant documents
results = retriever.invoke(question)

# Display results
print("=" * 50)
print("Question:")
print(question)
print("=" * 50)

print("\nRetrieved Documents:\n")

for i, doc in enumerate(results, start=1):
    print(f"Document {i}")
    print(doc.page_content)
    print("-" * 50)