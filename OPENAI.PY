from langchain_huggingface import HuggingFaceEmbeddings

# Our small knowledge base (a list of sentences)
texts = [
    "Python is a programming language.",
    "Streamlit is used to build web apps.",
    "Dogs are friendly animals.",
    "Cats like to sleep a lot."
]

# Free, local embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Ready!")

from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_texts(texts, embeddings)

print("All texts are stored in the vector store!")