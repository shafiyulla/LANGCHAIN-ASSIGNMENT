from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate an embedding
vector = embeddings.embed_query("I love India")

print("Vector length:", len(vector))
print("First 3 values:", vector[:3])


# Function to calculate cosine similarity
def similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0

    return np.dot(a, b) / denominator


# Generate embeddings for three sentences
v1 = embeddings.embed_query("I like dogs")
v2 = embeddings.embed_query("I love puppies")
v3 = embeddings.embed_query("45645")

# Calculate similarities
sim1 = similarity(v1, v2)
sim2 = similarity(v1, v3)

# Print results
print("\nCosine Similarity Results")
print("-" * 30)
print(f"Dogs vs Puppies : {sim1:.4f}")
print(f"Dogs vs Numbers : {sim2:.4f}")