from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the PDF
loader = PyPDFLoader("mybook.pdf")
documents = loader.load()

# Create the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split the PDF into chunks
chunks = splitter.split_documents(documents)

# Print the chunks
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)

    from langchain_huggingface import HuggingFaceEmbeddings

