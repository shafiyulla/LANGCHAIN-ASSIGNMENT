from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample.txt")
documents = loader.load()

print("Number of documents:", len(documents))
print("Content:")
print(documents[0].page_content)