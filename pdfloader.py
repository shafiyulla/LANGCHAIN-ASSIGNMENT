from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("mybook.pdf")
pages = loader.load()
print(pages[0].page_content)
