from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Extract the data from pdf
def load_pdf(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyMuPDFLoader)
    documents = loader.load()
    return documents


# Text splitter
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Embeddings
def download_hugging_face_embeddings(model):
    embeddings = HuggingFaceEmbeddings(model_name=model)
    return embeddings
