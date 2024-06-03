from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as PC
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_region = os.environ.get('PINECONE_REGION')
model = 'sentence-transformers/all-MiniLM-L6-v2'

extracted_data = load_pdf('data/')
text_chunks = text_split(extracted_data)
embeddings =  download_hugging_face_embeddings(model)

#pinecone initialization

pc = pinecone.Pinecone(api_key=pinecone_api_key)
index_name = 'llama-test'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine"
    )

docsearch = PC.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)