from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from dotenv import load_dotenv
import os
from src.prompt import *

load_dotenv()
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_region = os.environ.get('PINECONE_REGION')
model = os.environ.get('MODEL')
index_name = os.environ.get('INDEX_NAME')

embeddings =  download_hugging_face_embeddings(model)
app = Flask(__name__)

docsearch = Pinecone.from_existing_index(index_name, embeddings)
# query = 'What is biology?'
# docs = docsearch.similarity_search(query, k=3)

promt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
chain_type_kwargs = {"prompt": promt}

llm = CTransformers(
    model='TheBloke/Llama-2-7B-Chat-GGML',
    model_type='llama',
    config={
        'max_new_tokens': 512,
        'temperature': 0.8
    }
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response: ,", result['result'])
    return str(result['result'])


if __name__ == '__main__':
    app.run(debug=True)
    #  app.run(host='', port=3000, debug=True)