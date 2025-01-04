import os
import time
import re
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PDFMinerLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import ServerlessSpec
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Cargando la variable de entorno           - OK
load_dotenv(find_dotenv(), override=True) 


# Cargando el documento                     - OK
def load_document(filename, loader = 'pdfminer'):
    '''
    filename: str
    loader: 'pdfminer' or 'pypdf'
    '''
    pdf_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pdf_files', filename) 

    if loader == 'pdfminer':
        loader = PDFMinerLoader(pdf_file_path, concatenate_pages= True)
        # loader = PyPDFLoader(pdf_file_path)
        document = loader.load()[0]
    elif loader == 'pypdf':
        loader = PyPDFLoader(pdf_file_path)
        document = loader.load()
    
    print('Document loaded!')

    return document

# Funcion para dividir el texto en chunks   - OK
def text_into_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 512,
                                            chunk_overlap = 50,
                                            separators=[""])
    chunks = text_splitter.split_documents(documents = document)
    print('Text splitted into chunks!')
    
    return chunks

# Cargando el modelo de embeddings          - OK
def get_embedding_model():
    embedding = OpenAIEmbeddings()
    return embedding

# Creamos un cliente de Pinecone            - OK
def load_pinecone(index_name):

    pc = Pinecone()

    # Creamos indice en Pinecone o lo cargamos si ya existe - OK
    if index_name not in pc.list_indexes().names():
        
        for i in pc.list_indexes().names():
            pc.delete_index(i)
            print('Index deleted')
            print('-' * 50)
        
        print(f'Creating index {index_name}')
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
        print('Index created!')
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
            print('Index ready!')

    else:
        index = pc.Index(index_name)
        print(f'Index {index_name} loaded!')
        if index.describe_index_stats().total_vector_count > 0:
            index.delete(delete_all=True)
            print('Content deleted!')

# Creamos y cargamos la base vectorial en Pinecone      - OK
def load_vector_store(chunks, index_name):

    vector_store = PineconeVectorStore.from_documents(
        documents= chunks,
        embedding=get_embedding_model(),
        index_name=index_name
    )

    vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=get_embedding_model())
    print('Vector store loaded!')
    
    return vector_store


# PRUEBA DE FUNCIONAMIENTO
index_name = 'tenders-index'
document = load_document('document_1.pdf', loader = 'pypdf')
chunks = text_into_chunks(document)
load_pinecone(index_name)
vector_store = load_vector_store(chunks, index_name)

retriever = vector_store.as_retriever(search_type = 'similarity')
time.sleep(10)
relevant_chunks = retriever.invoke('¿Cual es el objeto del expediente?')

# prompt template
PROMPT_TEMPLATE = '''
Eres un asistente para preguntar y responder. Usa los contextos para responder la pregunta. Si no sabes que responder, solo di que no sabes. No te inventes las respuestas.

{context}
-----
Answer the question based on the above context: {question}
'''
content_text = "\n\n---\n\n".join([chunk.page_content for chunk in relevant_chunks])

# Create prompt
prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=content_text, question='¿Cual es el objeto del expediente?')

llm = ChatOpenAI(model='gpt-3.5-turbo')

response = llm.invoke(prompt)

print(response.content)

