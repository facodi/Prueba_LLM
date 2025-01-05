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
def load_enviorment():
    '''
    Load the APIs
    '''
    load_dotenv(find_dotenv(), override=True)
    print('Enviorment loaded!')

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

    else:
        index = pc.Index(index_name)
        print(f'Index {index_name} loaded!')
        if index.describe_index_stats().total_vector_count > 0:
            index.delete(delete_all=True)
            print('Content deleted!')

    while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    print('Index ready!')

# Creamos y cargamos la base vectorial en Pinecone      - OK
def load_vector_store(chunks, index_name):

    # load_pinecone(index_name)     # Probar despues
    pc = Pinecone()
    index = pc.Index(index_name)
    index.delete(delete_all=True)
    print('Content deleted.')

    vector_store = PineconeVectorStore.from_documents(
        documents= chunks,
        embedding=get_embedding_model(),
        index_name=index_name
    )

    while not index.describe_index_stats().total_vector_count > 0:
            time.sleep(1)
    # vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=get_embedding_model())
    print('Vector store loaded!')
    
    return vector_store

def getting_context(vector_store, question, search_type = 'similarity'):
    retriever = vector_store.as_retriever(search_type = search_type)
    docs_context = retriever.invoke(question)
    return docs_context

def generating_prompt(docs_context, question):
    # prompt template
    PROMPT_TEMPLATE = '''
    Eres un asistente para preguntar y responder. Usa los contextos para responder la pregunta. Si no sabes que responder, solo di que no sabes. No te inventes las respuestas.

    {context}
    -----
    Answer the question based on the above context: {question}
    '''

    content_text = "\n\n---\n\n".join([chunk.page_content for chunk in docs_context])

    # Create prompt
    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=content_text, question= question)

    return prompt

def load_model():
    model = ChatOpenAI(model='gpt-3.5-turbo')
    
    return model

def tenders_contracts(filename, question , loader= 'pdfminer'):
    '''
    Principal function
    '''
    print('Loading the APIs...')
    load_enviorment()
    print('Loading the document...')
    document = load_document(filename=filename, loader= loader)
    print('Splitting text into chunks...')
    chunks = text_into_chunks(document)
    print('Creating the vectorstore...')
    vector_store = load_vector_store(chunks, index_name)
    print('Obtaining retriever...')
    docs_context = getting_context(vector_store, question)
    prompt = generating_prompt(docs_context= docs_context, question= question)
    model = load_model()
    response = model.invoke(prompt)

    return response.content


if __name__ == '__main__':
    index_name = 'tenders-index'
    filename = 'document_1.pdf'
    question = '¿Cuál es el objeto del expediente?'
    resultado = tenders_contracts(filename=filename, question=question, loader = 'pypdf')
    print(resultado)
    print('TERMINADO')