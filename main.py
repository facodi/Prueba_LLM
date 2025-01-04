import os
import time
import re
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import ServerlessSpec
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Cargando la variable de entorno           - OK
load_dotenv(find_dotenv(), override=True) 

# Cargando el documento                     - OK
pdf_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pdf_files', 'document_1.pdf') 
loader = PDFMinerLoader(pdf_file_path)
document = loader.load()[0]

# Cargando el modelo de embeddings          - OK
embedding = OpenAIEmbeddings()

# Funcion para dividir el texto en chunks   - OK
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 0,
    separators=[""])
chunks = text_splitter.split_text(document.page_content)

# Creamos un cliente de Pinecone            - OK
pc = Pinecone()

index_name = 'tenders-index'

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
    index.delete(delete_all=True)
    print('Content deleted!')

# Creamos la base vectorial en Pinecone      - OK
vector_store = PineconeVectorStore.from_texts(
    texts= chunks,
    embedding=embedding,
    index_name=index_name
)

# Cargamos la base vectorial en Pinecone     - OK 
vstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

# ------------------------------------------------------------------------------------------------------
# PRUEBA DE FUNCIONAMIENTO
# ------------------------------------------------------------------------------------------------------

query = '¿Cual es el objeto del expediente?'

message = [
    # SystemMessage(content = 'You are a helpull asistant, Your name is Wall-E'),
    HumanMessage(content = query)
]

llm = ChatOpenAI(model='gpt-3.5-turbo')

docs = vstore.similarity_search(query=query, k=3)

chain = load_qa_chain(
    llm = llm,
    chain_type = 'stuff'
)

response = chain.invoke(input_documents = docs , question = query)

print(response)


print('Done!')