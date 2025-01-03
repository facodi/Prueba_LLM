import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv(find_dotenv(), override=True) # Cargando la variable de entorno

def get_embeddings_model():
    cloud_embeddings = OpenAIEmbeddings()
    return cloud_embeddings

def load_document():
    pdf_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pdf_files', 'document_1.pdf') 
    loader = PDFMinerLoader(pdf_file_path)
    main_document = loader.load()
    return main_document[0]

def split_document(document):
    # splitted_documents = []
    chunks = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 0,
        separators=[""]).split_text(text=document.page_content)
    return chunks

vectorstore_documents = []
def vectorize_document(chunks):
    for chunk in chunks:
        vectorstore_documents.append(Document(page_content=chunk))

    return FAISS.from_documents(documents = vectorstore_documents, embedding=get_embeddings_model())

variable = 'suministro'
def extract_top_documents(vectorstore, prompt_request, top_k, fetch_k):
    return [document for document in vectorstore.similarity_search(query=prompt_request, k=top_k, fetch_k=fetch_k)]

if __name__ == '__main__':
    document = load_document()
    splitted = split_document(document)
    vectors = vectorize_document(splitted)
    print("Document vectors into chunks")
    rag_query = f"Dime todo lo relacionado con {variable}"
    resultado = extract_top_documents(vectors, rag_query, 5, 5)
    print("Top documents extracted")
