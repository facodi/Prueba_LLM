from typing import List, Optional
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


def load_secrets(secret_path: str) -> None:
    '''
    Function to load secrets from a .env file
    '''
    load_dotenv(secret_path)
    logger.info('Secrets loaded successfully')
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    

def document_loader() -> List[Document]:
    '''
    Function to load a pdf document into a list of documents
    '''
    file_path = "src/data/pliego_clausulas.pdf"
    loader = PyMuPDFLoader(file_path)    
    docs = loader.load()
    logger.info('Document loaded successfully')
    
    return docs


def split_chunks_document(
    doc: List[Document],
    chunk_size=1000,
) -> List[Document]:
    '''
    Function to split a list of documents into chunks
    '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=200,
                                                   length_function=len,
                                                   separators=["\n\n", "\n"],
    )
    splitted_documents = text_splitter.split_documents(doc)
    logger.info(f'Document splitted successfully. {len(splitted_documents)} documents created')
    return splitted_documents

def create_vector_store(splitted_documents: List[Document]) -> FAISS:
    '''
    Function to create a vector store from a list of documents
    '''
    load_secrets(secret_path='src/.env')
    vector_store = FAISS.from_documents(documents=splitted_documents,
                                        embedding=OpenAIEmbeddings(),)
    logger.info('Vector store created successfully')
    return vector_store



pdf = document_loader()
splitted_docs = split_chunks_document(pdf)
vector_store = create_vector_store(splitted_docs)
logger.info(f'Process finished. Vector store.')
retriever = vector_store.as_retriever()

class GeneralExtractor(BaseModel):
    """Extracted information from a tender document"""
    
    organismo_licitador: Optional[str] = Field(default=None, description="Organismo que emite el documento para el proceso de licitacion")
    comunidad_autonoma: Optional[str] = Field(default=None, description="Comunidad autonoma donde se realiza la licitacion")
    provincia: Optional[str] = Field(default=None, description="Provincia donde se realiza la licitacion")
    fecha_publicacion: Optional[str] = Field(default=None, description="Fecha de publicacion del documento")   
    plazo_presentacion: Optional[str] = Field(default=None, description="Plazo de presentacion de ofertas")
    

class FinancialExtractor(BaseModel):
    """Extracted information from a tender document"""
    
    precio_maximo: Optional[float] = Field(default=None, description="Precio maximo estimado del contrato")
    

class FormulaExtractor(BaseModel):
    """Extracted formula from a tender document"""
    
    formula_economica: Optional[str] = Field(default=None, description="Cálculo que se utiliza para evaluar las ofertas de los licitadores en función de su precio.")


class HospitalesExtractor(BaseModel):
    """Extracted information from a tender document"""
    
    hospitales_adheridos: List[Optional[str]] = Field(default=None, description="Hospitales o centros adheridos al contrato")
    potenciales_hospitales: List[Optional[str]] = Field(default=None, description="Hospitales o centros potenciales para adherirse al contrato")


class ProductosExtractor(BaseModel):
    """Extracted information from a tender document"""
    
    productos_adquiridos: List[Optional[str]] = Field(default=None, description="Productos o medicamentos a suministrar en el contrato")
    

class PenalidasExtractor(BaseModel):
    """Extracted information from a tender document"""
    
    penalidades: Optional[str] = Field(default=None, description="Penalidades por incumplimiento del contrato")

class VigenciaExtractor(BaseModel):
    """Extracted information from a tender document"""
    
    vigencia_contrato: Optional[str] = Field(default=None, description="Vigencia del contrato")
    prorrogas: Optional[str] = Field(default=None, description="Vigencia maxima prorrogas del contrato")

llm = ChatOpenAI(
    model_name="gpt-4o-mini-2024-07-18",
    temperature=0,
)
template = """
Eres un algortimo avanzado de extraccion de informacion.
Tu funcion es extraer la informacion proporcionada en el siguiente contexto, si no encuentras la informacion retorna un mensaje de no especificado.:

{context}
"""
prompt = ChatPromptTemplate.from_template(template)

def parse_document(docs: List[Document]) -> str:
    '''
    Function to parse a list of documents into a string.
    '''
    return "\n".join([doc.page_content for doc in docs])

chain_general = (
    {"context": retriever | parse_document}
    | prompt
    | llm.with_structured_output(GeneralExtractor)
    
)

chain_financial = (
    {"context": retriever | parse_document, "question": RunnablePassthrough()}
    | prompt
    | llm.with_structured_output(FinancialExtractor)
    
)

chain_formula = (
    {"context": retriever | parse_document, "question": RunnablePassthrough()}
    | prompt
    | llm.with_structured_output(FormulaExtractor)
    
)

chain_hospitales = (
    {"context": retriever | parse_document, "question": RunnablePassthrough()}
    | prompt
    | llm.with_structured_output(HospitalesExtractor)
)

chain_productos = (
    {"context": retriever | parse_document, "question": RunnablePassthrough()}
    | prompt
    | llm.with_structured_output(ProductosExtractor)
)

chain_penalidades = (
    {"context": retriever | parse_document, "question": RunnablePassthrough()}
    | prompt
    | llm.with_structured_output(PenalidasExtractor)
)

chain_vigencia = (
    {"context": retriever | parse_document, "question": RunnablePassthrough()}
    | prompt
    | llm.with_structured_output(VigenciaExtractor)
)
    
general = chain_general.invoke('Extrae la informacion acerca del organismo licitador, comunidad autonoma, provincia, fecha de publicacion y plazo de presentacion del documento')
valor_contrato = chain_financial.invoke("Extrae la informacion del valor estimado del contrato")
vigencias = chain_vigencia.invoke("Extrae la informacion del plazo de duracion del contrato y las prorrogas")
formula_economica = chain_formula.invoke("Extrae la informacion de la formula de evaluacion economica")
hospitales = chain_hospitales.invoke("Extrae la informacion de cuales son los hospitales adheridos y potenciales")
suministro = chain_productos.invoke("Extrae la informacion de cuales son los productos o medicamentos a suministrar")
penalidades = chain_penalidades.invoke("Extrae la informacion de potenciales penalidades por incumplimiento del contrato")

extract = (general.dict()
           | valor_contrato.dict()
           | vigencias.dict()
           | formula_economica.dict()
           | hospitales.dict()
           | suministro.dict()
           | penalidades.dict()
          )



print(extract)