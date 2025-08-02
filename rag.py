import os
import typing
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate



import fitz

import numpy as np


os.environ["GOOGLE_API_KEY"]= r""
embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')

query_llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
generation_llm = query_llm



def get_text_from_pdf(pdf_path:str):
    """
    Docstring for get_text_from_pdf
    
    :param pdf_path: The path of the pdf file
    :type pdf_path: str
    """

    text = ""

    with fitz.open(pdf_path) as doc:
        for page in doc:
            full_text += page.get_text()
    return text

def split_text(text:str, chunk_size:int=1000, chunk_overlap:int=150):
    """
    Split the text using langchain.text_splitter.RecursiveCharacterTextSplitter

    :type text: str
    :type chunk_size: int
    :type chunk_overlap: int
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(text)
    return chunks

def embed_chunks_in_chroma(chunks, embeddings, persist_directory:str=None):

    vector_db = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_db


def multi_query_translation(queries: typing.List[str], num_queries=5):
    
    template = """You are an AI language model assistant. Your task is to generate {num_queries} 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | query_llm

    queries = [
        {"question": i,
         "num_queries":num_queries} 
        for i in queries
    ]

    response = chain.batch(queries)
    return response



def decomposition_query_translation(query):

    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""

    raise NotImplementedError("")

def step_back_query_translation(query: str):

    raise NotImplementedError("")
    #check fcc implementation once

def HyDE_query_translation(query: str, chunk_size: int):

    template = """Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth. The document size has be exactly {chunk_size} characters."""

    raise NotImplementedError("")



    



