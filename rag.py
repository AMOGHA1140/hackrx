import os, io
import typing
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import requests

import json
import fitz

import numpy as np


# os.environ["GOOGLE_API_KEY"]= r""
# embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')

# query_llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
# generation_llm = query_llm


def extract_text_from_web_pdf(pdf_url : str):    
    """
    Takes pdf's url as input and returns text of it.
    
    :pdf_url: URL to download the PDF.
    """

    try:
        response = requests.get(pdf_url, timeout=30)
        
        response.raise_for_status()
        pdf_in_memory = io.BytesIO(response.content)
        
        doc = fitz.open(stream=pdf_in_memory, filetype="pdf")
        full_text = ""

        for page_num, page in enumerate(doc):
            full_text += page.get_text()
        
        doc.close()
        return full_text

    except requests.exceptions.RequestException as e:
        return f"Error downloading the PDF: {e}"
    except Exception as e:
        return f"An error occurred during PDF processing: {e}"


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




#Functions for query translation
def multi_query_translation(queries: typing.List[str], query_llm, num_queries=5):
    """
    Docstring for multi_query_translation
    
    :param queries: List of queries to be translated. 
    :type queries: List[str]
    :param num_queries: Number of translations to make for each of the given queries. Default = 5
    :return: Returns the new queries as a 2D array of shape (n, num_queries), where n=len(queries)
    :rtype: List[List[str]]
    """
    
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

    responses = chain.batch(queries)
    
    output = [r.split('\n') for r in responses]
    return output

def decomposition_query_translation(queries, query_llm, num_queries=3):

    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Your output should be ONLY sub-questions, without any numbering, separated by only a newline \n
    Output ({num_queries} queries):"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | query_llm

    queries = [
        {
            'question': q, 
            "num_queries": num_queries
        }
        for q in queries
    ]

    responses = chain.batch(queries)
    output = [r.split('\n') for r in responses]

    return output

def step_back_query_translation(query: str, query_llm:ChatGoogleGenerativeAI):

    examples = [
    {
        "input":"Could the members of the Police perform lawful arrests?",
        "output":"What can the members of the Police do?"
    },
    {
        "input": "Jan Sindel's was born in what country?",
        "output": "What is Jan Sindel's personal history?",
    }
    ]

    example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    )

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        few_shot_prompt,
        ("user", "{question}"),
    ]
    )


    generate_queries_step_back = prompt | query_llm | StrOutputParser()
    generate_queries_step_back.invoke({"question":query})

    template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant. 
    # {normal_context}
    # {step_back_context}
    # Original question: {question}
    # Answer:"""

    chain = (
            {
                "normal_context": RunnableLambda(lambda x: x["query"]) | retriever,
                "step_back_context": generate_queries_step_back | retriever,
                "question": lambda x: x["query"],
            }
            | response_prompt
            | query_llm
            | StrOutputParser()
    )

    return chain.invoke({"question": query})

def HyDE_query_translation(queries: typing.List[str], chunk_size: int):
    """
    Generate a document containing a general answer for each of the given query, which can then be embedded later.

    
    queries: List[str] = list of queries to be translated

    chunk_size: int = approx size of the document, so that the original embedded documents and hypothetical docs are of similar length.

    """

    template = """Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth. The document size has be exactly {chunk_size} characters."""
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | query_llm

    queries = [
        {
            "query": q,
            "chunk_size":chunk_size
        }
        for q in queries
    ]

    response = chain.batch(queries)
    output = [r.content for r in response]
    
    return output

    



