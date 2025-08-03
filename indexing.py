import os
import typing

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

import chromadb



class MultiRepresentationIndexing:

    _summarizing_template = "Summarize the following document:\n\n{doc}"

    def __init__(
        self, 
        summarizing_model:ChatGoogleGenerativeAI, 
        embedding_model: GoogleGenerativeAIEmbeddings
    ):
        # self.summarizing_model = summarizing_model
        self.summarizing_chain = ChatPromptTemplate.from_template(self._summarizing_template) | summarizing_model
        self.embedding_model = embedding_model
        

    def _summarize(self, documents:typing.List[str]):

        queries = [
            {
                "doc": d
            }
            for d in documents
        ]

        responses = self.summarizing_chain.batch(queries)
        output = [str(i) for i in responses]
        return output
    
    def _embed_documents(self, documents:typing.List[str]):

        vectors = self.embedding_model.embed_documents(documents)
        return vectors
    
    def forward(self, documents:typing.List[str]):

        summary = self._summarize(documents)
        vectors = self._embed_documents(summary)

        return vectors
    


        

