from llama_index.core import Prompt 
from llama_index.llms.openai import OpenAI as OpenAI_llama
from llama_index.core.llms import ChatMessage 
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings 
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TextNode 
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever 
from llama_index.core import DocumentSummaryIndex
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
)
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexEmbeddingRetriever,
)
from llama_index.core import StorageContext, load_index_from_storage
from sklearn.metrics.pairwise import cosine_similarity
import requests 
from openai import OpenAI
import streamlit as st 
import numpy as np 
from dotenv import load_dotenv
import os 
from pathlib import Path

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

client_llama = OpenAI_llama(model='gpt-4o-2024-05-13')

data_path_adhd = Path("~/dataset/CREAI+IT_side_project/adhd_coaching").expanduser()

documents_adhd = SimpleDirectoryReader(str(data_path_adhd)).load_data()
text_splitter = SentenceSplitter(
    chunk_size=100,
    chunk_overlap=15
)

nodes_adhd = text_splitter.get_nodes_from_documents(documents_adhd)
adhd_vector_index = VectorStoreIndex(nodes_adhd)
adhd_vector_index.storage_context.persist('~/dataset/CREAI+IT_side_project/adhd_index')

data_path_asd = Path("~/dataset/CREAI+IT_side_project/asd_coaching").expanduser()

documents_asd = SimpleDirectoryReader(str(data_path_asd)).load_data()
nodes_asd = text_splitter.get_nodes_from_documents(documents_asd)
asd_vector_index = VectorStoreIndex(nodes_asd)
asd_vector_index.storage_context.persist('~/dataset/CREAI+IT_side_project/asd_index')

data_path_sda = Path('~/dataset/CREAI+IT_side_project/sda_coaching').expanduser()

documents_sda = SimpleDirectoryReader(str(data_path_sda)).load_data()
nodes_sda = text_splitter.get_nodes_from_documents(documents_sda)
sda_vector_index = VectorStoreIndex(nodes_sda)
sda_vector_index.storage_context.persist('~/dataset/CREAI+IT_side_project/sda_index')