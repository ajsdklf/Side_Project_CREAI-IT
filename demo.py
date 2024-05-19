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
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import get_response_synthesizer
import requests 
from openai import OpenAI
import streamlit as st 
import numpy as np 
from dotenv import load_dotenv
import os 
from pathlib import Path
from llama_index.core import PromptTemplate


st.header('👩🏻‍🍼 영유아 선생님 도우미 👩🏻‍🍼')

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

client_llama = OpenAI_llama(model='gpt-4o-2024-05-13')

option = st.sidebar.selectbox(
    '어떤 질병을 가진 아이인가요?',
    ['ASD', 'ADHD', 'SDA']
)

SUMMARIZER_PROMPT = """
The user will describe a symptom or behavior which child is showing.
You have to summerize the user's input into two sentences.
Note that the cause of the sypmtom is one of the followings; ADHD, Autism, Seperation anxiety disorder.
"""

FINAL_ANSWER_PROMPT = """
As an [Experience Kindergarten Teacher], you are tasked with suggesting [3 steps] of [code of conduct] for the user based on their symptoms. The response should provide a structured plan in 3 steps for the  kindergarten teachers who are in need of help when encountered with children who have problem behavior.

Your answer should be in the form instructed:
-------------
[Form of Answer]
[Step1] : [Things user should do first to mitigate children's misbehavior.]
[Step2] : [Thinkgs user should do after conductiong step1.]
[Step3] : [Final steps user should take to fully mitigate toddler's misbehavior.]
-------------

The child’s misbehavior you have to consult right now is the following:
-------------
[Misbehavior of child]
{query_str}

When providing a consulting, take into account the following context. Following context includes information related to children’s current misbehavior. Context is the following:
[Context information]
{context_str}
-------------
"""

new_prompt_refine = """
The original query is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
{context_msg}
------------
Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.
Use proper Korean, and answer in the style of kindergarten teacher, gentle and enthusiastic.
Refined Answer:
"""

if option == 'ASD':
    storage_context = StorageContext.from_defaults(persist_dir="~/dataset/CREAI+IT_side_project/asd_index")
    vector_index = load_index_from_storage(storage_context)
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=3
    )
    answer = st.chat_input('아이가 보이는 행동을 말해주세요!')
    if answer:
        with st.spinner('🧑🏻‍🏫조언을 생성하고 있어요!🧑🏻‍🏫'):
            summarizer_input = f"Disease toddler is showing : {option} \n {answer}"
            
            summarized_answer = client.chat.completions.create(
                model='gpt-4o-2024-05-13',
                messages=[
                    {'role': 'system', 'content': SUMMARIZER_PROMPT},
                    {'role': 'user', 'content': summarizer_input}
                ]
            ).choices[0].message.content
            
            nodes = retriever.retrieve(summarized_answer)
            
            query_engine = vector_index.as_query_engine()
            prompts_dict = query_engine.get_prompts()
            
            FINAL_ANSWER_PROMPT = PromptTemplate(FINAL_ANSWER_PROMPT)
            new_prompt_refine = PromptTemplate(new_prompt_refine)
            
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": FINAL_ANSWER_PROMPT, "response_synthesizer:refine_template": new_prompt_refine}
                )
            
            response_synthesizer = get_response_synthesizer(response_mode='compact')
            
            response = query_engine.query(summarized_answer)
            
            with st.expander('조언 생성이 완료됐습니다!'):
                st.markdown(f"👨‍⚕️최종적인 조언👨‍⚕️: {response.response}")
                for node in nodes:
                    st.markdown(f"{node.text}란 문장이랑 유사한 행동을 보이네요!")
                st.markdown(summarized_answer)
                st.markdown(f"Called DataBase : {storage_context}")

if option == 'ADHD':
    storage_context = StorageContext.from_defaults(persist_dir="~/dataset/CREAI+IT_side_project/adhd_index")
    vector_index = load_index_from_storage(storage_context)
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=3
    )
    answer = st.chat_input('아이가 보이는 행동을 말해주세요!')
    if answer:
        with st.spinner('🧑🏻‍🏫조언을 생성하고 있어요!🧑🏻‍🏫'):
            summarizer_input = f"Disease toddler is showing : {option} \n {answer}"
            
            summarized_answer = client.chat.completions.create(
                model='gpt-4o-2024-05-13',
                messages=[
                    {'role': 'system', 'content': SUMMARIZER_PROMPT},
                    {'role': 'user', 'content': summarizer_input}
                ]
            ).choices[0].message.content
            
            nodes = retriever.retrieve(summarized_answer)
            
            response_synthesizer = get_response_synthesizer(response_mode='compact')
            
            query_engine = vector_index.as_query_engine()
            prompts_dict = query_engine.get_prompts()
            
            FINAL_ANSWER_PROMPT = PromptTemplate(FINAL_ANSWER_PROMPT)
            new_prompt_refine = PromptTemplate(new_prompt_refine)
            
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": FINAL_ANSWER_PROMPT, "response_synthesizer:refine_template": new_prompt_refine}
                )
            
            response = query_engine.query(summarized_answer)

            with st.expander('조언 생성이 완료됐습니다!'):
                st.markdown(f"👨‍⚕️최종적인 조언👨‍⚕️: {response.response}")
                for node in nodes:
                    st.markdown(f"{node.text}란 문장이랑 유사한 행동을 보이네요!")
                st.markdown(summarized_answer)
                
                st.markdown(f"Called DataBase : {storage_context}")

if option == 'SDA':
    storage_context = StorageContext.from_defaults(persist_dir="~/dataset/CREAI+IT_side_project/sda_index")
    vector_index = load_index_from_storage(storage_context)
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=3
    )
    answer = st.chat_input('아이가 보이는 행동을 말해주세요!')
    if answer:
        with st.spinner('🧑🏻‍🏫 조언이 생성되고 있어요! 🧑🏻‍🏫'):
            summarizer_input = f"Disease toddler is showing : {option} \n {answer}"
            
            summarized_answer = client.chat.completions.create(
                model='gpt-4o-2024-05-13',
                messages=[
                    {'role': 'system', 'content': SUMMARIZER_PROMPT},
                    {'role': 'user', 'content': summarizer_input}
                ]
            ).choices[0].message.content
            
            nodes = retriever.retrieve(summarized_answer)
            
            response_synthesizer = get_response_synthesizer(response_mode='compact')
            
            FINAL_ANSWER_PROMPT = PromptTemplate(FINAL_ANSWER_PROMPT)
            new_prompt_refine = PromptTemplate(new_prompt_refine)
            
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": FINAL_ANSWER_PROMPT, "response_synthesizer:refine_template": new_prompt_refine}
                )
            
            query_engine = vector_index.as_query_engine()
            prompts_dict = query_engine.get_prompts()
            
            response = query_engine.query(summarized_answer)

            with st.expander('조언 생성이 완료됐습니다!'):
                st.markdown(f"👨‍⚕️최종적인 조언👨‍⚕️: {response.response}")
                for node in nodes:
                    st.markdown(f"{node.text}란 문장이랑 유사한 행동을 보이네요!")
                st.markdown(summarized_answer)
                st.markdown(f"Called DataBase : {storage_context}")

