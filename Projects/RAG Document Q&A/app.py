import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import time
groq_api = os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
llm = ChatGroq(model='openai/gpt-oss-120b',api_key=groq_api)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based on provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question:{input}
    """
)
st.title("RAG Context Based Q&A")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_vector_embeddings():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFLoader('LLM.pdf')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


user_prompt = st.text_input("Enter your query from the LLM Book")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("vector Database is ready")

if user_prompt:
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})

    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


    start = time.process_time()
    response = chain.invoke( user_prompt)
    print(f"Response time = {time.process_time()-start}")

    st.write(response)

    # with a streamlit expander
    with st.expander("Document similarity search"):
        docs = retriever.invoke(user_prompt)
        for doc in docs:
            st.write(doc.page_content)