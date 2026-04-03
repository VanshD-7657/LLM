# RAG Q&A Conversation with PDF including Chat History
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnablePassthrough,RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')



# Streamlit app
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload PDF and chat with thier content")

groq_api = st.text_input('Enter your Groq API Key',type='password')

if groq_api:
    llm = ChatGroq(model='openai/gpt-oss-120b',api_key=groq_api)
    session_id= st.text_input("Session ID", value='default_session')

    # Managing chat history
    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files = st.file_uploader('Choose A PDF file', type='pdf',accept_multiple_files=True)

    # Process uploaded PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf = f'./temp.pdf'
            with open(temppdf,'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and create embeddings for the documnets
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever = vectorstore.as_retriever()

            
        contexualize_prompt = (
            """
            Given a chat history and the latest user question
            which might refrence context in the chat history
            formulate a standalone question which can be understood
            without the chat histoy. Do not answer the question,
            just reformulate it if needed and otherwise return it as is.
            """
        )
        q_prompt = ChatPromptTemplate.from_messages([
            ('system', contexualize_prompt),
            MessagesPlaceholder('chat_history'),
            ('human',"{input}")
        ])
        # This is used to join page content from each document
        def format_docs(docs):
         return "\n\n".join(doc.page_content for doc in docs)
        
        contextualize_chain = (
            {
                "context": retriever | format_docs,
                "input": RunnablePassthrough(),
                'chat_history': RunnablePassthrough()
            }
            | q_prompt
            | llm
            | StrOutputParser()
        )    

        # Answer question
        system_prompt = (
            """
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, say that you don't know. Use three sentences 
            maximum and keep the answer concise.
            \n\n
            {context}
            """
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ('system', system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(
                    retriever.invoke(x["input"])
                )
            )
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='output'
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {'input': user_input},
                config={
                    'configurable':{'session_id':session_id}
                }
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response)
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter Your Groq API Key")