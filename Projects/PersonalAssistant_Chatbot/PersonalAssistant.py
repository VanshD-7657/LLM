
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

groq_api = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct', groq_api_key=groq_api)
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



# Load your text file
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("PersonalDoc.txt", encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    return Chroma.from_documents(split_docs, embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ==============================
# STEP 3: CONTEXTUALIZE QUESTION (History-aware)
# ==============================
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given chat history and latest user question, "
     "rewrite it as a standalone question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

# ==============================
# STEP 4: RETRIEVER STEP
# ==============================
def retrieve_with_history(inputs):
    # Convert follow-up question → standalone question
    standalone_question = contextualize_chain.invoke({
        "input": inputs["input"],
        "chat_history": inputs["chat_history"]
    })

    # Retrieve documents
    docs = retriever.invoke(standalone_question)

    # Combine docs into context
    context = "\n\n".join([doc.page_content for doc in docs])

    return {
        "context": context,
        "input": inputs["input"],
        "chat_history": inputs["chat_history"]
    }

# ==============================
# STEP 5: QA PROMPT
# ==============================
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a professional AI assistant who describes Vansh Dhall to others.

Your role is to answer questions ABOUT Vansh Dhall based ONLY on the provided context.

⚠️ Strict Rules:
- Always speak in THIRD PERSON (use "Vansh", "he", "his")
- NEVER speak as Vansh (do NOT use "I", "my", "me")
- Keep answers professional, human-like, and slightly storytelling
- Highlight his strengths, mindset, and journey naturally
- Do not exaggerate or add fake information

If the answer is not available in context, say:
"I don’t have enough information about Vansh Dhall on this, but based on what I know..."

Context:
{context}
"""),
    
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# ==============================
# STEP 6: FINAL RAG CHAIN
# ==============================
rag_chain = (
    retrieve_with_history
    | qa_prompt
    | llm
    | StrOutputParser()
)


# Streamlit app

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("""
<h1 style='text-align: center;'>
    🤖 Vansh's AI Assistant
</h1>
<p style='text-align: center;'>
    Ask anything about Vansh Dhall
</p>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.chat-container {
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)
# Show previous messages (ONLY last 2)
for message in st.session_state.chat_history[-2:]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# Input
user_input = st.chat_input("")

if user_input:
    try:
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history[-2:]
        })
    except Exception as e:
        response = "Sorry, something went wrong. Please try again."

    # Show new messages immediately
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        st.write(response)

    # Save to history
    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=response)
    ])