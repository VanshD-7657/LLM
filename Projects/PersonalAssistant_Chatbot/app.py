
# Step 1: Install dependencies
import os
import time
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


groq_api = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct', groq_api_key=groq_api)
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Step 2: Load and preprocess your document, create vectorstore, and retriever
@st.cache_resource
def load_vectorstore():
    try:
        import os
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(BASE_DIR, "PersonalDoc.txt")

        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = splitter.split_documents(docs)

        return Chroma.from_documents(split_docs, embeddings)

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# Step 3: CONTEXTUALIZE QUESTION (History-aware)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given chat history and latest user question, "
     "rewrite it as a standalone question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

# Step 4: Custom Retrieval Chain that uses contextualized question and retrieves relevant docs
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

# Step 5: QA Prompt (with strict instructions to avoid hallucination and maintain professionalism)  
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

Format the answer in a clean, structured, and visually appealing way using:

- Headings (###)
- Bullet points
- Emojis (limited but meaningful)
- Proper spacing

Avoid long paragraphs.
Make the response easy to read and professional.
If the answer is not available in context, say:
"I don’t have enough information about Vansh Dhall on this, but based on what I know..."

Context:
{context}
"""),
    
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Final RAG Chain
rag_chain = (
    retrieve_with_history
    | qa_prompt
    | llm
    | StrOutputParser()
)


# Streamlit app

# Session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Style
st.markdown("""
<style>
img {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# Image
col1, col2, col3 = st.columns([1,2,1])
with col2:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "avatar_img.png")
    st.image(image_path, caption="Vansh's Personal AI Assistant", use_container_width=True)

st.markdown("<p style='text-align: center;'>Ask anything about Vansh Dhall</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)



# Empty state
if not st.session_state.chat_history:
    st.markdown("### 👋 Start a conversation...")

# Streaming function (improved)
def stream_response(response):
    words = response.split()
    chunk_size = 3
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size]) + " "
        time.sleep(0.05)

# Show previous messages (ONLY last 2)
for message in st.session_state.chat_history[-2:]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="🧑"):
            st.write(message.content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message.content)

# Chat input (clean)
user_input = st.chat_input("")

# Handle new input
if user_input:
    with st.chat_message("user", avatar="🧑"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            time.sleep(0.5)
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history[-2:]
            })

        # Step 1: Show streaming effect
        message_placeholder = st.empty()
        streamed_text = ""

        for chunk in stream_response(response):
            streamed_text += chunk
            message_placeholder.write(streamed_text)

        # Step 2: Final clean formatted output
        message_placeholder.write(response)

    # Save history
    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=response)
    ])

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
# Footer
st.markdown("---")
st.caption("🚀 Built by Vansh Dhall • Personal AI Assistant")