import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import StateGraph, END
from typing import TypedDict

# STREAMLIT
st.set_page_config(page_title="LangGraph Math + Data Assistant")
st.title(" Math Solver + Data Assistant (LangGraph)")

groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

if not groq_api_key:
    st.warning("Enter API Key")
    st.stop()

# Model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

# STATE
class AgentState(TypedDict):
    question: str
    route: str
    answer: str

# ROUTER
router_prompt = ChatPromptTemplate.from_template("""
Decide the type of question:
- math → numerical or calculation
- search → factual or Wikipedia
- reasoning → logic/explanation

Question: {question}

Answer ONLY: math, search, or reasoning
""")

router_chain = router_prompt | llm | StrOutputParser()

def router(state: AgentState):
    route = router_chain.invoke({"question": state["question"]}).strip().lower()
    return {"route": route}

# TOOLS

# 1.Math
math_prompt = ChatPromptTemplate.from_template(
    "Solve step by step:\n{question}"
)
math_chain = math_prompt | llm | StrOutputParser()

def solve_math(state: AgentState):
    result = math_chain.invoke({"question": state["question"]})
    return {"answer": result}

# 2.Wikipedia
wiki = WikipediaAPIWrapper()

def search_wiki(state: AgentState):
    result = wiki.run(state["question"])
    return {"answer": result}

# 3.Reasoning
reason_prompt = ChatPromptTemplate.from_template(
    "Explain logically:\n{question}"
)
reason_chain = reason_prompt | llm | StrOutputParser()

def reasoning(state: AgentState):
    result = reason_chain.invoke({"question": state["question"]})
    return {"answer": result}

# Graph
graph = StateGraph(AgentState)

graph.add_node("router", router)
graph.add_node("math", solve_math)
graph.add_node("search", search_wiki)
graph.add_node("reasoning", reasoning)

# Routing logic
def route_decision(state: AgentState):
    if "math" in state["route"]:
        return "math"
    elif "search" in state["route"]:
        return "search"
    else:
        return "reasoning"

graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "math": "math",
        "search": "search",
        "reasoning": "reasoning"
    }
)

# End connections
graph.add_edge("math", END)
graph.add_edge("search", END)
graph.add_edge("reasoning", END)

graph.set_entry_point("router")

app = graph.compile()

# UI
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me math or general questions 😊"}
    ]

# Clear Chat Button
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask me anything 😊"}
        ]
        st.rerun()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            result = app.invoke({
                "question": user_input
            })

            answer = result["answer"]

            st.write(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })