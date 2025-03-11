import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)


st.title("AI for Personalized Nutrition")
st.caption("Your personal nutritionist for diet and nutrition goals")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("Model Name", "llama3")  # Valid model name
    model_version = st.text_input("Model Version", "1.0.0")

    st.divider()
    st.markdown("### Chat Configuration")
    st.markdown(""" 
                - 🤖 User Inputs
                - 👩‍💼 AI Responses
                - 👨‍💼 System Messages
                - 📝 Chat Prompts
                """)

    st.divider()
    st.markdown("Built with [Langchain](https://langchain.com) | [Ollama](https://ollama.com)")
    st.header("Developed By: codeX")

# Function to generate AI response
def generate_ai_response(prompt_chain, llm_engine, user_input):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({"input": user_input})

# Function to build prompt chain
def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "User":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "Ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# Initialize chat engine
llm_engine = ChatOllama(
    model=model_name,
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt (focused on nutrition advice)
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a certified nutritionist. Provide personalized dietary recommendations "
    "based on user input about their health goals, preferences, and constraints. "
    "Ask clarifying questions if needed."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "Ai", "content": "Hi! I'm your AI nutritionist. How can I help you today?"}
    ]

# Chat interface
chat_container = st.container()

with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_query = st.chat_input("Type your message...")
    if user_query:
        # Add user message to log
        st.session_state.message_log.append({"role": "User", "content": user_query})
        
        # Generate response
        with st.spinner("Generating AI response..."):
            prompt_chain = build_prompt_chain()
            ai_response = generate_ai_response(prompt_chain, llm_engine, user_query)
            
            # Add AI response to log
            st.session_state.message_log.append({"role": "Ai", "content": ai_response})
            
            # Refresh UI
            st.rerun()
