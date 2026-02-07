import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant."),
        ("human", "Question: {question}")
    ]
)

# Function to generate response
def generate_response(question, llm_name, temperature, max_tokens):
    model = ChatGroq(
        model_name=llm_name,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=st.secrets["GROQ_API_KEY"],
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    return chain.invoke({"question": question})


# ---------------- UI ---------------- #

st.set_page_config(page_title="Groq Q&A Chatbot", page_icon="🤖")
st.title("🤖 Q & A Chatbot with Groq")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    llm = st.selectbox(
        "Choose Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"]
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
    max_tokens = st.slider("Max Tokens", 64, 2048, 512)

# User input
query = st.text_input("What's your question?")

# Submit button
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = generate_response(
                query, llm, temperature, max_tokens
            )
        st.success("Answer")
        st.write(response)