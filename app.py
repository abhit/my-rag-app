import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain

# --- Page Config ---
st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
st.title("🦙 Local RAG with Ollama")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    # No API Key needed!
    model_name = st.selectbox("Choose Local Model", ["mistral", "llama3", "gemma"], index=0)
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


# --- Functions ---

@st.cache_resource
def process_pdf(uploaded_file, model_name):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Load and Split
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create Embeddings & Vector Store using OLLAMA
    # Note: Using the same model for embeddings as generation
    embeddings = OllamaEmbeddings(
        model=model_name,
        base_url="http://ollama:11434"  # <--- POINT TO DOCKER SERVICE NAME
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="local_rag_db"
    )
    return vectorstore


def get_conversational_chain(vectorstore, model_name):
    # Initialize Local LLM (Ollama)
    llm = ChatOllama(model=model_name,
                     temperature=0,
                     base_url="http://ollama:11434" # <--- POINT TO DOCKER SERVICE NAME
    )

    retriever = vectorstore.as_retriever()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain


# --- Main Logic ---

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_file:

    with st.spinner("Processing PDF locally... (This might take a moment)"):
        # Pass the selected model name to the processor
        vectorstore = process_pdf(uploaded_file, model_name)
        chain = get_conversational_chain(vectorstore, model_name)

    if prompt := st.chat_input("Ask a question..."):
        # Show User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking (Locally)..."):
                response = chain.invoke({
                    "question": prompt,
                    "chat_history": st.session_state.chat_history
                })

                answer = response['answer']
                st.session_state.chat_history.append((prompt, answer))
                st.markdown(answer)

        # Save Assistant Message
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Please upload a PDF to begin.")