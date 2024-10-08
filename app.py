import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from the uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  
    return text.strip()  

# Split the text into manageable chunks for processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vector store using Google Generative AI embeddings and save it locally
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load the conversational chain with a specific prompt template
def get_conversational_chain():
    prompt_template = """
    You are an AI assistant specialized in providing in-depth answers about a PDF document. 
    When a user asks a question, analyze the provided context and any previously answered questions to deliver comprehensive, detailed responses. 
    Aim to include relevant examples, explanations, and connections to enhance the user's understanding. 
    If you cannot find the answer in the context, be transparent about it.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Load the summarization chain for Gemini
def get_summary_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = load_summarize_chain(model, chain_type="map_reduce")
    return chain

# Handle the user's input when Ollama is selected
def ollama_input(user_question, docs):
    # Combine the documents into a single string for context
    context = "\n".join([doc.page_content for doc in docs])  # Assuming docs is a list of document objects
    prompt = f"Context:\n{context}\n\nQuestion: {user_question}\nAnswer:"

    model = Ollama(model="llama3.2")  
    response = model(prompt)  

    return response

# Handle the user's input (question or summarization request)
def user_input(user_question, chat_history, model_choice):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    if model_choice == "Ollama":
        output = ollama_input(user_question, docs)
    else:  # Assume "Gemini"
        if "summarize" in user_question.lower():
            chain = get_summary_chain()
            response = chain({"input_documents": docs}, return_only_outputs=True)
            output = response["output_text"]
        else:
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            output = response["output_text"]

    chat_history.append({"question": user_question, "answer": output})
    return chat_history

# Main function to handle the UI and interaction logic
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")

    # Initialize session state for chat history and model choice
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "Select Model"  # or a default value

    # Dropdown to select the model
    st.sidebar.title("Choose a model")
    st.session_state.model_choice = st.sidebar.selectbox(
        "Select a model:",
        options=["Gemini", "llama 3.2"],  # Add your models here
        index=0  # Default index
    )

    # User input area using text_area to prevent enter submission
    user_question = st.text_input("Ask a Question or Request a Summary of the PDF Files", key="user_question")

    if st.button("Ask"):
        if user_question:
            st.session_state.chat_history = user_input(user_question, st.session_state.chat_history, st.session_state.model_choice)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files...", accept_multiple_files=True)
        if st.button("Submit"):         
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing complete!")
                    else:
                        st.error("No text found in the uploaded PDFs.")
            else:
                st.error("Please upload at least one PDF file.")

    if st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history):
            st.write(f"**You:** {chat['question']}")
            st.write(f"**Chatbot:** {chat['answer']}")

if __name__ == "__main__":
    main()
