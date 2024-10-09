# Chat-With-PDF using Streamlit and Langchain

## 👀 Project Overview
This project is a PDF chatbot application built using Streamlit, LangChain, and multiple LLMs (Large Language Models) both local and API-driven. 
Users can upload PDF documents and engage in a conversational interface that leverages advanced language models such as Google Gemini and Ollama (Llama 3.2). 
The application is designed for flexibility and scalability, enabling the processing of multiple document types and providing intelligent responses based on document content.

## 🧠 Project Structure

> `chatpdf/`:

- **app.py:** The main application file that handles the Streamlit UI and interaction logic.
- **requirements.txt:** Lists the Python packages required to run the application.
- **.env:** Stores environment variables such as API keys (not included in version control).

## 📦 Technology Stack

- **Streamlit:** A fast way to create web applications for machine learning and data science projects.
- **LangChain:** Provides a framework to work with LLMs for document processing and conversational tasks.
- **FAISS:** A library for efficient similarity search and clustering of dense vectors.
- **Ollama:** Facilitates the use of local models like Llama 3.2 for LLMs.
- **Google Generative AI:** Used for embedding text and generating conversational responses.
- **PyPDF2:** Library for extracting text from PDF files.
- **AWS Deployment:** AWS services are leveraged for hosting and deploying the application.

## ⚒ Key Code Components

- **get_pdf_text(pdf_docs):** Extracts text from uploaded PDF files.
- **get_text_chunks(text):** Splits extracted text into manageable chunks for processing.
- **get_vector_store(text_chunks):** Creates a vector store using Google Generative AI embeddings and saves it locally.
- **get_conversational_chain():** Loads the conversational chain with a specific prompt template for answering user questions.
- **ollama_input(user_question, docs):** Handles user questions using the Ollama model.
- **user_input(user_question, chat_history, model_choice):** Processes user input and retrieves relevant document context to generate answers.

## 🛠 Running the Project Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/aswinjak/ChatBot-with-PDF.git
   ```

2. Navigate to the project directory:
   ```bash
   cd chatpdf
   ```

3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows use `venv\Scripts\activate`
   ```

4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables:
Create a `.env` file and add your Google API key:
   ```bash
   GOOGLE_API_KEY=YOUR_API_KEY
   ```

6. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## 🔐 Deploying on a Cloud Platform

To deploy on services like AWS, ensure you have the necessary setup for Python and any environment variables configured in the hosting environment.

## 💡 References

- LangChain Documentation
- Ollama
- Google Generative AI
- Streamlit Documentation

  
