# 🩺 Medical Document Q&A System (RAG)

This project is a simple Retrieval-Augmented Generation (RAG) web application built for my internship. It allows users to upload medical PDF documents and ask questions. The system extracts the text, generates vector embeddings, and uses a Large Language Model (LLM) to answer questions **strictly based on the uploaded document**.

## 🚀 Features
* **PDF Processing:** Extracts text from uploaded medical PDF documents.
* **Smart Chunking:** Splits text into overlapping chunks to maintain context without losing critical medical terms.
* **Local Embeddings & Vector Search:** Uses HuggingFace's free `all-MiniLM-L6-v2` model and FAISS for fast, locally-hosted vector retrieval.
* **Accurate QA Generation:** Integrates Google's Gemini LLM to generate precise answers. 
* **Hallucination Prevention:** The system is explicitly prompted to state *"I could not find the answer in the provided document"* if the information is missing.
* **Source Transparency:** Displays the exact text chunks used to generate the answer so the user can verify the source.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit
* **Orchestration:** LangChain
* **Embeddings:** HuggingFace (`sentence-transformers`)
* **Vector Database:** FAISS (In-memory)
* **LLM:** Google Gemini API (`gemini-3.6-flash`)

## ⚙️ Setup and Installation Instructions

### Prerequisites
* Python 3.8 or higher installed on your system.
* A free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).

"C:\Users\Aswim\Downloads\images.png"

