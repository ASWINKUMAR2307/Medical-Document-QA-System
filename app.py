import streamlit as st
import os

# Updated imports to ensure compatibility and avoid module errors
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# (Replace with your actual API key)
os.environ["GOOGLE_API_KEY"] = ""

st.title("Medical Document Q&A System 🩺")
st.write("Upload a Medical PDF and ask questions based *only* on its content.")

# ==========================================
# 2. FILE UPLOAD & PROCESSING
# ==========================================
uploaded_file = st.file_uploader("Upload a Medical PDF", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file locally temporarily so PyPDFLoader can read it
    with open("temp_medical_doc.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    st.success("File uploaded successfully. Processing...")

    # Load the PDF
    loader = PyPDFLoader("temp_medical_doc.pdf")
    pages = loader.load()

    # Split the document into chunks
    # We use a chunk size of 1000 characters, with 200 characters of overlap so no sentences get cut in half.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)

    # ==========================================
    # 3. EMBEDDINGS & VECTOR DATABASE
    # ==========================================
    # We use a free, local HuggingFace model to turn text into numbers
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store the embeddings in a FAISS vector database (runs in memory)
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Create a retriever tool to fetch the top 3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # ==========================================
    # 4. LLM SETUP & STRICT PROMPT
    # ==========================================
    # Initialize the Google Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

    # This prompt strictly forces the LLM to only use the uploaded document
    prompt_template = """
    You are a medical assistant. Answer the question based ONLY on the context provided below.
    If the context does not contain the answer, say exactly: "I could not find the answer in the provided document."
    Do not use outside knowledge.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # ==========================================
    # 5. USER QUESTION & ANSWER GENERATION
    # ==========================================
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        with st.spinner("Searching document for answer..."):
            
            # 1. Retrieve the relevant chunks from FAISS
            relevant_docs = retriever.invoke(user_question)
            
            # 2. Combine the text from the retrieved chunks into one big string
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # 3. Format the prompt with the chunks and the user's question
            final_prompt = prompt.format(context=context_text, question=user_question)
            
            # 4. Get the answer from the LLM
            response = llm.invoke(final_prompt)
            
            # 5. Display the Answer
            st.subheader("Answer:")
            # Clean up the output to extract ONLY the text, ignoring the extra JSON/metadata
            if isinstance(response.content, list):
                answer_text = response.content[0].get("text", "")
            else:
                answer_text = response.content

            st.write(answer_text)
            
            # 6. Display the Source Chunks used to generate the answer
            st.subheader("Relevant Source Sections (Context Used):")
            for i, doc in enumerate(relevant_docs):
                st.info(f"**Chunk {i+1} (Page {doc.metadata.get('page', 'Unknown')}):**\n\n{doc.page_content}")