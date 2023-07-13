from dotenv import load_dotenv
import os
import pandas as pd
import textract
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain 
import streamlit as st

chat_history = []
chunks = None

load_dotenv()

# Load the OpenAI API key from the environment variable
if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    print("OPENAI_API_KEY is not set")
    exit(1)
else:
    print("OPENAI_API_KEY is set")

 

@st.cache_data
def make_restful_call():
    pdf_file = './tokyo_api_reference_7-12-2023.pdf'
    loader = PyPDFLoader(pdf_file)
    chunks = loader.load_and_split() 
    embeddings = OpenAIEmbeddings()
 
    return FAISS.from_documents(chunks, embeddings)  

def main(): 

    st.set_page_config(page_title="ServiceNow Chatbot")

    db = make_restful_call()
    # Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.2), db.as_retriever())

    # st.header("SNowGPT")
    st.write("The ServiceNow LLM Knowledge base")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message[0])
        with st.chat_message("assistant"):
            st.markdown(message[1])
            
    # React to user input
    if prompt := st.chat_input("Query the 'Tokyo API Reference'"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        with st.spinner(text="In progress..."): 
            result = qa({"question": prompt, "chat_history": st.session_state.chat_history})
            st.session_state.chat_history.append((prompt, result['answer']))

            response = f"{result['answer']}"
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response) 
     

if __name__ == "__main__":
    main()
