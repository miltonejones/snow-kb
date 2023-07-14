from dotenv import load_dotenv
import os
import pandas as pd
import textract
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain 

from blank import blank_page
import streamlit as st

load_dotenv()
 
def convert_file_name(file_name): 
  file_name = file_name.split(".")[0] 
  words = file_name.split("_") 
  words = [word.capitalize() for word in words] 
  return " ".join(words)
 
def get_document_list():
  folder_path = './src'
  file_object = {}

  for file_name in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, file_name)):
      normalized_key = convert_file_name(file_name)
      file_object[normalized_key] = './src/' + file_name
 
  return file_object



# Load the OpenAI API key from the environment variable
if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
  print("OPENAI_API_KEY is not set")
  exit(1) 

 
if "pdf_file" not in st.session_state:
  st.session_state.pdf_file = './src/client_api_reference.pdf'
    

@st.cache_data
def load_source_document(pdf_file): 
  with st.spinner(text="Loading "+ pdf_file +"..."): 
    loader = PyPDFLoader(pdf_file)
    chunks = loader.load_and_split() 
    embeddings = OpenAIEmbeddings()

    return FAISS.from_documents(chunks, embeddings)  

def main(): 
  st.set_page_config(page_title="ServiceNow Chatbot")

  db = load_source_document(st.session_state.pdf_file) 
 
  key_value_pairs = get_document_list()

  if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
      
  if "selected_option" not in st.session_state:
    st.session_state.selected_option = ''
      
  if "llm_temparature" not in st.session_state:
    st.session_state.llm_temparature = 0.2

  llm = OpenAI(temperature=st.session_state.llm_temparature)
      
  # Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
  qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever())

  # build sidebar 
  with st.sidebar:

    st.header("ServiceNow Chatbot") 
    st.write("The ServiceNow LLM Knowledge base") 
    
    # Display radio buttons and handle selection
    selected_option = st.radio('Select a document:', list(key_value_pairs.keys()), index=list(key_value_pairs.values()).index(st.session_state.pdf_file))

    # "---"
    ""
    # Update session variable on button selection
    if selected_option:
        st.session_state.pdf_file = key_value_pairs[selected_option]
        st.session_state.selected_option = selected_option 

    with st.expander(":gear: Settings"):
      # Create the slider and update 'llm_temparature' when it's changed
      st.session_state.llm_temparature = st.slider('Set LLM temperature', 0.0, 1.0, step=0.1, value=st.session_state.llm_temparature )

      if len(st.session_state.chat_history) > 0: 
        if st.button("➕ New Chat",  help="Reset conversation"):
          st.session_state.chat_history = []

  # render existing chats
  for message in st.session_state.chat_history:
    with st.chat_message("user"):
      st.markdown(message[0])
    with st.chat_message("assistant"):
      st.markdown(message[1])
          
  # React to user input
  if prompt := st.chat_input(f"Query '{st.session_state.selected_option}' documentation"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    with st.spinner(text="In progress..."): 
      result = qa({"question": prompt, "chat_history": st.session_state.chat_history})
      st.session_state.chat_history.append((prompt, result['answer']))

      response = f"{result['answer']}"
      # Display assistant response in chat message container
      with st.chat_message("assistant"):
        st.markdown(response) 

  # empty chat screen
  else:  
    if len(st.session_state.chat_history) == 0:
      blank_page()
     
if __name__ == "__main__":
  main()
