from dotenv import load_dotenv
import os
import pandas as pd
import textract
import matplotlib.pyplot as plt
# from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain 

from blank import blank_page
import streamlit as st

load_dotenv()
 
@st.cache_data
def load_source_document(pdf_file): 
  base_name = os.path.basename(pdf_file)
  folder_name = os.path.splitext(base_name)[0]
  embeddings = OpenAIEmbeddings()

  with st.spinner(text="Loading "+ folder_name +"..."): 
    folder_path = f'db/{folder_name}'
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
      # Folder already exists, do something if needed
      return FAISS.load_local(folder_path, embeddings) 
    else: 
      loader = PyPDFLoader(pdf_file)
      chunks = loader.load_and_split()  
      db = FAISS.from_documents(chunks, embeddings)  
      db.save_local(folder_path)
      return db
    
def save_uploaded_file(uploaded_file):
  # Create the 'src' folder if it doesn't exist
  if not os.path.exists('src'):
    os.makedirs('src')
  
  # Save the uploaded file to the 'src' folder
  file_path = os.path.join('src', uploaded_file.name)
  with open(file_path, 'wb') as f:
    f.write(uploaded_file.getbuffer())
  
  load_source_document(file_path)
  return file_path

def convert_file_name(file_name): 
  file_name = file_name.split(".")[0] 
  words = file_name.split("_") 
  words = [word.capitalize() for word in words] 
  return " ".join(words)
  

def get_document_list():
  folder_path = './src'
  file_object = {}

  # Iterate over each file in the folder
  for file_name in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, file_name)):
      # Normalize the file name
      normalized_key = convert_file_name(file_name)
      
      # Store the file path in the dictionary
      file_object[normalized_key] = './src/' + file_name

  # Sort the file list alphabetically
  sorted_files = dict(sorted(file_object.items()))

  return sorted_files


@st.cache_data
def load_description(_qa, pdf):
  st.session_state.chat_history = []
  result = _qa({"question": 'Give a one paragraph summary of the document, then Describe up to five subjects covered with one sentence each. use markup where possible', "chat_history": []})
  # st.balloons()
  return result


# Load the OpenAI API key from the environment variable
if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
  print("OPENAI_API_KEY is not set")
  exit(1) 

 
if "pdf_file" not in st.session_state:
  st.session_state.pdf_file = './src/client_api_reference.pdf'
     
 

def main(): 
  st.set_page_config(page_title="SNow-GPT")

 
  key_value_pairs = get_document_list()

  if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
      
  if "selected_option" not in st.session_state:
    st.session_state.selected_option = ''
      
  if "selected_desc" not in st.session_state:
    st.session_state.selected_desc = None
      
  if "llm_temparature" not in st.session_state:
    st.session_state.llm_temparature = 0.7

  # build sidebar 
  with st.sidebar:

    st.header("ServiceNow Chatbot") 
    st.write("The ServiceNow LLM Knowledge base") 
    
    # Display radio buttons and handle selection
    selected_option = st.radio('Select a document:', list(key_value_pairs.keys()), key="selected_option_key", 
                                index=list(key_value_pairs.keys()).index(st.session_state.get('selected_option_key', 'Dev Ops')))

    # "---"
    # "" 
    # Update session variable on button selection
    if selected_option:
      st.session_state.pdf_file = key_value_pairs[selected_option]
      st.session_state.selected_option = selected_option 
      st.session_state.selected_desc = None
      # st.session_state.chat_history = []

    with st.expander(":gear: Settings"):
      # Create the slider and update 'llm_temparature' when it's changed
      st.session_state.llm_temparature = st.slider('Set LLM temperature', 0.0, 1.0, step=0.1, value=st.session_state.llm_temparature )

      if len(st.session_state.chat_history) > 0: 
        if st.button("➕ New Chat",  help="Reset conversation"):
          st.session_state.chat_history = []

    with st.expander(":file_folder: Add document"):
      uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

      if uploaded_file is not None:
        # Save the uploaded file and display a success message
        file_path = save_uploaded_file(uploaded_file)
        st.success(f"File saved successfully: {file_path}")
        key_value_pairs = get_document_list()


  llm = OpenAI(temperature=st.session_state.llm_temparature)
      
  db = load_source_document(st.session_state.pdf_file) 

  # Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
  qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever())

  st.session_state.selected_desc = load_description(qa, st.session_state.pdf_file) ['answer'] 

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

    with st.spinner(text="Generating answer..."): 
      result = qa({"question": prompt, "chat_history": st.session_state.chat_history})
      response = f"{result['answer']}"
      st.session_state.chat_history.append((prompt, response))

      # Display assistant response in chat message container
      with st.chat_message("assistant"):
        st.markdown(response) 

  # empty chat screen
  else:  
    if len(st.session_state.chat_history) == 0:
      blank_page()
     
if __name__ == "__main__":
  main()
