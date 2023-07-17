from dotenv import load_dotenv
import os 
from langchain.chains import ConversationalRetrievalChain 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from blank import blank_page
from util import get_document_list, initialize_session, load_description, load_source_document, save_uploaded_file
import streamlit as st

load_dotenv()
  
# Load the OpenAI API key from the environment variable
if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
  print("OPENAI_API_KEY is not set")
  exit(1) 

# build and render page sidebar 
def render_sidebar():
  source_docs = get_document_list()
  
  with st.sidebar:

    st.header("ServiceNow Chatbot") 
    st.write("The ServiceNow LLM Knowledge base") 
    
    # Display radio buttons and handle document selection
    selected_option = st.radio('Select a document:', 
                                list(source_docs.keys()), 
                                key="selected_option_key", 
                                index=list(source_docs.keys()).index(st.session_state.get('selected_option_key', 'Dev Ops')))
 
    # Update session variable on button selection
    if selected_option:
      st.session_state.pdf_file = source_docs[selected_option]
      st.session_state.selected_option = selected_option 
      st.session_state.selected_desc = None 

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


def main(): 

  st.set_page_config(page_title="SNow-GPT", layout="wide") 
  
  # set initial session values
  initialize_session() 

  # draw page side bar
  render_sidebar()
 
  # Create a ChatOpenAI object for streaming chat with specified temperature
  chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=st.session_state.llm_temparature)

  # Load the source document for the conversation retrieval
  db = load_source_document(st.session_state.pdf_file)

  # Create a conversation chain that uses the vectordb as a retriever, allowing for chat history management
  qa = ConversationalRetrievalChain.from_llm(chat, db.as_retriever())

  # Load the description using the conversation chain and store the answer in selected_desc
  st.session_state.selected_desc = load_description(qa, st.session_state.pdf_file)['answer'] 

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

    # Display a spinner while generating the answer
    with st.spinner(text="Generating answer..."):
      # Use the conversation chain (qa) to generate an answer based on the prompt and chat history
      result = qa({"question": prompt, "chat_history": st.session_state.chat_history})

      # Get the answer from the result
      response = f"{result['answer']}"

      # Append the prompt and response to the chat history
      st.session_state.chat_history.append((prompt, response))

      # Display the assistant's response in the chat message container
      with st.chat_message("assistant"):
          st.markdown(response)

  # empty chat screen
  else:  
    if len(st.session_state.chat_history) == 0:
      blank_page()
     
if __name__ == "__main__":
  main()
