import os
import pandas as pd
import textract
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain 
# from IPython.display import display
# import ipywidgets as widgets

load_dotenv()
  
print("Loading dataset...")
 
# Simple method - Split by pages 
loader = PyPDFLoader("./tokyo_api_reference_7-12-2023.pdf")
pages = loader.load_and_split()
chunks = pages

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)


# Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.7, max_tokens=1024), db.as_retriever())

chat_history = []

os.system('clear')
print("Welcome to the SNow Knowledge Base")
print("Type a question to continue")
print("")
while True:
    # Get user input for the prompt or follow-up question
    user_input = input("\n$ ")

    if user_input.lower() == "exit":
        break  # End the conversation if 'exit' is entered

    # Create a HumanMessage object with the user input as the content
    result = qa({"question": user_input, "chat_history": chat_history})
    chat_history.append((user_input, result['answer']))
 
    # Generate a response for the conversation 
    print (result["answer"])
 
print("Conversation ended.")


