import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NEW: Import ConversationalRetrievalChain instead of RetrievalQA
from langchain_classic.chains import ConversationalRetrievalChain
# Old code (unused):
# from langchain_classic.chains import RetrievalQA

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
HUGGINGFACE_REPO_ID="meta-llama/Meta-Llama-3-8B-Instruct"

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_vectorstore():
    embedding_model = get_embedding_model()
    if os.path.exists(DB_FAISS_PATH):
        try:
            db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)
            return db
        except Exception:
            return None
    return None

def set_custom_prompt(custom_prompt_template):
  prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
  return prompt 

@st.cache_resource
def load_llm(huggingface_repo_id, HF_TOKEN):
  llm=HuggingFaceEndpoint(
    repo_id = huggingface_repo_id,
    temperature=0.5,
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512
  )
  chat_model = ChatHuggingFace(llm=llm)
  return chat_model

@st.cache_resource
def get_qa_chain():
    HF_TOKEN=os.getenv("HF_TOKEN")
    llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
    vectorstore = get_vectorstore()
    
    if vectorstore is None:
        return None
        
    # --- OLD UNUSED CODE (Commented out) ---
    # custom_prompt_template = """
    #         If user gives a 'Hy or Hello', just greet politely for one time.(Dont mention about one time in repsone) and ask for how can i help you.
    #         Use the pieces of information provided in the context to answer user's question.
    #         ...
    #         Context: {context}
    #         Question:{question}
    #         Start the answer directly. No small talk please.
    #         """
    # qa_chain=RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(search_kwargs={"k":2}),
    #     chain_type="stuff",
    #     return_source_documents=True,
    #     chain_type_kwargs={"prompt":set_custom_prompt(custom_prompt_template)}
    # )
    
    # --- NEW CODE (ConversationalRetrievalChain) ---
    custom_prompt_template = """
If the user gives a greeting or closing ('Hi', 'Hello', 'Thank you', etc.), respond politely to that specific message without summarizing unrelated medical information.
Use the pieces of information provided in the context to answer the user's question or explain the concept they mentioned.
If the user types a single word or phrase (like "Arthritis" or "mast cells"), provide a summary or definition of it based on the context.
If you don't know the answer or the context doesn't have relevant information, just say you don't know. Don't try to make up an answer.
Don't provide anything out of the given context if answering a medical question.

Context: {context}
Question: {question}

Start the answer directly.
"""
    prompt = set_custom_prompt(custom_prompt_template)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k":6}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return qa_chain

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(documents)
        
        embedding_model = get_embedding_model()
        vectorstore = get_vectorstore()
        
        if vectorstore is None:
            vectorstore = FAISS.from_documents(text_chunks, embedding_model)
        else:
            vectorstore.add_documents(text_chunks)
            
        vectorstore.save_local(DB_FAISS_PATH)
        
        get_vectorstore.clear()
        get_qa_chain.clear()
        return True
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return False
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def check_login():
    """Simple authentication check using Streamlit session state."""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        st.title("🔒 Login to DocAssist")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if username == "admin" and password == "medbot123":
                    st.session_state["logged_in"] = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        return False
        
    with st.sidebar:
        if st.button("Logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.rerun()
            
    return True

def main():
  if not check_login():
      return

  st.title("Ask DocAssist!")

  with st.sidebar:
      st.header("Document Upload")
      st.write("Upload a PDF to expand the DocAssist knowledge base.")
      uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
      if uploaded_file is not None:
          if st.button("Process Document"):
              with st.spinner("Processing & Adding to Vector Database..."):
                  success = process_uploaded_file(uploaded_file)
                  if success:
                      st.success("Document added successfully! You can now ask questions about it.")

  # Initialize both UI messages and background langChain history
  if 'messages' not in st.session_state:
    st.session_state.messages=[]
    st.session_state.chat_history=[] # For Conversational Retrieval memory

  for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

  prompt = st.chat_input("Ask a question")

  if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    qa_chain = get_qa_chain()
    
    if qa_chain is None:
        st.error("Knowledge base is empty. Please upload a PDF in the sidebar first.")
        return

    try:
      with st.spinner("Thinking..."):
          # --- OLD UNUSED CODE (Commented out) ---
          # response_data = qa_chain.invoke({"query":prompt})
          # result = response_data['result']

          # --- NEW CODE (Uses conversational memory) ---
          response_data = qa_chain.invoke({
              "question": prompt, 
              "chat_history": st.session_state.chat_history
          })
          result = response_data['answer']
          
          # Append this conversation turn to the internal memory buffer
          st.session_state.chat_history.append((prompt, result))
          
      st.chat_message('assistant').markdown(result)
      st.session_state.messages.append({"role":"assistant","content":result})
      
    except Exception as e:
      st.error(f"Error: {e}")
    
if __name__ == "__main__":
  main()
