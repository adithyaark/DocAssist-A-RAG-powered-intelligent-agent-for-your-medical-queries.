

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


import os
# 1. load raw pdfs.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, "data")

def load_pdf_files(data):
  loader = DirectoryLoader(data,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader)

  documents = loader.load()
  return documents


documents = load_pdf_files(DATA_PATH)
print("length of pdf: ", len(documents))

# 2. chunking
def create_chunks(extracted_data):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  text_chunks = text_splitter.split_documents(extracted_data)
  return text_chunks

text_chunks = create_chunks(documents)
print("length of chunks: ", len(text_chunks))


# 3. vector embeddings

def get_embedding_model():
  embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  return embedding_model

embedding_model = get_embedding_model()

# 4. store in vector DB

DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print("vector store created successfully")  