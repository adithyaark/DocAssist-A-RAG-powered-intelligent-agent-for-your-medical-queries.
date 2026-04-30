import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
HUGGINGFACE_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

CUSTOM_PROMPT = """
If user gives a 'Hi or Hello', just greet politely and ask how you can help.
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# ── Lazy-load singletons ──────────────────────────────────────────────────────
_vectorstore = None
_qa_chain = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return _vectorstore

def get_qa_chain():
    global _qa_chain
    if _qa_chain is None:
        llm = HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            temperature=0.5,
            huggingfacehub_api_token=HF_TOKEN,
            max_new_tokens=512
        )
        chat_model = ChatHuggingFace(llm=llm)
        prompt = PromptTemplate(template=CUSTOM_PROMPT, input_variables=["context", "question"])
        _qa_chain = RetrievalQA.from_chain_type(
            llm=chat_model,
            retriever=get_vectorstore().as_retriever(search_kwargs={"k": 6}),
            chain_type="stuff",
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )
    return _qa_chain

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MedBot API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:5082"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        qa = get_qa_chain()
        result = qa.invoke({"query": req.message})
        return ChatResponse(reply=result["result"])
    except Exception as e:
        return ChatResponse(reply=f"Sorry, I ran into an error: {str(e)}")
