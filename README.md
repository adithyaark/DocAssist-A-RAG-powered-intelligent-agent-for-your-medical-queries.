# DocAssist: A RAG‑powered intelligent agent for your medical queries

DocAssist is a medical question‑answering assistant that uses **Retrieval‑Augmented Generation (RAG)** to answer your queries from uploaded PDF documents. Instead of relying only on a general model, it first retrieves relevant information from your medical files and then generates a grounded response.

## Features

- Upload medical PDF documents.  
- Automatic text extraction and chunking.  
- Semantic search over your documents.  
- RAG‑based answers for medical questions.  

## How it works

1. You upload a medical PDF.  
2. The system extracts and chunks the text, then creates embeddings.  
3. For each query, it retrieves the most relevant chunks.  
4. The LLM generates an answer using that retrieved context.  

## Tech stack

- Python  
- Large Language Model (LLM)  
- RAG pipeline  
- PDF text extraction  
- Embedding model & vector retrieval  

## Installation

```bash
git clone https://github.com/adithyaark/DocAssist-A-RAG-powered-intelligent-agent-for-your-medical-queries.git
cd DocAssist-A-RAG-powered-intelligent-agent-for-your-medical-queries
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

> Replace `app.py` with your main file if the name is different.

## Limitations

- For educational use only; not a substitute for real medical advice.  
- Responses depend on the quality of uploaded documents.  

## Author

**Adithya R K**
