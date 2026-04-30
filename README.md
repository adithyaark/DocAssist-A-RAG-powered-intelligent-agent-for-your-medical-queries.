# DocAssist: RAG-Powered Medical Intelligent Agent

DocAssist is a Retrieval-Augmented Generation (RAG) powered medical chatbot designed to securely answer your health queries. It actively retrieves context from massive medical encyclopedias and dynamically reasons over custom PDF documents that you upload.

## Features
- **Dynamic Knowledge Base Expansion**: Upload any medical PDF directly from the UI to instantly expand the bot's context.
- **Conversational Memory**: Remembers past interactions in the chat to seamlessly handle follow-up questions.
- **Fast Retrieval**: Powered by `FAISS` for rapid, high-dimensional similarity searches.
- **Open-Source LLMs**: Connects seamlessly to Hugging Face models (default: `Meta-Llama-3-8B-Instruct`).
- **Dual Architecture**: Includes both a responsive `Streamlit` user interface and a flexible `FastAPI` backend.

## Project Structure
```text
DocAssist/
├── src/
│   ├── app.py                  # Main Streamlit User Interface
│   ├── api.py                  # FastAPI Backend Server
│   ├── database.py             # Logic for building the initial FAISS database
│   └── connect_mem_llm.py      # LLM model and chain connection logic
├── tests/                      # Unit tests
├── data/                       # Directory for default medical PDFs
├── vectorstore/                # FAISS vector embeddings database
├── .env                        # Environment variables (API tokens)
└── Pipfile                     # Dependency configuration
```

## Prerequisites
- **Python 3.12+**
- **Pipenv** (for dependency management)
- A **Hugging Face** API Token (Get one for free at [huggingface.co](https://huggingface.co/))

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adithyaark/DocAssist-A-RAG-powered-intelligent-agent-for-your-medical-queries..git
   cd DocAssist-A-RAG-powered-intelligent-agent-for-your-medical-queries.
   ```

2. **Install dependencies:**
   Pipenv will automatically create a virtual environment and install all packages required for this project.
   ```bash
   pipenv install
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory of the project and add your Hugging Face API token:
   ```env
   HF_TOKEN="your_hugging_face_api_token_here"
   ```

## How to Run

### 1. Launch the Streamlit User Interface
To start the chatbot UI, run the following command in your terminal:
```bash
pipenv run streamlit run src/app.py
```
*(If you are running the legacy flat structure, use `pipenv run streamlit run docassist.py`)*

Your browser will automatically open to `http://localhost:8501`. From there, you can upload PDFs into the sidebar and start asking medical questions!

### 2. (Optional) Launch the FastAPI Backend
If you wish to use DocAssist headlessly as an API service, start the backend server:
```bash
pipenv run uvicorn src.api:app --reload
```
The API will be available at `http://localhost:8000`. You can interact with it via `http://localhost:8000/docs`.
