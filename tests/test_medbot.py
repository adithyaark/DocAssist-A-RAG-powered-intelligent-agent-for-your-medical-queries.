import pytest
from langchain_core.documents import Document
from create_memory import create_chunks
from medbot import set_custom_prompt, get_embedding_model
from langchain_community.vectorstores import FAISS

def test_create_chunks():
    # 1. Arrange: Create a fake document longer than our chunk_size (500)
    long_text = "This is a sentence that we will repeat to make a very long document. " * 50
    docs = [Document(page_content=long_text)]
    
    # 2. Act: Run the chunking function
    chunks = create_chunks(docs)
    
    # 3. Assert: Check if it successfully broke it down
    assert len(chunks) > 1, "The document should have been split into multiple chunks"
    
    # Check if the chunk sizes are within limits (500 + some leeway for word boundaries)
    for chunk in chunks:
        assert len(chunk.page_content) <= 550, f"Chunk size too large: {len(chunk.page_content)}"

def test_custom_prompt():
    # 1. Arrange
    template = "Context: {context}\nQuestion: {question}"
    
    # 2. Act
    prompt = set_custom_prompt(template)
    
    # 3. Assert
    assert prompt.template == template
    assert "context" in prompt.input_variables, "Prompt must contain 'context' variable"
    assert "question" in prompt.input_variables, "Prompt must contain 'question' variable"

def test_embedding_model_creation():
    # Test if the model loads successfully
    model = get_embedding_model()
    assert hasattr(model, "embed_query"), "Embedding model should have 'embed_query' method"

def test_faiss_vectorstore_retrieval():
    # 1. Arrange: Create a tiny fake memory base
    model = get_embedding_model()
    fake_docs = [
        Document(page_content="MedBot is an AI medical assistant."),
        Document(page_content="Python is a programming language.")
    ]
    
    # 2. Act: Build a temporary vector store
    db = FAISS.from_documents(fake_docs, model)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    
    # Try to retrieve medical info
    results = retriever.invoke("What does MedBot do?")
    
    # 3. Assert: Ensure it retrieves the right document
    assert len(results) == 1
    assert "medical assistant" in results[0].page_content, "RAG should retrieve the most relevant context"
