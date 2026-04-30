import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# 1.Setup LLm
HF_TOKEN= os.environ.get("HF_TOKEN")
huggingface_repo_id="meta-llama/Meta-Llama-3-8B-Instruct"

def load_llm(huggingface_repo_id):
  llm=HuggingFaceEndpoint(
    repo_id = huggingface_repo_id,
    temperature=0.5,  #temperature controls the randomness of the output. Lower values make the output more deterministic and predictable. Higher values make the output more random and creative.
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512
  )
  chat_model = ChatHuggingFace(llm=llm)
  return chat_model
# 2. Connect LLM with FAISS and Create chain

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question:{question}

Start the answer directly. No small talk please.

"""

def set_custom_prompt(custom_prompt_template):
  prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
  return prompt 

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

#create QA chain
qa_chain=RetrievalQA.from_chain_type(
  llm= load_llm(huggingface_repo_id),
  retriever=db.as_retriever(search_kwargs={"k":2}),   #from where answer is retirevd(db as retierver) #search.kwargs ---> based on ranked results(top 2 in this case)
  chain_type="stuff",          #which type of chain
  return_source_documents=True,   #to return the source documents(mmay be from which pdf)
  chain_type_kwargs={"prompt":set_custom_prompt(custom_prompt_template)}
)

# Invoke with single query
user_query=input("Write Query Here:")
response=qa_chain.invoke({"query":user_query})
print("RESULT:" , response['result'])
print("SOURCE DOC: ", response['source_documents'])
