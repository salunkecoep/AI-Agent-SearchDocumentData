from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("domain_info.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_csv_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for rowIndex, row in df.iterrows():
        document = Document(
            page_content=row["Information"],
            metadata={"Topic": row["Topic"]},
            id=str(rowIndex)
        )        
        ids.append(str(rowIndex))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="data",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 10000}
)






