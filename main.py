from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_store import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an exeprt in answering questions about a Control Plan and Inspection Plan Module which is in Teamcenter.
Here are information about Control Plan and Inspection Plan Module: {data}
Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

question = input("Ask your question: ")
data = retriever.invoke(question)

result = chain.invoke({"data": data, "question": question})
print(result)


