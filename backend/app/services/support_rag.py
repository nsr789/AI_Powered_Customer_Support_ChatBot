from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.core.llm import OpenAIProvider, FakeLLM
from app.core.vector_store import get_collection

_prompt = PromptTemplate.from_template(
    """You are an e-commerce support agent.
Use the provided context to answer the user's question in a concise paragraph.
If not in the docs, reply "I'm sorry, I don't have that information yet.".

Context:
{context}

Question: {question}
Answer:"""
)

def answer(query: str) -> dict:
    vectordb = get_collection("support_kb")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = OpenAIProvider() if _using_openai() else FakeLLM()
    chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type_kwargs={"prompt": _prompt}
    )
    result = chain(query)
    return {"answer": result["result"], "sources": [d.metadata for d in result["source_documents"]]}

def _using_openai() -> bool:
    import os
    return bool(os.getenv("OPENAI_API_KEY"))
