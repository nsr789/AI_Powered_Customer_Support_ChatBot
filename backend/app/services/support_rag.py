"""
Retrieve-Augmented Generation chain for customer-support answers.
Returns dict with 'answer' and 'sources'.
"""
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.core.llm import OpenAIProvider, FakeLLM
from app.core.vector_store import get_collection

_PROMPT = PromptTemplate.from_template(
    """
You are an e-commerce support agent.
Use the context below to answer the question in 1–3 sentences.
If the information is not in context, reply:
"I'm sorry, I don't have that information yet.".

Context:
{context}

Question: {question}
Answer:
"""
)


def answer(query: str) -> dict:
    vectordb = get_collection("support_kb")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = OpenAIProvider() if bool(os.getenv("OPENAI_API_KEY")) else FakeLLM()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": _PROMPT},
        return_source_documents=True,
    )

    result = chain(query)
    return {
        "answer": result["result"],
        "sources": [d.metadata for d in result["source_documents"]],
    }
