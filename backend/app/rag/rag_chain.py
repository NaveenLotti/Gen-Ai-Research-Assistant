import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.rag.retriever import retrieve_documents

load_dotenv()


def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0
    )


def generate_answer(query: str):

    # Retrieve relevant chunks
    documents = retrieve_documents(query, k=4)

    # Build context
    context_parts = []

    for i, doc in enumerate(documents, start=1):

        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"""
SOURCE {i}
PAGE: {page}

{doc.page_content}
"""
        )

    context = "\n".join(context_parts)

    # Prompt Gemini
    prompt = f"""
You are a research paper assistant.

Answer the user's question using ONLY the provided research-paper
context.

If the answer cannot be found in the context, say:
"I could not find this information in the provided research paper."

Do not invent information.

Always mention the page number(s) used for your answer.

Research Paper Context:
-----------------------
{context}
-----------------------

User Question:
{query}

Provide a clear and concise answer.
"""

    llm = get_llm()

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [
            {
                "page": doc.metadata.get("page"),
                "content": doc.page_content
            }
            for doc in documents
        ]
    }