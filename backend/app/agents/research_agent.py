from app.rag.rag_chain import generate_answer
from app.rag.llm import get_llm


def research_agent(query: str):
    """
    Research Agent:
    Uses the existing RAG pipeline to answer
    questions about uploaded research papers.
    """

    result = generate_answer(query)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "agent": "Research Agent"
    }