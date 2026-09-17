from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from app.agents.supervisor import research_graph
from app.rag.loader import load_pdf
from app.rag.chunker import split_documents
from app.rag.vectorstore import add_documents_to_vector_store


app = FastAPI(
    title="Research Paper AI Assistant",
    description="Multi-agent RAG assistant for research papers",
    version="1.0.0"
)


UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():

    return {
        "message": "Research Paper AI Assistant API",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }
@app.get("/papers")
def get_papers():

    papers = []

    for file_path in UPLOAD_DIR.glob("*.pdf"):

        papers.append({
            "filename": file_path.name,
            "size": file_path.stat().st_size
        })

    return {
        "success": True,
        "count": len(papers),
        "papers": papers
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # Validate file
    if not file.filename.lower().endswith(".pdf"):

        return {
            "success": False,
            "error": "Only PDF files are supported."
        }


    # Save uploaded PDF
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )


    # Load PDF
    documents = load_pdf(
        str(file_path)
    )


    # Add paper metadata
    for document in documents:

        document.metadata["paper_name"] = file.filename


    # Create chunks
    chunks = split_documents(
        documents
    )


    # Preserve metadata
    for chunk in chunks:

        chunk.metadata["paper_name"] = file.filename


    # Store in Qdrant
    add_documents_to_vector_store(
        chunks
    )


    return {

        "success": True,

        "filename": file.filename,

        "pages": len(documents),

        "chunks": len(chunks),

        "message":
            "Research paper uploaded and indexed successfully."
    }


@app.post("/query")
def query_research_paper(
    request: QueryRequest
):

    try:

        result = research_graph.invoke({

            "query": request.query,

            "answer": "",

            "sources": [],

            "agent": ""
        })


        return {

            "success": True,

            "query": request.query,

            "answer": result["answer"],

            "agent": result["agent"],

            "sources": result["sources"]
        }


    except Exception as e:

        return {

            "success": False,

            "query": request.query,

            "answer":
                "The AI service is temporarily unavailable.",

            "agent": "ERROR",

            "sources": [],

            "error": str(e)
        }