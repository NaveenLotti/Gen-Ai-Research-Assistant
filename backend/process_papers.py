from pathlib import Path
from uuid import uuid4

from app.rag.vectorstore import create_vector_store
from app.rag.loader import load_pdf
from app.rag.chunker import split_documents


# ============================================================
# PDF DIRECTORY
# ============================================================

PDF_DIR = Path("./data")


# ============================================================
# PROCESS ALL PDFs
# ============================================================

def process_papers():

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:

        print("No PDF files found.")

        return

    print(f"Found {len(pdf_files)} PDF(s).")

    all_chunks = []

    # --------------------------------------------------------
    # Process each PDF
    # --------------------------------------------------------

    for pdf_path in pdf_files:

        print("\n========================================")
        print(f"Processing: {pdf_path.name}")
        print("========================================")

        # Unique ID for this paper
        paper_id = str(uuid4())

        # Load PDF
        documents = load_pdf(
            str(pdf_path)
        )

        # Add paper metadata
        for document in documents:

            document.metadata["paper_id"] = paper_id

            document.metadata["paper_name"] = pdf_path.name

        # Split into chunks
        chunks = split_documents(
            documents
        )

        # Make sure metadata is preserved
        for chunk in chunks:

            chunk.metadata["paper_id"] = paper_id

            chunk.metadata["paper_name"] = pdf_path.name

        all_chunks.extend(chunks)

        print(f"Paper ID: {paper_id}")
        print(f"Pages: {len(documents)}")
        print(f"Chunks: {len(chunks)}")

    # --------------------------------------------------------
    # Add all chunks to Qdrant
    # --------------------------------------------------------

    print("\n========================================")
    print("Adding documents to Qdrant...")
    print("========================================")

    create_vector_store(
    all_chunks
)

    print("\n========================================")
    print("ALL PAPERS PROCESSED SUCCESSFULLY")
    print("========================================")


if __name__ == "__main__":

    process_papers()