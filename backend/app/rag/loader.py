from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str):
    """
    Load a PDF and convert each page into a LangChain Document.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(path))

    documents = loader.load()

    print(f"PDF loaded successfully!")
    print(f"Number of pages: {len(documents)}")

    return documents