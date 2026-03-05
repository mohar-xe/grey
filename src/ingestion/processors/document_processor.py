import time
from pathlib import Path
from typing import List, Literal
from core.exceptions import ProcessingError

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

from docling.document_converter import DocumentConverter

class DocumentIngester:
    def __init__(self):
        self.docling_converter = DocumentConverter()

    def load(self, file_path: str | Path, method: Literal["langchain", "docling"] = "langchain") -> List[Document]:
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")

        file_extension = file_path.suffix.lower()

        if method == "docling":
            if file_extension == ".txt":
                print(f"Warning: Docling does not natively support .txt files. Falling back to LangChain TextLoader for {file_path.name}.")
                return self._load_text_langchain(file_path)
            return self._load_with_docling(file_path)
        
        elif method == "langchain":
            if file_extension == ".txt":
                return self._load_text_langchain(file_path)
            elif file_extension == ".pdf":
                return self._load_pdf_langchain(file_path)
            else:
                raise ProcessingError(f"LangChain routing not implemented for {file_extension}")
        
        else:
            raise ProcessingError(f"Unknown method: {method}. Choose 'langchain' or 'docling'.")

    def _load_text_langchain(self, file_path: Path) -> List[Document]:
        print(f"Loading {file_path.name} using LangChain TextLoader...")
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def _load_pdf_langchain(self, file_path: Path) -> List[Document]:
        print(f"Loading {file_path.name} using LangChain PyPDFLoader...")
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def _load_with_docling(self, file_path: Path) -> List[Document]:
        print(f"Loading {file_path.name} using Docling...")
        conversion_result = self.docling_converter.convert(str(file_path))
        text_content = conversion_result.document.export_to_markdown()
        
        metadata = {
            "source": str(file_path),
            "title": file_path.name,
            "parser": "docling",
            "file_type": file_path.suffix,
            "timestamp": time.time()
        }
        return [Document(page_content=text_content, metadata=metadata)]