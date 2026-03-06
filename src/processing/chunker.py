from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.exceptions import ProcessingError
from transformers import AutoTokenizer

class SemanticTokenChunker:

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 250,
            model_name: str = "meta/llama-3.1-70b-instruct",
            separators: Optional[List[str]] = None
            ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.separators = separators or ["\n\n", "\n", ".", ",", "!", "?", " "]

        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not text.strip():
            raise ProcessingError("Cannot chunk empty or whitespace-only text.")

        docs = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata or {}]
        )

        formatted_chunks = []
        for i, doc in enumerate(docs):
            chunk_data = {
                "text": doc.page_content,
                "tokens": self.count_tokens(doc.page_content),
                "metadata": {**doc.metadata, "chunk_index": i}
            }
            formatted_chunks.append(chunk_data)

        return formatted_chunks