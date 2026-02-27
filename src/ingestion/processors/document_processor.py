from docling.document_converter import DocumentConverter
from pathlib import Path
from urllib.parse import urlparse
from core.exceptions import ProcessingError
from core.constants import SUPPORTED_EXTENSIONS

def _is_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ('http', 'https')

def docling_convert(source: str) -> str:
    converter = DocumentConverter()
    result = converter.convert(source).document
    return result.export_to_markdown()

def text_processor(source: str) -> str:
    if _is_url(source):
        try:
            return docling_convert(source)
        except Exception as e:
            raise ProcessingError(f"Error processing URL {source}: {e}")
    else:
        path = Path(source)
        if not path.exists():
            raise ProcessingError(f"File not found: {source}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ProcessingError(f"Unsupported file type: {path.suffix}")
        try:
            return docling_convert(source)
        except Exception as e:
            raise ProcessingError(f"Error processing file {source}: {e}")