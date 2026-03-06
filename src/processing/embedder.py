import os
from dotenv import load_dotenv
from typing import List
from openai import OpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()

class Embedder:

    def __init__(
        self, 
        model_name: str = None,
        base_url: str = None,
        api_key: str = None,
    ):
        self.model_name = model_name or os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
        base_url = base_url or os.getenv("EMBED_BASE_URL", "https://integrate.api.nvidia.com/v1")

        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key and "integrate.api.nvidia.com" in base_url:
            raise ValueError("NVIDIA_API_KEY environment variable is not set.")

        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key
        )

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name,
            encoding_format="float",
            extra_body={"input_type": "query"} 
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
            encoding_format="float",
            extra_body={"input_type": "passage"}
        )
        return [data.embedding for data in response.data]


def get_langchain_nim_embedder(
    model_name: str = None,
):
    model_name = model_name or os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
    return NVIDIAEmbeddings(model=model_name)
