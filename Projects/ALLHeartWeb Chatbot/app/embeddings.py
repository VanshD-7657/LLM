import abc
import time
from typing import List, Union
import numpy as np
import requests
from app.config import GEMINI_API_KEY, HF_TOKEN, OPENAI_API_KEY, DEFAULT_EMBEDDING_PROVIDER

def call_with_retry(api_func, *args, max_retries=5, initial_delay=2.0, backoff_factor=2.0, **kwargs):
    """Call a function with retry logic and exponential backoff, handling rate limits."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return api_func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            # If we've run out of retries, raise the exception
            if attempt == max_retries - 1:
                raise e
            
            print(f"API call failed: {e}. Retrying in {delay:.2f}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
            delay *= backoff_factor

class EmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        pass

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        pass


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "models/gemini-embedding-2"):
        self.model_name = model_name
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        self.client = genai

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        # We batch in groups of 50 to prevent hitting rate or payload size limits.
        embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Call with retry for the batch
                response = call_with_retry(
                    self.client.embed_content,
                    model=self.model_name,
                    content=batch,
                    task_type="retrieval_document"
                )
                embeddings.extend(response['embedding'])
                # A small sleep to prevent hitting RPM limits on free tier
                time.sleep(1.0)
            except Exception as e:
                print(f"Batch embedding failed for range {i} to {i+len(batch)}: {e}. Falling back to single embedding with retry...")
                # Fallback to single requests with retry and delay
                for text in batch:
                    response = call_with_retry(
                        self.client.embed_content,
                        model=self.model_name,
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(response['embedding'])
                    time.sleep(1.0) # sleep 1 second between single requests to avoid rate limits
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = call_with_retry(
            self.client.embed_content,
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return response['embedding']


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.model_name = model_name
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        # Batch in sizes of 50 with retry logic
        embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = call_with_retry(
                self.client.embeddings.create,
                input=batch,
                model=self.model_name
            )
            embeddings.extend([data.embedding for data in response.data])
            time.sleep(0.5)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = call_with_retry(
            self.client.embeddings.create,
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Uses Hugging Face Feature Extraction API for models like BGE-large."""
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    def _query_api(self, texts: List[str]) -> List[List[float]]:
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN is required for Hugging Face Inference API.")
        
        def _post():
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": texts, "options": {"wait_for_model": True}}
            )
            if response.status_code != 200:
                raise Exception(f"Hugging Face API error: {response.text}")
            return response.json()
            
        result = call_with_retry(_post)
        
        # If model returns 3D array (for token embeddings), we average it.
        # But standard feature extraction models return 2D array.
        if isinstance(result, list):
            # Check if it is a list of lists of floats
            if result and isinstance(result[0], list):
                if isinstance(result[0][0], float):
                    return result
                elif isinstance(result[0][0], list):
                    # 3D structure: [batch_size, seq_len, hidden_dim], perform mean pooling
                    pooled = []
                    for doc in result:
                        arr = np.mean(doc, axis=0)
                        pooled.append(arr.tolist())
                    return pooled
        raise ValueError(f"Unexpected response structure from HF API: {result}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # Batch sizes of 16 to avoid HF timeouts
        embeddings = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            embeddings.extend(self._query_api(batch))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # BGE models require query instruction
        query_text = f"Represent this sentence for searching relevant passages: {text}"
        return self._query_api([query_text])[0]


def get_embedding_provider(provider_name: str = None) -> EmbeddingProvider:
    if provider_name is None:
        provider_name = DEFAULT_EMBEDDING_PROVIDER
    
    provider_name = provider_name.lower()
    if provider_name == "gemini":
        return GeminiEmbeddingProvider()
    elif provider_name == "openai":
        return OpenAIEmbeddingProvider()
    elif provider_name == "huggingface" or provider_name == "bge":
        return HuggingFaceEmbeddingProvider()
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")
