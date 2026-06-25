import string
import uuid
from typing import List, Dict, Any, Tuple
import chromadb
from rank_bm25 import BM25Okapi
from app.config import DB_DIR
from app.embeddings import EmbeddingProvider

def tokenize_text(text: str) -> List[str]:
    """Simple tokenizer for BM25 keyword search."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()

class HybridRetriever:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
        
        # Two collections as specified in Step 5 (Query Routing)
        self.api_collection = self.chroma_client.get_or_create_collection(
            name="api_docs", 
            metadata={"hnsw:space": "cosine"}
        )
        self.knowledge_collection = self.chroma_client.get_or_create_collection(
            name="company_knowledge", 
            metadata={"hnsw:space": "cosine"}
        )
        
        # In-memory BM25 Okapi instances and corpus references
        self.api_bm25 = None
        self.api_corpus = []  # List of dicts: {"id": str, "content": str, "metadata": dict}
        
        self.knowledge_bm25 = None
        self.knowledge_corpus = []  # List of dicts
        
        # Build/load BM25 indices on startup if documents exist in DB
        self.rebuild_bm25_indices()

    def rebuild_bm25_indices(self):
        """Load documents from ChromaDB and fit BM25 indices."""
        # 1. API Docs collection
        api_data = self.api_collection.get()
        self.api_corpus = []
        if api_data and api_data["ids"]:
            for idx, doc_id in enumerate(api_data["ids"]):
                self.api_corpus.append({
                    "id": doc_id,
                    "content": api_data["documents"][idx],
                    "metadata": api_data["metadatas"][idx]
                })
            tokenized_api_corpus = [tokenize_text(doc["content"]) for doc in self.api_corpus]
            if tokenized_api_corpus:
                self.api_bm25 = BM25Okapi(tokenized_api_corpus)
                
        # 2. Company Knowledge collection
        knowledge_data = self.knowledge_collection.get()
        self.knowledge_corpus = []
        if knowledge_data and knowledge_data["ids"]:
            for idx, doc_id in enumerate(knowledge_data["ids"]):
                self.knowledge_corpus.append({
                    "id": doc_id,
                    "content": knowledge_data["documents"][idx],
                    "metadata": knowledge_data["metadatas"][idx]
                })
            tokenized_knowledge_corpus = [tokenize_text(doc["content"]) for doc in self.knowledge_corpus]
            if tokenized_knowledge_corpus:
                self.knowledge_bm25 = BM25Okapi(tokenized_knowledge_corpus)

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Embed chunks and insert them into the appropriate ChromaDB collections."""
        api_docs_batch = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
        knowledge_batch = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
        
        texts_to_embed = [c["content"] for c in chunks]
        print(f"Generating embeddings for {len(texts_to_embed)} chunks...")
        embeddings = self.embedding_provider.embed_documents(texts_to_embed)
        print("Embeddings generation complete.")
        
        for chunk, embedding in zip(chunks, embeddings):
            doc_id = str(uuid.uuid4())
            doc_type = chunk["metadata"].get("document_type", "api_docs")
            
            # Ensure metadata values are simple types (str, int, float, bool) for ChromaDB
            meta = {k: str(v) for k, v in chunk["metadata"].items()}
            
            if doc_type == "api_docs":
                api_docs_batch["ids"].append(doc_id)
                api_docs_batch["embeddings"].append(embedding)
                api_docs_batch["documents"].append(chunk["content"])
                api_docs_batch["metadatas"].append(meta)
            else:
                knowledge_batch["ids"].append(doc_id)
                knowledge_batch["embeddings"].append(embedding)
                knowledge_batch["documents"].append(chunk["content"])
                knowledge_batch["metadatas"].append(meta)
                
        # Upsert in ChromaDB
        if api_docs_batch["ids"]:
            print(f"Upserting {len(api_docs_batch['ids'])} chunks into 'api_docs' collection...")
            self.api_collection.upsert(
                ids=api_docs_batch["ids"],
                embeddings=api_docs_batch["embeddings"],
                documents=api_docs_batch["documents"],
                metadatas=api_docs_batch["metadatas"]
            )
            
        if knowledge_batch["ids"]:
            print(f"Upserting {len(knowledge_batch['ids'])} chunks into 'company_knowledge' collection...")
            self.knowledge_collection.upsert(
                ids=knowledge_batch["ids"],
                embeddings=knowledge_batch["embeddings"],
                documents=knowledge_batch["documents"],
                metadatas=knowledge_batch["metadatas"]
            )
            
        # Rebuild BM25 indices after new documents are ingested
        self.rebuild_bm25_indices()

    def semantic_search(self, query: str, collection_type: str, k: int = 5) -> List[Dict[str, Any]]:
        """Vector similarity search in specified collection."""
        collection = self.api_collection if collection_type == "api_docs" else self.knowledge_collection
        
        # Verify collection has items
        if collection.count() == 0:
            return []
            
        from app.cache import embedding_cache
        query_embedding = embedding_cache.get(query)
        if query_embedding is None:
            query_embedding = self.embedding_provider.embed_query(query)
            embedding_cache.set(query, query_embedding)
            
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count())
        )
        
        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][idx],
                    "content": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "score": 1.0 - results["distances"][0][idx] if results["distances"] else 0.0
                })
        return formatted_results

    def keyword_search(self, query: str, collection_type: str, k: int = 5) -> List[Dict[str, Any]]:
        """BM25 keyword search in specified collection."""
        bm25 = self.api_bm25 if collection_type == "api_docs" else self.knowledge_bm25
        corpus = self.api_corpus if collection_type == "api_docs" else self.knowledge_corpus
        
        if not bm25 or not corpus:
            return []
            
        tokenized_query = tokenize_text(query)
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0.0:  # Only return documents with some overlap
                results.append({
                    "id": corpus[idx]["id"],
                    "content": corpus[idx]["content"],
                    "metadata": corpus[idx]["metadata"],
                    "score": float(scores[idx])
                })
        return results

    def hybrid_search(self, query: str, collection_type: str, k: int = 5, rrf_constant: int = 60) -> List[Dict[str, Any]]:
        """Combines semantic search and keyword search using Reciprocal Rank Fusion (RRF)."""
        from app.cache import retrieval_cache
        cache_key = f"{collection_type}:{k}:{query}"
        cached_results = retrieval_cache.get(cache_key)
        if cached_results is not None:
            return cached_results
            
        semantic_results = self.semantic_search(query, collection_type, k=20)
        keyword_results = self.keyword_search(query, collection_type, k=20)
        
        # Reciprocal Rank Fusion (RRF) algorithm
        scores = {}
        chunk_map = {}
        
        for rank, chunk in enumerate(semantic_results):
            chunk_id = chunk["id"]
            chunk_map[chunk_id] = chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank + 1 + rrf_constant)
            
        for rank, chunk in enumerate(keyword_results):
            chunk_id = chunk["id"]
            chunk_map[chunk_id] = chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank + 1 + rrf_constant)
            
        # Prioritize technical details for API documentation searches
        if collection_type == "api_docs":
            boosted_scores = {}
            for cid, base_score in scores.items():
                chunk = chunk_map[cid]
                content = chunk["content"].lower()
                metadata = chunk["metadata"]
                section = metadata.get("section", "").lower()
                
                # Penalty if marketing/about files somehow got into technical searches
                file_name = metadata.get("file_name", "").lower()
                if "about_us.txt" in file_name or "homepage_overview.txt" in file_name or "general_knowledge" in file_name:
                    boosted_scores[cid] = base_score - 0.5
                    continue
                
                boost = 0.0
                
                # Query-aware filename and title matching boost (e.g., "authenticate" matching "authentication.md")
                query_words = [w.strip().lower() for w in tokenize_text(query) if len(w) >= 3]
                title = metadata.get("title", "").lower()
                for qw in query_words:
                    if qw in file_name or qw in title:
                        boost += 0.80
                    elif len(qw) >= 5:
                        prefix = qw[:5]
                        if prefix in file_name or prefix in title:
                            boost += 0.60
                            
                # Priority 1: Endpoint documentation / HTTP methods
                if any(method in chunk["content"] for method in ["GET ", "POST ", "PUT ", "DELETE "]) or "endpoint" in content or "endpoint" in section:
                    boost += 0.25
                # Priority 2: Code examples (python/bash/curl/javascript)
                if "```python" in content or "import requests" in content or "```bash" in content or "curl " in content or "code example" in content or "code example" in section:
                    boost += 0.20
                # Priority 3: Parameters
                if "| parameter" in content or "| name" in content or "query parameters" in content or "parameter" in section or "required" in content:
                    boost += 0.15
                # Priority 4: Authentication
                if "authentication" in content or "bearer" in content or "api key" in content or "authorization" in content or "auth" in section:
                    boost += 0.10
                # Priority 5: Request/Response examples
                if "response" in content or "example response" in content or "json" in content or "xml" in content or "response" in section:
                    boost += 0.05
                    
                boosted_scores[cid] = base_score + boost
                
            sorted_ids = sorted(boosted_scores.keys(), key=lambda x: boosted_scores[x], reverse=True)
        else:
            # Sort by fusion score descending
            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Retrieve the top-k chunks
        top_chunks = []
        for cid in sorted_ids[:k]:
            chunk = chunk_map[cid]
            # Add fusion score to metadata or info
            chunk["fusion_score"] = scores[cid]
            top_chunks.append(chunk)
            
        retrieval_cache.set(cache_key, top_chunks)
        return top_chunks

    def hybrid_search_mixed(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves from both collections and performs a global fusion."""
        from app.cache import retrieval_cache
        cache_key = f"mixed:{k}:{query}"
        cached_results = retrieval_cache.get(cache_key)
        if cached_results is not None:
            return cached_results
            
        # Query both api_docs and company_knowledge collections
        api_semantic = self.semantic_search(query, "api_docs", k=20)
        api_keyword = self.keyword_search(query, "api_docs", k=20)
        
        knowledge_semantic = self.semantic_search(query, "company_knowledge", k=20)
        knowledge_keyword = self.keyword_search(query, "company_knowledge", k=20)
        
        # Apply RRF globally across all retrieved sets
        scores = {}
        chunk_map = {}
        rrf_constant = 60
        
        # RRF for API
        for rank, chunk in enumerate(api_semantic):
            cid = chunk["id"]
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rank + 1 + rrf_constant)
        for rank, chunk in enumerate(api_keyword):
            cid = chunk["id"]
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rank + 1 + rrf_constant)
            
        # RRF for Knowledge
        for rank, chunk in enumerate(knowledge_semantic):
            cid = chunk["id"]
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rank + 1 + rrf_constant)
        for rank, chunk in enumerate(knowledge_keyword):
            cid = chunk["id"]
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rank + 1 + rrf_constant)
            
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        top_chunks = []
        for cid in sorted_ids[:k]:
            chunk = chunk_map[cid]
            chunk["fusion_score"] = scores[cid]
            top_chunks.append(chunk)
            
        retrieval_cache.set(cache_key, top_chunks)
        return top_chunks

    def get_stats(self) -> Dict[str, int]:
        """Returns document counts for collections."""
        return {
            "api_docs_count": self.api_collection.count(),
            "company_knowledge_count": self.knowledge_collection.count()
        }
        
    def clear_db(self):
        """Clears both collections and BM25 state by dropping and recreating collections to support dimension changes."""
        try:
            self.chroma_client.delete_collection("api_docs")
        except Exception as e:
            print(f"Error dropping api_docs: {e}")
            
        try:
            self.chroma_client.delete_collection("company_knowledge")
        except Exception as e:
            print(f"Error dropping company_knowledge: {e}")
            
        self.api_collection = self.chroma_client.get_or_create_collection(
            name="api_docs", 
            metadata={"hnsw:space": "cosine"}
        )
        self.knowledge_collection = self.chroma_client.get_or_create_collection(
            name="company_knowledge", 
            metadata={"hnsw:space": "cosine"}
        )
        
        self.api_corpus = []
        self.api_bm25 = None
        self.knowledge_corpus = []
        self.knowledge_bm25 = None
