import time
from app.ingestion import ingest_all
from app.embeddings import GeminiEmbeddingProvider
from app.db import HybridRetriever

def main():
    print("==================================================")
    print("Starting WHOIS Data Center RAG Ingestion Pipeline")
    print("==================================================")
    
    start_time = time.time()
    
    # 1. Ingest all docs and knowledge files
    chunks = ingest_all()
    print(f"Successfully processed and chunked {len(chunks)} documents in total.")
    
    if not chunks:
        print("No document chunks were found. Ingestion aborted.")
        return
        
    # 2. Initialize the embedding provider (Gemini text-embedding-004)
    print("Initializing Gemini Embedding Provider...")
    embedding_provider = GeminiEmbeddingProvider()
    
    # 3. Initialize the Hybrid Retriever (creates ChromaDB Persistent Client)
    print("Connecting to ChromaDB and setting up collections...")
    retriever = HybridRetriever(embedding_provider)
    
    # Optional: Clear existing DB to ensure a clean run
    print("Clearing previous vector store indexes for a fresh reload...")
    retriever.clear_db()
    
    # 4. Insert chunks
    print("Indexing document chunks in vector database (ChromaDB) and fitting BM25 index...")
    retriever.add_chunks(chunks)
    
    elapsed_time = time.time() - start_time
    stats = retriever.get_stats()
    
    print("\n==================================================")
    print("Ingestion Pipeline Completed Successfully!")
    print("==================================================")
    print(f"Total Chunks Indexed: {len(chunks)}")
    print(f"  - API Docs Chunks: {stats['api_docs_count']}")
    print(f"  - Company Knowledge Chunks: {stats['company_knowledge_count']}")
    print(f"Total Processing Time: {elapsed_time:.2f} seconds")
    print("ChromaDB is saved at: /chroma_db")
    print("==================================================")

if __name__ == "__main__":
    main()
