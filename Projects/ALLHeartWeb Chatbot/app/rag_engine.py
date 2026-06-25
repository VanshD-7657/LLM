from typing import List, Dict, Any, Tuple, Optional
import time
from app.embeddings import get_embedding_provider
from app.llm import get_llm_provider, LLMProvider
from app.db import HybridRetriever
from app.routing import QueryRouter
from app.memory import ConversationMemory
from app.config import DEFAULT_EMBEDDING_PROVIDER, DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL

def merge_overlapping_text(text1: str, text2: str) -> str:
    """Find the overlap between the end of text1 and start of text2 and merge them."""
    max_overlap = min(len(text1), len(text2))
    for l in range(max_overlap, 0, -1):
        if text1.endswith(text2[:l]):
            return text1 + text2[l:]
    return text1 + "\n\n" + text2

def compress_context(chunks: List[Dict[str, Any]], max_chunks: int = 3) -> List[Dict[str, Any]]:
    """Deduplicates chunks, merges overlapping text from the same file, and restricts count."""
    seen_content = set()
    unique_chunks = []
    for c in chunks:
        content = c["content"].strip()
        if content not in seen_content:
            seen_content.add(content)
            unique_chunks.append(c)
            
    merged_chunks = []
    file_groups = {}
    for c in unique_chunks:
        fname = c["metadata"].get("file_name", "Unknown")
        file_groups.setdefault(fname, []).append(c)
        
    for fname, group in file_groups.items():
        if not group:
            continue
        current = group[0]
        for next_chunk in group[1:]:
            merged_content = merge_overlapping_text(current["content"], next_chunk["content"])
            current["content"] = merged_content
        merged_chunks.append(current)
        
    return merged_chunks[:max_chunks]

def sanitize_sections(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Post-processes LLM output to strictly force section fallbacks if source context blocks are missing."""
    # Check if context contains python code
    context_has_python = any("```python" in c["content"] or "import requests" in c["content"] for c in retrieved_chunks)
    
    # Check if context contains JSON/XML blocks
    context_has_response = any("```json" in c["content"] or "```xml" in c["content"] or ("{" in c["content"] and "}" in c["content"]) for c in retrieved_chunks)
    
    # Standardize splits by ensuring uniform newline separators
    standardized = answer
    if standardized.startswith("## "):
        standardized = "\n" + standardized
        
    sections = standardized.split("\n## ")
    new_sections = []
    
    new_sections.append(sections[0])
    
    for sec in sections[1:]:
        parts = sec.split("\n", 1)
        header = parts[0]
        content = parts[1] if len(parts) > 1 else ""
        header_lower = header.strip().lower()
        
        if "python example" in header_lower and not context_has_python:
            content = "\nI could not find this information in the indexed API documentation.\n"
        elif "response" in header_lower and not context_has_response:
            content = "\nI could not find this information in the indexed API documentation.\n"
            
        new_sections.append(f"{header}\n{content}")
        
    result = "\n## ".join(new_sections)
    return result.strip()

SYSTEM_PROMPT = (
    "You are WHOIS Data Center Technical Support Assistant.\n"
    "Your objective is to answer the user's questions based ONLY on the retrieved context provided below.\n"
    "Never invent endpoints, parameters, response fields, authentication methods, or code examples. Do not guess.\n"
    "CRITICAL: If any required section information is not explicitly present in the retrieved context, you MUST output exactly "
    "\"I could not find this information in the indexed API documentation.\" for that section. Do not write placeholder code, do not invent code, and do not make up parameter tables.\n\n"
    "Determine if the query is about a specific API endpoint (e.g. WHOIS Lookup, Historical WHOIS, Reverse WHOIS) or a general technical question (e.g. general authentication setup, rate limits, error codes).\n\n"
    "For specific API endpoint technical questions:\n"
    "- Prioritize code examples (Python and cURL).\n"
    "- Prioritize endpoint documentation.\n"
    "- Prioritize parameters.\n"
    "- Prioritize official examples.\n"
    "- Ignore company/business information unless explicitly requested.\n"
    "- Structure the technical response strictly using the following Markdown sections and headers:\n\n"
    "## Endpoint\n"
    "Provide the HTTP method and endpoint URL (e.g. `GET /api/v2/domain/registrar`). If not present in context, output exactly: I could not find this information in the indexed API documentation.\n\n"
    "## Purpose\n"
    "Provide a brief, 1-2 sentence description of what the endpoint does. If not present in context, output exactly: I could not find this information in the indexed API documentation.\n\n"
    "## Python Example\n"
    "Provide a complete, copy-pasteable Python code example for calling the endpoint (always include `import requests`, endpoint URL, headers, and requests library usage). This must appear before descriptive text. If the retrieved context does not contain a Python code snippet, output exactly: I could not find this information in the indexed API documentation. Do NOT invent a python example.\n\n"
    "## Parameters\n"
    "Provide a Markdown table showing parameters. Columns must be: Parameter, Type, Required/Optional, Description. If not present in context, output exactly: I could not find this information in the indexed API documentation. Do NOT invent parameters.\n\n"
    "## Response\n"
    "Provide the expected response format (JSON or XML) in a code block. If the retrieved context does not contain a JSON/XML response payload example, output exactly: I could not find this information in the indexed API documentation. Do NOT generate a mock JSON payload.\n\n"
    "## Documentation Source\n"
    "List the source file path (e.g. `docs/core_whois/whois_lookup.md`) or the Official Endpoint. If not present in context, output exactly: I could not find this information in the indexed API documentation.\n\n"
    "For general technical questions (such as general authentication setup, rate limits, or error codes):\n"
    "- Do NOT use the strict endpoint sections above.\n"
    "- Provide a direct, structured markdown answer based strictly on the retrieved context (e.g., explaining how to format headers, listing rate limit details, or mapping error codes).\n\n"
    "For business questions:\n"
    "- Use company knowledge documents to provide concise, business-focused answers.\n\n"
    "For mixed questions:\n"
    "- Separate technical and business sections. Never mix business information into technical answers unless the user explicitly asks for it."
)

def construct_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks and user query into a structural RAG prompt."""
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks):
        meta = chunk["metadata"]
        context_str += f"--- Document [{idx+1}] ---\n"
        context_str += f"Source: {meta.get('source', 'Unknown')}\n"
        context_str += f"Category: {meta.get('category', 'General')}\n"
        context_str += f"Section: {meta.get('section', 'Main')}\n"
        context_str += f"Content:\n{chunk['content']}\n\n"
        
    prompt = (
        "Retrieved Context:\n"
        "=========================================\n"
        f"{context_str}"
        "=========================================\n\n"
        "User Question:\n"
        f"\"{query}\"\n\n"
        "Instructions: Answer the User Question strictly using the Retrieved Context above. "
        "Remember to cite sources (e.g., [docs/...] or [knowledge/...]) directly in your text when referencing facts. "
        "If the information is not present in the context, say \"I could not find this information in the available documentation.\""
    )
    return prompt

class RAGEngine:
    def __init__(self, 
                 embedding_provider: Optional[str] = None, 
                 llm_provider: Optional[str] = None, 
                 llm_model: Optional[str] = None):
        
        self.embedding_provider_name = embedding_provider or DEFAULT_EMBEDDING_PROVIDER
        self.llm_provider_name = llm_provider or DEFAULT_LLM_PROVIDER
        self.llm_model_name = llm_model or DEFAULT_LLM_MODEL
        
        # Initialize subcomponents
        self.embeddings = get_embedding_provider(self.embedding_provider_name)
        self.llm = get_llm_provider(self.llm_provider_name, model_name=self.llm_model_name)
        
        self.retriever = HybridRetriever(self.embeddings)
        self.router = QueryRouter(self.llm)

    def calculate_technical_relevance(self, query: str) -> float:
        """Calculate confidence score indicating if a query is technical."""
        tech_keywords = [
            "api", "endpoint", "parameter", "request", "response", "authenticate", 
            "authentication", "token", "key", "curl", "python", "code", "sdk", 
            "json", "xml", "header", "method", "post", "get", "query", "lookup", 
            "format", "rate limit", "error", "dns", "whois lookup", "reverse", 
            "domain", "historical", "integration", "example"
        ]
        query_lower = query.lower()
        matches = sum(1 for kw in tech_keywords if kw in query_lower)
        if not matches:
            return 0.0
        # Each matching keyword yields 0.20 score, capped at 1.0 (5+ matches -> 100%)
        return min(1.0, matches * 0.20)

    def answer_query(self, query: str, memory: ConversationMemory) -> Dict[str, Any]:
        """
        Runs the complete RAG pipeline:
        1. Contextualizes the query based on conversation history
        2. Calculates Technical Relevance Score
        3. Routes and filters chunk retrieval based on confidence score (threshold 80%)
        4. Generates an answer using the LLM & formatted context
        5. Logs interaction into memory
        """
        start_time = time.time()
        metrics = {}
        
        # Step 1: Contextualize
        t0 = time.time()
        standalone_query = memory.contextualize_query(query, self.llm)
        t_contextualize = time.time() - t0
        
        # Check response cache
        from app.cache import response_cache
        cached_res = response_cache.get(standalone_query)
        if cached_res is not None:
            # Reconstruct response from cache with zeroed metrics for caching efficiency
            cached_res = cached_res.copy()
            cached_res["metrics"] = {
                "intent_classification_time": 0.0,
                "retrieval_time": 0.0,
                "reranking_time": 0.0,
                "context_build_time": 0.0,
                "llm_generation_time": 0.0,
                "total_response_time": time.time() - start_time,
                "cached": True
            }
            # Save to Conversational Memory
            memory.add_message("user", query)
            memory.add_message("assistant", cached_res["answer"])
            return cached_res
            
        # Step 2: Route & Retrieve Chunks
        t1 = time.time()
        routing = self.router.route_query(standalone_query)
        metrics["intent_classification_time"] = time.time() - t1
        
        t2 = time.time()
        tech_score = self.calculate_technical_relevance(standalone_query)
        
        # If Technical Confidence Score >= 80% or routed to technical collection,
        # query ONLY api_docs and strictly exclude company knowledge (Step 4 & 5)
        if tech_score >= 0.80 or routing == "api_docs":
            # Pass 1: Retrieve top relevant chunk candidates
            candidate_chunks = self.retriever.hybrid_search(standalone_query, "api_docs", k=3)
            candidate_chunks = [c for c in candidate_chunks if c["metadata"].get("document_type") == "api_docs"]
            
            if candidate_chunks:
                # Identify the top document file to retrieve its full content
                top_files = []
                seen_files = set()
                for c in candidate_chunks:
                    fname = c["metadata"].get("file_name", "")
                    if fname and fname not in seen_files:
                        seen_files.add(fname)
                        top_files.append(fname)
                        if len(top_files) >= 2:
                            break
                            
                # Pass 2: Retrieve all chunks for the top file to provide complete contiguous context
                chunks = []
                for fname in top_files:
                    file_data = self.retriever.api_collection.get(
                        where={"file_name": fname}
                    )
                    if file_data and file_data["documents"]:
                        for idx in range(len(file_data["ids"])):
                            meta = file_data["metadatas"][idx]
                            if "category" not in meta:
                                meta["category"] = "General"
                            chunks.append({
                                "id": file_data["ids"][idx],
                                "content": file_data["documents"][idx],
                                "metadata": meta
                            })
                chunks = chunks[:5]
            else:
                chunks = []
        elif routing == "company_knowledge":
            chunks = self.retriever.hybrid_search(standalone_query, "company_knowledge", k=2)
        else:  # Mixed Query or lower technical confidence
            chunks = self.retriever.hybrid_search_mixed(standalone_query, k=3)
            
        metrics["retrieval_time"] = time.time() - t2
        
        # Step 3: Reranking & Compression
        t3 = time.time()
        max_chunks = 3 if (tech_score >= 0.80 or routing == "api_docs" or routing == "mixed") else 2
        chunks = compress_context(chunks, max_chunks=max_chunks)
        metrics["reranking_time"] = time.time() - t3
        
        # Step 4: Context Construction
        t4 = time.time()
        if not chunks:
            prompt = ""
        else:
            prompt = construct_prompt(standalone_query, chunks)
        metrics["context_build_time"] = time.time() - t4
        
        # Step 5: Generate Answer
        t5 = time.time()
        if not chunks:
            answer = "I could not find this information in the indexed API documentation."
            sources = []
        else:
            try:
                answer = self.llm.generate(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    history=None  # Handle context inside the structured prompt
                )
                
                # Sanitize sections based on actual context contents to prevent formatting hallucinations
                answer = sanitize_sections(answer, chunks)
                
                # Standardize unavailable information response format if the response is just a short statement
                if len(answer.strip()) < 200 and ("could not find" in answer.lower() or "not present in the" in answer.lower()):
                    answer = "I could not find this information in the indexed API documentation."
            except Exception as e:
                answer = f"Error generating answer: {e}"
                
            # Extract unique sources
            sources = []
            seen_sources = set()
            for c in chunks:
                src = c["metadata"].get("source", "")
                title = c["metadata"].get("title", "")
                section = c["metadata"].get("section", "")
                file_name = c["metadata"].get("file_name", "")
                doc_type = c["metadata"].get("document_type", "")
                
                # De-duplicate citation metadata
                key = (src, section)
                if key not in seen_sources:
                    seen_sources.add(key)
                    category = c["metadata"].get("category", "General")
                    sources.append({
                        "source": src,
                        "title": title,
                        "section": section,
                        "file_name": file_name,
                        "document_type": doc_type,
                        "category": category,
                        "content": c["content"]
                    })
        metrics["llm_generation_time"] = time.time() - t5
        metrics["total_response_time"] = time.time() - start_time
        metrics["cached"] = False
        
        # Save to Conversational Memory (original query & output)
        memory.add_message("user", query)
        memory.add_message("assistant", answer)
        
        result = {
            "answer": answer,
            "sources": sources,
            "routing": routing,
            "contextualized_query": standalone_query,
            "retrieved_chunks": chunks,
            "metrics": metrics
        }
        
        # Cache final response
        response_cache.set(standalone_query, result)
        
        return result
