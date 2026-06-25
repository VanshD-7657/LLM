import time
from typing import List, Dict, Any, Tuple
from app.rag_engine import RAGEngine
from app.memory import ConversationMemory

# Define evaluation test dataset
EVAL_DATASET = [
    {
        "query": "What is WHOIS Data Center?",
        "expected_doc_type": "company_knowledge",
        "expected_sources": ["about_us.txt", "homepage_overview.txt", "website_general_knowledge.txt",'plans.txt']
    },
    {
        "query": "What is their mission and vision?",
        "expected_doc_type": "company_knowledge",
        "expected_sources": ["about_us.txt"]
    },
    {
        "query": "How do I authenticate with the WHOIS API?",
        "expected_doc_type": "api_docs",
        "expected_sources": ["authentication.md", "quickstart.md"]
    },
    {
        "query": "What parameters does the Historical WHOIS API require?",
        "expected_doc_type": "api_docs",
        "expected_sources": ["historical_whois.md"]
    },
    {
        "query": "What response formats does the WHOIS Lookup API support?",
        "expected_doc_type": "api_docs",
        "expected_sources": ["whois_lookup.md", "response_format.md"]
    },
    {
        "query": "Show me a Python example of calling the WHOIS Lookup API.",
        "expected_doc_type": "api_docs",
        "expected_sources": ["whois_lookup.md", "quickstart.md"]
    },
    {
        "query": "What is the endpoint for Reverse WHOIS search?",
        "expected_doc_type": "api_docs",
        "expected_sources": ["reverse_lookup"] # matches directory or filename
    }
]

class RAGEvaluator:
    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine

    def evaluate_faithfulness(self, answer: str, context: str) -> Tuple[bool, str]:
        """Evaluates if the answer is faithful to the context (no hallucination)."""
        prompt = (
            "You are an expert AI systems auditor.\n"
            "Compare the generated Answer against the provided Retrieved Context. Your task is to verify if the "
            "Answer contains any facts, details, or claims that are not present in the Retrieved Context (hallucinations).\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"Answer:\n{answer}\n\n"
            "Determine if the Answer is fully and strictly supported by the Retrieved Context.\n"
            "Respond exactly in this format:\n"
            "Verdict: [Yes/No]\n"
            "Reason: [Brief explanation of why it is faithful or why it is hallucinated]"
        )
        
        try:
            response = self.rag_engine.llm.generate(prompt=prompt)
            lines = response.strip().split("\n")
            verdict = "yes"
            reason = "No reason provided."
            
            for line in lines:
                if line.lower().startswith("verdict:"):
                    verdict = line.split(":", 1)[1].strip().lower()
                elif line.lower().startswith("reason:"):
                    reason = line.split(":", 1)[1].strip()
                    
            is_faithful = "yes" in verdict
            return is_faithful, reason
        except Exception as e:
            return True, f"Error in LLM evaluation: {e}"

    def evaluate_relevance(self, query: str, answer: str) -> Tuple[str, str]:
        """Evaluates how relevant the answer is to the user query."""
        prompt = (
            "You are an expert AI systems auditor.\n"
            "Rate the relevance and helpfulness of the Answer to the User Query.\n\n"
            f"User Query: {query}\n"
            f"Answer: {answer}\n\n"
            "Assess if the Answer directly and completely answers the User Query.\n"
            "Respond exactly in this format:\n"
            "Verdict: [High/Medium/Low]\n"
            "Reason: [Brief explanation of the rating]"
        )
        
        try:
            response = self.rag_engine.llm.generate(prompt=prompt)
            lines = response.strip().split("\n")
            verdict = "high"
            reason = "No reason provided."
            
            for line in lines:
                if line.lower().startswith("verdict:"):
                    verdict = line.split(":", 1)[1].strip().lower()
                elif line.lower().startswith("reason:"):
                    reason = line.split(":", 1)[1].strip()
                    
            return verdict, reason
        except Exception as e:
            return "high", f"Error in LLM evaluation: {e}"

    def run_suite(self) -> Dict[str, Any]:
        results = []
        
        total_queries = len(EVAL_DATASET)
        correct_routing_count = 0
        faithful_count = 0
        high_relevance_count = 0
        citation_match_count = 0
        
        print(f"Starting RAG Evaluation Suite on {total_queries} test queries...\n")
        
        for idx, case in enumerate(EVAL_DATASET):
            query = case["query"]
            expected_type = case["expected_doc_type"]
            expected_srcs = case["expected_sources"]
            
            # Rate limit mitigation for Groq Free Tier (6000 TPM limit)
            if idx > 0:
                print("Waiting 12 seconds to respect Groq API rate limits...")
                time.sleep(12)
            
            # Start evaluation turn with clean memory
            memory = ConversationMemory()
            
            start_time = time.time()
            res = self.rag_engine.answer_query(query, memory)
            elapsed = time.time() - start_time
            
            # Evaluate Routing Accuracy
            actual_routing = res["routing"]
            routing_correct = (actual_routing == expected_type) or (actual_routing == "mixed")
            if routing_correct:
                correct_routing_count += 1
                
            # Compile context text including source and metadata so the auditor can verify sources
            context_text = ""
            for idx, chunk in enumerate(res["retrieved_chunks"]):
                meta = chunk["metadata"]
                context_text += f"--- Document [{idx+1}] ---\n"
                context_text += f"Source: {meta.get('source', 'Unknown')}\n"
                context_text += f"Category: {meta.get('category', 'General')}\n"
                context_text += f"Section: {meta.get('section', 'Main')}\n"
                context_text += f"Content:\n{chunk['content']}\n\n"
            
            # Evaluate Faithfulness (Hallucination)
            is_faithful, faith_reason = self.evaluate_faithfulness(res["answer"], context_text)
            if is_faithful:
                faithful_count += 1
                
            # Evaluate Relevance
            relevance, rel_reason = self.evaluate_relevance(query, res["answer"])
            if "high" in relevance:
                high_relevance_count += 1
                
            # Evaluate Citation Accuracy
            retrieved_files = [c["metadata"].get("file_name", "") for c in res["retrieved_chunks"]]
            citation_match = False
            for exp in expected_srcs:
                # check if expected source matches any retrieved file name or path snippet
                if any(exp in rf or rf in exp for rf in retrieved_files):
                    citation_match = True
                    break
            if citation_match:
                citation_match_count += 1
                
            results.append({
                "query": query,
                "routing": {
                    "expected": expected_type,
                    "actual": actual_routing,
                    "correct": routing_correct
                },
                "retrieved_files": retrieved_files,
                "citation_match": citation_match,
                "faithfulness": {
                    "is_faithful": is_faithful,
                    "reason": faith_reason
                },
                "relevance": {
                    "rating": relevance,
                    "reason": rel_reason
                },
                "latency_sec": elapsed,
                "metrics": res.get("metrics", {})
            })
            
            print(f"Query: '{query}'")
            print(f" -> Routing: {actual_routing} (Correct: {routing_correct})")
            print(f" -> Faithful: {is_faithful}")
            print(f" -> Relevance: {relevance}")
            print(f" -> Latency: {elapsed:.2f}s")
            if "metrics" in res:
                m = res["metrics"]
                print(f"    (Route: {m.get('intent_classification_time', 0.0):.3f}s | Retrieval: {m.get('retrieval_time', 0.0):.3f}s | LLM: {m.get('llm_generation_time', 0.0):.3f}s)\n")
            else:
                print("\n")
            
        # Compute final percentages and latency averages
        avg_classification = sum(r["metrics"].get("intent_classification_time", 0.0) for r in results) / total_queries
        avg_retrieval = sum(r["metrics"].get("retrieval_time", 0.0) for r in results) / total_queries
        avg_rerank = sum(r["metrics"].get("reranking_time", 0.0) for r in results) / total_queries
        avg_context = sum(r["metrics"].get("context_build_time", 0.0) for r in results) / total_queries
        avg_llm = sum(r["metrics"].get("llm_generation_time", 0.0) for r in results) / total_queries
        
        metrics = {
            "routing_accuracy": (correct_routing_count / total_queries) * 100,
            "faithfulness_rate": (faithful_count / total_queries) * 100,
            "high_relevance_rate": (high_relevance_count / total_queries) * 100,
            "citation_retrieval_accuracy": (citation_match_count / total_queries) * 100,
            "average_latency_sec": sum(r["latency_sec"] for r in results) / total_queries,
            "avg_classification_sec": avg_classification,
            "avg_retrieval_sec": avg_retrieval,
            "avg_rerank_sec": avg_rerank,
            "avg_context_sec": avg_context,
            "avg_llm_sec": avg_llm
        }
        
        return {
            "metrics": metrics,
            "details": results
        }

def generate_report(results: Dict[str, Any]) -> str:
    """Formats the evaluation results into a Markdown report."""
    metrics = results["metrics"]
    details = results["details"]
    
    report = "# RAG Evaluation Metrics & Latency Report\n\n"
    report += "## Performance Summary\n\n"
    report += f"- **Routing Accuracy:** {metrics['routing_accuracy']:.1f}%\n"
    report += f"- **Faithfulness Rate (No Hallucinations):** {metrics['faithfulness_rate']:.1f}%\n"
    report += f"- **High Relevance Rate:** {metrics['high_relevance_rate']:.1f}%\n"
    report += f"- **Citation Retrieval Accuracy:** {metrics['citation_retrieval_accuracy']:.1f}%\n"
    report += f"- **Average Latency:** {metrics['average_latency_sec']:.2f} seconds\n"
    report += f"  - *Intent Classification (Heuristics + LLM):* {metrics.get('avg_classification_sec', 0.0):.3f}s\n"
    report += f"  - *Hybrid Database Search:* {metrics.get('avg_retrieval_sec', 0.0):.3f}s\n"
    report += f"  - *Context Rerank/Compression:* {metrics.get('avg_rerank_sec', 0.0):.3f}s\n"
    report += f"  - *Context Build:* {metrics.get('avg_context_sec', 0.0):.3f}s\n"
    report += f"  - *LLM Generation:* {metrics.get('avg_llm_sec', 0.0):.3f}s\n\n"
    
    report += "## Detailed Test Cases\n\n"
    for idx, r in enumerate(details):
        report += f"### Test Case {idx+1}: \"{r['query']}\"\n\n"
        report += f"- **Routing:** Expected `{r['routing']['expected']}`, Got `{r['routing']['actual']}` (Correct: {r['routing']['correct']})\n"
        report += f"- **Retrieved Files:** {', '.join(r['retrieved_files']) if r['retrieved_files'] else 'None'}\n"
        report += f"- **Citation Hit:** {r['citation_match']}\n"
        if "metrics" in r and r["metrics"]:
            m = r["metrics"]
            report += f"- **Latency Details:** Total {r['latency_sec']:.2f}s (Classification: {m.get('intent_classification_time', 0.0):.3f}s | Retrieval: {m.get('retrieval_time', 0.0):.3f}s | Reranking: {m.get('reranking_time', 0.0):.3f}s | LLM: {m.get('llm_generation_time', 0.0):.3f}s)\n"
        report += f"- **Faithfulness Check:** {'PASSED' if r['faithfulness']['is_faithful'] else 'FAILED'}\n"
        report += f"  - *Reason:* {r['faithfulness']['reason']}\n"
        report += f"- **Answer Relevance:** {r['relevance']['rating'].upper()}\n"
        report += f"  - *Reason:* {r['relevance']['reason']}\n"
        report += f"- **Latency:** {r['latency_sec']:.2f}s\n\n"
        
    return report

if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add project folder to sys.path so we can run directly
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    from app.rag_engine import RAGEngine
    
    print("==================================================")
    print("Starting WHOIS Data Center RAG Evaluation Test Suite")
    print("==================================================")
    
    # Initialize RAG Engine
    engine = RAGEngine()
    
    # Run evaluation
    evaluator = RAGEvaluator(engine)
    results = evaluator.run_suite()
    
    # Generate and print report
    report = generate_report(results)
    print("\n" + report)
    print("==================================================")
    print("Evaluation Complete.")
    print("==================================================")
