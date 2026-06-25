import re
from typing import Dict, Any, Optional
from app.llm import LLMProvider

class QueryRouter:
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider
        
        # Keyword patterns for rule-based fallback
        self.api_keywords = [
            r"\bapi\b", r"\bendpoint\b", r"\bparameters?\b", r"\brequest\b", 
            r"\bresponse\b", r"\bauthenticate\b", r"\bauthentication\b", 
            r"\btoken\b", r"\bkey\b", r"\bcurl\b", r"\bpython\b", r"\bcode\b", 
            r"\bsdk\b", r"\bjson\b", r"\bxml\b", r"\bheaders?\b", r"\bmethods?\b", 
            r"\bpost\b", r"\bget\b", r"\bquery\b", r"\blookup\b", r"\bformat\b",
            r"\brate limit\b", r"\berrors?\b", r"\bdns\b", r"\bwhois lookup\b",
            r"\breverse\b", r"\bdomain\b", r"\bhistorical\b"
        ]
        
        self.knowledge_keywords = [
            r"\babout\b", r"\bwho is\b", r"\bwhois data center\b", r"\bservices?\b", 
            r"\bproducts?\b", r"\bpricing\b", r"\bmission\b", r"\bvision\b", 
            r"\boverview\b", r"\bcompany\b", r"\bhistory\b", r"\bcareers\b",
            r"\bcontact\b", r"\bphone\b", r"\bemail\b", r"\baddress\b",
            r"\ballheart\b", r"\bparent\b", r"\bowner\b"
        ]

    def _rule_based_classify(self, query: str) -> str:
        """Fallback keyword-based classifier."""
        query_lower = query.lower()
        
        has_api = any(re.search(pattern, query_lower) for pattern in self.api_keywords)
        has_knowledge = any(re.search(pattern, query_lower) for pattern in self.knowledge_keywords)
        
        if has_api and has_knowledge:
            return "mixed"
        elif has_api:
            return "api_docs"
        elif has_knowledge:
            return "company_knowledge"
        else:
            # Default fallback
            return "mixed"

    def route_query(self, query: str) -> str:
        """
        Classifies incoming query to route to:
        - 'api_docs'
        - 'company_knowledge'
        - 'mixed'
        """
        query_lower = query.lower()
        
        # High-confidence direct routing rules to bypass LLM latency
        has_api_strong = any(re.search(r'\b' + kw + r'\b', query_lower) for kw in [
            "api", "endpoint", "endpoints", "curl", "python", "requests", "authenticate", 
            "authentication", "bearer", "headers", "parameter", "parameters", "json", 
            "xml", "csv", "sdk", "sdks", "github", "error", "errors", "http", "get", "post"
        ])
        has_knowledge_strong = any(re.search(r'\b' + kw + r'\b', query_lower) for kw in [
            "pricing", "price", "prices", "mission", "vision", "about", "company", 
            "whois data center", "allheart", "all heart", "owner", "ownership", "history", 
            "careers", "contact", "address", "phone", "email"
        ])
        
        if has_api_strong and not has_knowledge_strong:
            return "api_docs"
        elif has_knowledge_strong and not has_api_strong:
            return "company_knowledge"
            
        if not self.llm_provider:
            return self._rule_based_classify(query)
            
        system_prompt = (
            "You are an expert query router for a technical RAG assistant.\n"
            "Your task is to classify user queries into one of three categories:\n"
            "1. 'api_docs' (Technical Query): Questions about using endpoints, API requests/responses, code examples (Python, cURL, JS, SDKs), request parameters, authentication examples, integrations, and errors. Example: 'Give me Python code for Reverse WHOIS'.\n"
            "2. 'company_knowledge' (Business Query): Questions about what the company is, services offered, pricing, company background, features, benefits, or mission. Example: 'What pricing plans do they offer?'.\n"
            "3. 'mixed' (Mixed Query): Queries asking for BOTH technical API details and general company/business information. Example: 'What is WHOIS Data Center and how do I authenticate with their API?'.\n\n"
            "Respond ONLY with one of the three strings: 'api_docs', 'company_knowledge', or 'mixed'. Do not add any markdown, reasoning, or punctuation."
        )
        
        prompt = f"Classify the following query:\n\"{query}\""
        
        try:
            response = self.llm_provider.generate(prompt=prompt, system_prompt=system_prompt)
            classification = response.strip().lower().replace("'", "").replace('"', "")
            
            # Match strictly to expected outcomes
            if classification in ["api_docs", "company_knowledge", "mixed"]:
                return classification
            
            # Fallback regex search on response
            if "mixed" in classification:
                return "mixed"
            elif "api" in classification or "doc" in classification:
                return "api_docs"
            elif "company" in classification or "knowledge" in classification:
                return "company_knowledge"
                
        except Exception as e:
            print(f"LLM Routing failed: {e}. Falling back to rule-based router.")
            
        return self._rule_based_classify(query)
