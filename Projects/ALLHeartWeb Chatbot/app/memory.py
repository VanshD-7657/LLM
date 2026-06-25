from typing import List, Dict, Optional
from app.llm import LLMProvider

class ConversationMemory:
    def __init__(self, limit: int = 10):
        # Format: [{"role": "user" or "assistant", "content": "text"}]
        self.history: List[Dict[str, str]] = []
        self.limit = limit

    def add_message(self, role: str, content: str):
        """Adds a message to the memory history, maintaining the history limit."""
        self.history.append({"role": role, "content": content})
        # Keep history within limit
        if len(self.history) > self.limit:
            self.history.pop(0)

    def get_history(self) -> List[Dict[str, str]]:
        """Returns the conversation history."""
        return self.history

    def clear(self):
        """Clears all conversation history."""
        self.history = []

    def contextualize_query(self, query: str, llm_provider: LLMProvider) -> str:
        """
        Rewrites the query using conversation history to make it self-contained for search.
        If no history exists, returns the query unchanged.
        """
        if not self.history:
            return query

        system_prompt = (
            "You are a helpful customer support assistant.\n"
            "Given the conversation history and a follow-up query, rewrite the follow-up query to be a standalone, "
            "independent query that retains all context and references (like 'it', 'they', 'this', etc.) from the conversation.\n"
            "Do NOT answer the question. Do NOT add extra explanations. Simply return the rewritten query text."
        )

        # Build history transcript
        transcript = ""
        for message in self.history:
            role = "Customer" if message["role"] == "user" else "Assistant"
            transcript += f"{role}: {message['content']}\n"

        prompt = (
            f"Conversation History:\n{transcript}\n"
            f"Follow-up Query: {query}\n\n"
            f"Standalone Query:"
        )

        try:
            standalone_query = llm_provider.generate(prompt=prompt, system_prompt=system_prompt)
            standalone_query = standalone_query.strip()
            
            # Clean response
            standalone_query = standalone_query.strip('"').strip("'")
            if standalone_query:
                return standalone_query
        except Exception as e:
            print(f"Error during query contextualization: {e}. Using original query.")
            
        return query
