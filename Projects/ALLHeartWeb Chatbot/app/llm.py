import abc
import time
from typing import List, Dict, Optional
import google.generativeai as genai
from app.config import GEMINI_API_KEY, OPENAI_API_KEY, GROQ_API_KEY

def call_with_retry(api_func, *args, max_retries=5, initial_delay=5.0, backoff_factor=2.0, **kwargs):
    """Call an LLM API function with retry logic and exponential backoff, handling rate limits."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return api_func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            print(f"LLM API call failed: {e}. Retrying in {delay:.2f}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
            delay *= backoff_factor

class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None, 
                 history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate a response from the LLM."""
        pass


class GeminiLLMProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
        genai.configure(api_key=GEMINI_API_KEY)

    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None, 
                 history: Optional[List[Dict[str, str]]] = None) -> str:
        # Construct model with system prompt if provided
        generation_config = {
            "temperature": 0.1,  # Low temperature for factual RAG answers
            "top_p": 0.95,
        }
        
        # Initialize model
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=generation_config
        )
        
        if history:
            # Format history for google-generativeai chat interface
            # History structure: [{"role": "user"/"assistant", "content": "text"}]
            # Gemini expects roles to be "user" or "model"
            chat_history = []
            for h in history:
                role = "user" if h["role"] == "user" else "model"
                chat_history.append({
                    "role": role,
                    "parts": [h["content"]]
                })
            
            chat = model.start_chat(history=chat_history)
            response = call_with_retry(chat.send_message, prompt)
        else:
            response = call_with_retry(model.generate_content, prompt)
            
        return response.text


class OpenAILLMProvider(LLMProvider):
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None, 
                 history: Optional[List[Dict[str, str]]] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            for h in history:
                messages.append({"role": h["role"], "content": h["content"]})
                
        messages.append({"role": "user", "content": prompt})
        
        response = call_with_retry(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content


class GroqLLMProvider(LLMProvider):
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
        from openai import OpenAI
        # Groq is OpenAI-compatible
        self.client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None, 
                 history: Optional[List[Dict[str, str]]] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            for h in history:
                messages.append({"role": h["role"], "content": h["content"]})
                
        messages.append({"role": "user", "content": prompt})
        
        response = call_with_retry(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, model_name: str = "claude-3-5-sonnet-latest"):
        self.model_name = model_name
        import os
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment variables.")
        from anthropic import Anthropic
        self.client = Anthropic(api_key=anthropic_key)

    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None, 
                 history: Optional[List[Dict[str, str]]] = None) -> str:
        messages = []
        if history:
            for h in history:
                messages.append({"role": h["role"], "content": h["content"]})
        
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.model_name,
            "max_tokens": 4000,
            "temperature": 0.1,
            "messages": messages
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        response = call_with_retry(self.client.messages.create, **kwargs)
        return response.content[0].text


def get_llm_provider(provider_name: str = "gemini", model_name: str = "gemini-3.5-flash") -> LLMProvider:
    provider_name = provider_name.lower()
    if provider_name == "gemini":
        return GeminiLLMProvider(model_name=model_name)
    elif provider_name == "openai":
        return OpenAILLMProvider(model_name=model_name)
    elif provider_name == "groq":
        return GroqLLMProvider(model_name=model_name)
    elif provider_name == "anthropic" or provider_name == "claude":
        return AnthropicLLMProvider(model_name=model_name)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
