import os
from typing import Dict, List, Optional, Union
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from .infinigence_embeddings import InfinigenceEmbeddings, QwenEmbeddings, FLowEmbeddings, LocalEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
import requests
logger = logging.getLogger("websocietysimulator")

class RateLimitError(Exception):
    pass

class LLMBase:
    def __init__(self, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize LLM base class
        
        Args:
            model: Model name, defaults to deepseek-chat
        """
        self.model = model
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call LLM to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        raise NotImplementedError("Subclasses need to implement this method")
    
    def get_embedding_model(self):
        """
        Get the embedding model for text embeddings
        
        Returns:
            OpenAIEmbeddings: An instance of OpenAI's text embedding model
        """
        raise NotImplementedError("Subclasses need to implement this method")

class InfinigenceLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize Deepseek LLM
        
        Args:
            api_key: Deepseek API key
            model: Model name, defaults to qwen2.5-72b-instruct
        """
        super().__init__(model)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://cloud.infini-ai.com/maas/v1"
        )
        self.embedding_model = InfinigenceEmbeddings(api_key=api_key)
        
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=10, max=300),  # 等待时间从10秒开始，指数增长，最长300秒
        stop=stop_after_attempt(10)  # 最多重试10次
    )
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call Infinigence AI API to get response with rate limit handling
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_strs,
                n=n,
            )
            
            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        except Exception as e:
            if "429" in str(e):
                logger.warning("Rate limit exceeded")
                raise RateLimitError("Rate limit exceeded") from e
            else:
                logger.error(f"LLM Error: {e}")
                raise e
    
    def get_embedding_model(self):
        return self.embedding_model

class QwenLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize Qwen LLM
        
        Args:
            api_key: Qwen API key
            model: Model name, defaults to qwen2.5-72b-instruct
        """
        super().__init__(model)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.embedding_model = QwenEmbeddings(api_key=api_key)
        self.usage_input = 0
        self.usage_output = 0
        
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),  # 等待时间从10秒开始，指数增长，最长300秒
        stop=stop_after_attempt(10)  # 最多重试10次
    )
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call Infinigence AI API to get response with rate limit handling
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_strs,
                n=n,
            )
            self.usage_input += response.usage.prompt_tokens
            self.usage_output += response.usage.completion_tokens
            
            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        except Exception as e:
            if "429" in str(e):
                logger.warning("Rate limit exceeded")
            else:
                logger.error(f"Other LLM Error: {e}")
            raise e
    
    def get_embedding_model(self):
        return self.embedding_model
    
    def get_usage(self):
        return self.usage_input, self.usage_output
    
class FlowLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "Qwen/Qwen2.5-72B-Instruct"):
        """
        Initialize Qwen LLM
        
        Args:
            api_key: Qwen API key
            model: Model name, defaults to qwen2.5-72b-instruct
        """
        super().__init__(model)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        self.embedding_model = FLowEmbeddings(api_key=api_key)
        self.usage_input = 0
        self.usage_output = 0
        
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),  # 等待时间从10秒开始，指数增长，最长300秒
        stop=stop_after_attempt(10)  # 最多重试10次
    )
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call Infinigence AI API to get response with rate limit handling
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_strs,
                n=n,
            )
            self.usage_input += response.usage.prompt_tokens
            self.usage_output += response.usage.completion_tokens
            
            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        except Exception as e:
            if "429" in str(e):
                logger.warning("Rate limit exceeded")
            else:
                logger.error(f"Other LLM Error: {e}")
            raise e
    
    def get_embedding_model(self):
        return self.embedding_model
    
    def get_usage(self):
        return self.usage_input, self.usage_output

class OpenAILLM(LLMBase):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI LLM
        
        Args:
            api_key: OpenAI API key
            model: Model name, defaults to gpt-3.5-turbo
        """
        super().__init__(model)
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = OpenAIEmbeddings(api_key=api_key)
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call OpenAI API to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_strs,
            n=n
        )
        
        if n == 1:
            return response.choices[0].message.content
        else:
            return [choice.message.content for choice in response.choices]
    
    def get_embedding_model(self):
        return self.embedding_model 


def _ollama_is_gemma4(model_name: str) -> bool:
    """Gemma 4 tags look like gemma4:12b / gemma4:e2b (thinking on by default)."""
    return "gemma4" in (model_name or "").lower()


def _strip_ollama_thinking(text: str) -> str:
    """Drop reasoning preambles that can leak into content for thinking models."""
    import re

    if not text:
        return text
    # Ollama / Gemma-style blocks
    text = re.sub(
        r"(?is)^\s*Thinking\.\.\..*?\.\.\.done thinking\.\s*",
        "",
        text,
    )
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    text = re.sub(r"(?is)<\|think\|>.*?<\|/think\|>\s*", "", text)
    return text.strip()


class OllamaLLM(LLMBase):
    def __init__(self, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize Ollama LLM
        
        Args:
            model: Model name, defaults to qwen2.5-72b-instruct
        """
        super().__init__(model)
        self._embedding_model = None

    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, \
                 temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, \
                    n: int = 1) -> Union[str, List[str]]:

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        # Force CPU-only inference when OLLAMA_NUM_GPU is set (e.g. "0" to keep the
        # GPU free for a concurrent training job). Ollama interprets num_gpu as the
        # number of layers offloaded to the GPU; 0 => pure CPU.
        num_gpu_env = os.environ.get("OLLAMA_NUM_GPU")
        if num_gpu_env is not None and num_gpu_env.strip() != "":
            try:
                options["num_gpu"] = int(num_gpu_env)
            except ValueError:
                pass

        model_name = model or self.model
        # Keep Qwen/Llama payloads bit-identical to prior runs (no `think` field).
        # Only Gemma 4 burns num_predict on Thinking… and truncates before the list.
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if _ollama_is_gemma4(model_name):
            # Default off for ranking. OLLAMA_THINK=1 to re-enable for debugging.
            think_env = os.environ.get("OLLAMA_THINK", "0").strip().lower()
            payload["think"] = think_env in ("1", "true", "yes", "on")

        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=int(os.environ.get("OLLAMA_TIMEOUT", "120"))
            )

            response.raise_for_status()
            data = response.json()

            # safe extraction
            if "message" in data and "content" in data["message"]:
                res = data["message"]["content"]
            elif "response" in data:
                res = data["response"]
            else:
                raise ValueError(f"Unexpected Ollama response: {data}")

            if _ollama_is_gemma4(model_name):
                res = _strip_ollama_thinking(str(res or ""))

        except Exception as e:
            print(f"Ollama error: {e}")
            res = ""

        return res

    def get_embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = LocalEmbedding()
        return self._embedding_model

            
