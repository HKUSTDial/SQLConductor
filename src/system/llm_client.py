"""
LLM client
Supports calls to locally deployed models
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
from openai import (
    OpenAI, AsyncOpenAI, OpenAIError,
    RateLimitError, APITimeoutError, APIConnectionError,
    InternalServerError, BadRequestError
)
from tenacity import (
    retry, retry_if_exception_type,
    stop_after_attempt, wait_random_exponential
)
from src.system.config import ModelConfig
from pydantic import BaseModel


class EmptyResponseError(Exception):
    """Custom exception for empty LLM responses, used to trigger retry."""
    pass


@dataclass
class LLMResponse:
    """LLM response"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    contents: Optional[List[str]] = None

class LLMClient:
    """Base LLM client."""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.async_client = AsyncOpenAI(
            base_url=model_config.base_url,
            api_key=model_config.api_key,
            max_retries=0,  # Tenacity handles retries.
            timeout=getattr(model_config, 'timeout', 120.0)
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate text"""
        raise NotImplementedError

class OpenAIClient(LLMClient):
    """OpenAI Compatible Client"""
    
    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(15),
        retry=retry_if_exception_type((
            RateLimitError, APITimeoutError, APIConnectionError,
            InternalServerError, BadRequestError, EmptyResponseError
        )),
        reraise=True
    )
    async def _call_llm(self, messages: List[Dict], temperature: float, max_tokens: int, stop: Optional[List[str]], **kwargs):
        """Internal LLM call with tenacity retry. Raises on recoverable errors."""
        
        # Pass vLLM-specific sampling options through extra_body.
        if "top_p" not in kwargs and self.model_config.top_p is not None:
            kwargs["top_p"] = self.model_config.top_p
            
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}
        if self.model_config.top_k is not None and "top_k" not in kwargs["extra_body"]:
            kwargs["extra_body"]["top_k"] = self.model_config.top_k
        if self.model_config.repetition_penalty is not None and "repetition_penalty" not in kwargs["extra_body"]:
            kwargs["extra_body"]["repetition_penalty"] = self.model_config.repetition_penalty
            
        if not kwargs["extra_body"]:
            kwargs.pop("extra_body")
            
        response = await self.async_client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs
        )
        
        if not response.choices:
            raise EmptyResponseError(f"No response from the model: {response}")
        
        for choice in response.choices:
            if choice.message.content is None or choice.message.content.strip() == "":
                raise EmptyResponseError(f"Model returned empty content (possibly filtered): {response}")
        
        return response
    
    async def generate(self, 
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop: Optional[List[str]] = None,
                      **kwargs) -> LLMResponse:
        """Generate text"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await self._call_llm(
                messages=messages,
                temperature=self.model_config.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.model_config.max_tokens,
                stop=stop,
                **kwargs
            )
            choices_contents = [choice.message.content for choice in response.choices]
            
            return LLMResponse(
                content=choices_contents[0] if choices_contents else "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                finish_reason=response.choices[0].finish_reason if response.choices else None,
                contents=choices_contents
            )
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return LLMResponse(
                content="",
                model=self.model_config.model_name,
                error=str(e)
            )

class AsyncLLMClient(LLMClient):
    """Async LLM client."""
    
    async def generate(self, 
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop: Optional[List[str]] = None,
                      **kwargs) -> LLMResponse:
        """Generate text asynchronously."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.model_config.api_key}"
                }
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                data = {
                    "model": self.model_config.model_name,
                    "messages": messages,
                    "temperature": self.model_config.temperature if temperature is None else temperature,
                    "max_tokens": max_tokens or self.model_config.max_tokens,
                    "stop": stop,
                    "top_p": self.model_config.top_p,
                    **kwargs
                }
                
                if self.model_config.top_k is not None:
                    data["top_k"] = self.model_config.top_k
                if self.model_config.repetition_penalty is not None:
                    data["repetition_penalty"] = self.model_config.repetition_penalty
                
                async with session.post(
                    f"{self.model_config.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.model_config.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        choices = result.get("choices", [])
                        choices_contents = [choice["message"]["content"] for choice in choices]
                        return LLMResponse(
                            content=choices_contents[0] if choices_contents else "",
                            model=result["model"],
                            usage=result.get("usage"),
                            finish_reason=choices[0].get("finish_reason") if choices else None,
                            contents=choices_contents
                        )
                    else:
                        error_text = await response.text()
                        logger.error(f"LLM API error: {response.status} - {error_text}")
                        return LLMResponse(
                            content="",
                            model=self.model_config.model_name,
                            error=f"API error: {response.status}"
                        )
                        
        except asyncio.TimeoutError:
            logger.error("LLM generation timeout")
            return LLMResponse(
                content="",
                model=self.model_config.model_name,
                error="Timeout"
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return LLMResponse(
                content="",
                model=self.model_config.model_name,
                error=str(e)
            )

class LLMClientFactory:
    """LLM client factory."""
    
    @staticmethod
    def create_client(model_config: ModelConfig, use_async: bool = False) -> LLMClient:
        """Create an LLM client."""
        if use_async:
            return AsyncLLMClient(model_config)
        else:
            return OpenAIClient(model_config)

_llm_clients: Dict[str, LLMClient] = {}

def get_llm_client(model_config: ModelConfig, use_async: bool = False) -> LLMClient:
    """Get an LLM client with caching."""
    cache_key = f"{model_config.base_url}_{model_config.model_name}_{use_async}"
    
    if cache_key not in _llm_clients:
        _llm_clients[cache_key] = LLMClientFactory.create_client(model_config, use_async)
    
    return _llm_clients[cache_key] 
