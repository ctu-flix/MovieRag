import base64
import hashlib
import io
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

try:
    from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError
except ImportError:
    AsyncOpenAI = None
    AsyncAzureOpenAI = None

    class APIConnectionError(Exception):
        pass

    class RateLimitError(Exception):
        pass

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )
except ImportError:
    def retry(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None

    def retry_if_exception_type(*args, **kwargs):
        return None

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs, EmbeddingFunc, logger
from .base import BaseKVStorage

global_openai_async_client = None
global_azure_openai_async_client = None
global_ollama_client = None
global_deepseek_async_client = None
global_st_embedder = None
global_st_model_name = None


def _get_sentence_transformer_embedder(model_name: str = "BAAI/bge-base-en-v1.5"):
    global global_st_embedder, global_st_model_name
    resolved = (
        os.environ.get("MOVIERAG_ST_MODEL")
        if model_name in ("nomic-embed-text", "", None)
        else model_name
    ) or "BAAI/bge-base-en-v1.5"
    if global_st_embedder is None or global_st_model_name != resolved:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading local CPU SentenceTransformer embedding model: {resolved}")
        global_st_embedder = SentenceTransformer(resolved, trust_remote_code=True)
        global_st_model_name = resolved
    return global_st_embedder



def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        if AsyncOpenAI is None:
            raise ImportError("openai package is required to initialize AsyncOpenAI client.")
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        if AsyncAzureOpenAI is None:
            raise ImportError("openai package is required to initialize AsyncAzureOpenAI client.")
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client


def get_ollama_async_client_instance():
    global global_ollama_client
    if global_ollama_client is None:
        from ollama import AsyncClient

        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        global_ollama_client = AsyncClient(host=host)
    return global_ollama_client


def get_deepseek_async_client_instance(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    global global_deepseek_async_client
    resolved_key = (
        api_key
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "sk-placeholder"
    )
    resolved_url = (
        base_url
        or os.environ.get("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com"
    )
    if api_key is not None or base_url is not None:
        if AsyncOpenAI is None:
            return None
        return AsyncOpenAI(api_key=resolved_key, base_url=resolved_url)
    if global_deepseek_async_client is None and AsyncOpenAI is not None:
        global_deepseek_async_client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
        )
    return global_deepseek_async_client


# Setup LLM Configuration.
@dataclass
class LLMConfig:
    # To be set
    embedding_func_raw: callable
    embedding_model_name: str
    embedding_dim: int
    embedding_max_token_size: int
    embedding_batch_num: int
    embedding_func_max_async: int
    query_better_than_threshold: float

    best_model_func_raw: callable
    best_model_name: str
    best_model_max_token_size: int
    best_model_max_async: int

    cheap_model_func_raw: callable
    cheap_model_name: str
    cheap_model_max_token_size: int
    cheap_model_max_async: int

    # Assigned in post init
    embedding_func: EmbeddingFunc = None
    best_model_func: callable = None
    cheap_model_func: callable = None

    def __post_init__(self):
        embedding_wrapper = wrap_embedding_func_with_attrs(
            embedding_dim=self.embedding_dim,
            max_token_size=self.embedding_max_token_size,
            model_name=self.embedding_model_name,
        )
        self.embedding_func = embedding_wrapper(self.embedding_func_raw)
        self.best_model_func = lambda prompt, *args, **kwargs: self.best_model_func_raw(
            self.best_model_name, prompt, *args, **kwargs
        )
        self.cheap_model_func = lambda prompt, *args, **kwargs: self.cheap_model_func_raw(
            self.cheap_model_name, prompt, *args, **kwargs
        )


##### OpenAI Configuration
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def gpt_4o_complete(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data], dtype=np.float32)


openai_config = LLMConfig(
    embedding_func_raw=openai_embedding,
    embedding_model_name="text-embedding-3-small",
    embedding_dim=1536,
    embedding_max_token_size=8192,
    embedding_batch_num=32,
    embedding_func_max_async=16,
    query_better_than_threshold=0.2,
    best_model_func_raw=gpt_4o_complete,
    best_model_name="gpt-4o",
    best_model_max_token_size=32768,
    best_model_max_async=16,
    cheap_model_func_raw=gpt_4o_mini_complete,
    cheap_model_name="gpt-4o-mini",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=16,
)

openai_4o_mini_config = LLMConfig(
    embedding_func_raw=openai_embedding,
    embedding_model_name="text-embedding-3-small",
    embedding_dim=1536,
    embedding_max_token_size=8192,
    embedding_batch_num=32,
    embedding_func_max_async=16,
    query_better_than_threshold=0.2,
    best_model_func_raw=gpt_4o_mini_complete,
    best_model_name="gpt-4o-mini",
    best_model_max_token_size=32768,
    best_model_max_async=16,
    cheap_model_func_raw=gpt_4o_mini_complete,
    cheap_model_name="gpt-4o-mini",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=16,
)


###### Azure OpenAI Configuration
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data], dtype=np.float32)


azure_openai_config = LLMConfig(
    embedding_func_raw=azure_openai_embedding,
    embedding_model_name="text-embedding-3-small",
    embedding_dim=1536,
    embedding_max_token_size=8192,
    embedding_batch_num=32,
    embedding_func_max_async=16,
    query_better_than_threshold=0.2,
    best_model_func_raw=azure_gpt_4o_complete,
    best_model_name="gpt-4o",
    best_model_max_token_size=32768,
    best_model_max_async=16,
    cheap_model_func_raw=azure_gpt_4o_mini_complete,
    cheap_model_name="gpt-4o-mini",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=16,
)


###### Ollama Configuration
async def ollama_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    ollama_client = get_ollama_async_client_instance()

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await ollama_client.chat(model=model, messages=messages)

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": response["message"]["content"], "model": model}}
        )
        await hashing_kv.index_done_callback()

    return response["message"]["content"]


async def ollama_complete(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await ollama_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_mini_complete(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await ollama_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def ollama_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    try:
        ollama_client = get_ollama_async_client_instance()
        response = await ollama_client.embed(model=model_name, input=texts)
        embeddings = response["embeddings"]
        return np.array(embeddings, dtype=np.float32)
    except ImportError:
        import httpx

        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{host}/api/embed",
                json={"model": model_name, "input": texts},
                timeout=60.0,
            )
            resp.raise_for_status()
            return np.array(resp.json()["embeddings"], dtype=np.float32)


ollama_config = LLMConfig(
    embedding_func_raw=ollama_embedding,
    embedding_model_name="nomic-embed-text",
    embedding_dim=768,
    embedding_max_token_size=8192,
    embedding_batch_num=1,
    embedding_func_max_async=1,
    query_better_than_threshold=0.2,
    best_model_func_raw=ollama_complete,
    best_model_name="gemma2:latest",
    best_model_max_token_size=32768,
    best_model_max_async=1,
    cheap_model_func_raw=ollama_mini_complete,
    cheap_model_name="olmo2",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=1,
)


###### DeepSeek Configuration (Chat, Tool Calling, Embeddings & Vision Sub-Tool)
def _normalize_chat_completions_url(base_url: Optional[str] = None) -> str:
    raw = (
        base_url
        or os.environ.get("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com/v1"
    ).rstrip("/")
    if raw.endswith("/chat/completions"):
        return raw
    if not raw.endswith("/v1"):
        raw = f"{raw}/v1"
    return f"{raw}/chat/completions"


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def deepseek_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)
    api_key = kwargs.pop("api_key", None) or os.environ.get("DEEPSEEK_API_KEY", "sk-*******")
    base_url = kwargs.pop("base_url", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    client = get_deepseek_async_client_instance(api_key=api_key, base_url=base_url)
    is_flash_model = "flash" in str(model or "").lower()
    if client is not None:
        req_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if is_flash_model:
            try:
                response = await client.chat.completions.create(
                    **req_kwargs,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except Exception:
                response = await client.chat.completions.create(**req_kwargs)
        else:
            response = await client.chat.completions.create(**req_kwargs)
        content = response.choices[0].message.content or ""
    else:
        import httpx

        url = _normalize_chat_completions_url(base_url)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if is_flash_model:
            payload["thinking"] = {"type": "disabled"}

        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                url,
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            if is_flash_model and response.status_code in (400, 422):
                payload.pop("thinking", None)
                response = await http_client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"].get("content") or ""

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert({args_hash: {"return": content, "model": model}})
        await hashing_kv.index_done_callback()

    return content


async def deepseek_complete(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await deepseek_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_chat_with_tools(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    tool_choice: Optional[str] = None,
) -> dict:
    """Execute a chat completion call with optional OpenAI-compatible tool definitions.

    Returns a normalized assistant message dict:
    {
        "role": "assistant",
        "content": str,
        "tool_calls": [
            {
                "id": str,
                "type": "function",
                "function": {"name": str, "arguments": str}
            },
            ...
        ]
    }
    """
    resolved_model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
    is_flash_model = "flash" in str(resolved_model).lower()
    thinking_env = os.environ.get("DEEPSEEK_CHAT_THINKING", "enabled").strip().lower()
    thinking_type = "disabled" if thinking_env in ("0", "false", "off", "disabled") else "enabled"
    effective_max_tokens = max(max_tokens, 4096) if (is_flash_model and thinking_type == "enabled") else max_tokens
    resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or "sk-*******"
    client = get_deepseek_async_client_instance(api_key=resolved_key, base_url=base_url)

    if client is not None:
        req_kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }
        if tools:
            req_kwargs["tools"] = tools
            if tool_choice:
                req_kwargs["tool_choice"] = tool_choice
        if is_flash_model:
            try:
                response = await client.chat.completions.create(
                    **req_kwargs,
                    extra_body={"thinking": {"type": thinking_type}},
                )
            except Exception:
                response = await client.chat.completions.create(**req_kwargs)
        else:
            response = await client.chat.completions.create(**req_kwargs)
        msg = response.choices[0].message
        norm_tool_calls = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                norm_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": getattr(tc, "type", "function") or "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                )
        out: Dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }
        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content and norm_tool_calls:
            out["reasoning_content"] = reasoning_content
        if norm_tool_calls:
            out["tool_calls"] = norm_tool_calls
        return out

    import httpx

    url = _normalize_chat_completions_url(base_url)
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": effective_max_tokens,
    }
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
    if is_flash_model:
        payload["thinking"] = {"type": thinking_type}

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            url,
            headers=headers,
            json=payload,
            timeout=90.0,
        )
        if is_flash_model and response.status_code in (400, 422):
            payload.pop("thinking", None)
            response = await http_client.post(
                url,
                headers=headers,
                json=payload,
                timeout=90.0,
            )
        response.raise_for_status()
        result = response.json()
        msg_dict = result["choices"][0]["message"]
        out = {
            "role": "assistant",
            "content": msg_dict.get("content") or "",
        }
        if msg_dict.get("reasoning_content") and msg_dict.get("tool_calls"):
            out["reasoning_content"] = msg_dict["reasoning_content"]
        if msg_dict.get("tool_calls"):
            out["tool_calls"] = msg_dict["tool_calls"]
        return out


def _deterministic_ngram_hash_embedding(texts: list[str], dim: int = 768) -> np.ndarray:
    """Fast, dependency-free local CPU fallback embedding preserving lexical/token cosine similarity."""
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())
        if not tokens:
            matrix[row_idx, 0] = 1.0
            continue
        features = list(tokens)
        features.extend(f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
        for feat in features:
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            matrix[row_idx, bucket] += sign
        norm = np.linalg.norm(matrix[row_idx])
        if norm > 0:
            matrix[row_idx] /= norm
    return matrix


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def bge_m3_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    import httpx

    api_url = os.environ.get("EMBEDDING_API_URL", "https://api.siliconflow.cn/v1/embeddings")
    api_key = (
        os.environ.get("SILICONFLOW_API_KEY")
        or os.environ.get("EMBEDDING_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "sk-******"
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name or "BAAI/bge-m3",
                "input": texts,
                "encoding_format": "float",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        return np.array(embeddings, dtype=np.float32)


async def local_or_api_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    """Flexible local/API text embedding supporting:
    - 'sentence_transformers' / 'st': Local CPU SentenceTransformer (default 'BAAI/bge-base-en-v1.5', 768-dim)
    - 'ollama': Local Ollama daemon ('nomic-embed-text', 768-dim)
    - 'siliconflow' / 'api' / 'openai': OpenAI-compatible embedding API
    - 'hash': Deterministic n-gram hash fallback (768-dim)
    """
    provider = os.environ.get("MOVIERAG_EMBEDDING_PROVIDER", "auto").lower()
    target_dim = int(os.environ.get("MOVIERAG_EMBEDDING_DIM", "768"))

    if provider == "hash":
        return _deterministic_ngram_hash_embedding(texts, dim=target_dim)

    if provider in ("sentence_transformers", "st", "bge-base"):
        embedder = _get_sentence_transformer_embedder(model_name)
        vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    if provider in ("siliconflow", "api") or (
        provider == "auto" and os.environ.get("SILICONFLOW_API_KEY")
    ):
        return await bge_m3_embedding(model_name, texts)

    if provider == "openai" or (
        provider == "auto"
        and os.environ.get("EMBEDDING_API_KEY")
        and not os.environ.get("OLLAMA_HOST")
    ):
        return await bge_m3_embedding(model_name, texts)

    if provider == "ollama" or (provider == "auto" and os.environ.get("OLLAMA_HOST")):
        try:
            return await ollama_embedding(model_name, texts)
        except Exception as exc:
            if provider == "ollama":
                raise
            logger.warning(f"Ollama embedding ({model_name}) unreachable ({exc}); trying local SentenceTransformer.")

    # Auto mode: try local SentenceTransformer (768-dim BAAI/bge-base-en-v1.5), then Ollama, then hash fallback
    try:
        embedder = _get_sentence_transformer_embedder(model_name)
        vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.shape[1] == target_dim:
            return arr
    except Exception:
        pass

    try:
        return await ollama_embedding(model_name, texts)
    except Exception as exc:
        logger.warning(
            f"Local Ollama/SentenceTransformer unavailable ({exc}); falling back to deterministic CPU embedding."
        )
        return _deterministic_ngram_hash_embedding(texts, dim=target_dim)


def _encode_keyframe_data_uri(image_path: str, max_width: int = 960, jpeg_quality: int = 85) -> str:
    """Downscale a keyframe to max_width and encode as a base64 JPEG data URL."""
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            if max_width and rgb.width > max_width:
                new_h = max(1, int(round(rgb.height * (max_width / float(rgb.width)))))
                rgb = rgb.resize((max_width, new_h), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_data}"
    except ImportError:
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"


def _build_vision_subtool_prompt(
    visual_question: str,
    scene_metadata: dict,
    character_glossary: Optional[dict[str, str]] = None,
) -> str:
    scene_id = scene_metadata.get("scene_id", "unknown_scene")
    start_t = scene_metadata.get("start_time", 0.0)
    end_t = scene_metadata.get("end_time", 0.0)
    chars = scene_metadata.get("characters", [])

    # Extract color legend from keyframes if present
    color_pairs = []
    seen_pairs = set()
    for kf in scene_metadata.get("keyframes", []):
        for cv in kf.get("characters_visible", []):
            if isinstance(cv, dict):
                cname = cv.get("name")
                ccolor = cv.get("color")
                if cname and ccolor and (cname, ccolor) not in seen_pairs:
                    seen_pairs.add((cname, ccolor))
                    color_pairs.append(f"{cname} ({ccolor} box)")
            elif isinstance(cv, str) and cv not in seen_pairs:
                seen_pairs.add(cv)
                color_pairs.append(cv)

    legend_str = ", ".join(color_pairs) if color_pairs else (", ".join(chars) if chars else "None")

    glossary_lines = []
    if character_glossary:
        active_names = set(chars)
        for cp in color_pairs:
            cname = cp.split("(")[0].strip()
            if cname:
                active_names.add(cname)
        for cname, gdesc in character_glossary.items():
            if not active_names or cname in active_names:
                glossary_lines.append(f"- {cname}: {gdesc}")
    glossary_block = (
        "[Character Reference Glossary]\n" + "\n".join(glossary_lines) + "\n\n"
        if glossary_lines
        else ""
    )

    return (
        f"[Scene Context]\n"
        f"Scene ID: {scene_id} | Time Range: {start_t:.2f}s - {end_t:.2f}s\n"
        f"Characters tracked in scene: {', '.join(chars) if chars else 'None'}\n"
        f"Synthetic Tracking Box Overlay Key: {legend_str}\n"
        f"CRITICAL VISUAL RULE: Any colored rectangles drawn on the frames are synthetic computer-vision "
        f"tracking overlays. NEVER mention bounding boxes, rectangles, or box colors in your answer, and "
        f"NEVER confuse a tracking box color with hair, clothing, or object colors.\n\n"
        f"{glossary_block}"
        f"[Visual Question]\n"
        f"{visual_question}\n\n"
        f"[Instructions]\n"
        f"Inspect the attached shot keyframes carefully and answer the visual question with concrete, "
        f"frame-accurate visual observations (postures, objects, expressions, clothing, spatial layout). "
        f"If a requested detail is not visible in the keyframes, state that clearly without guessing."
    )


async def vision_subtool_complete(
    visual_question: str,
    scene_metadata: dict,
    keyframe_paths: list[str],
    character_glossary: Optional[dict[str, str]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_image_width: int = 960,
    max_tokens: int = 450,
) -> str:
    """Option A Vision Sub-Tool (`inspect_keyframes`): queries Vision API on up to 4 annotated shot keyframes."""
    valid_paths = [p for p in (keyframe_paths or []) if p and os.path.isfile(p)][:4]
    if not valid_paths:
        return (
            f"No valid keyframe image files found on disk for scene "
            f"{scene_metadata.get('scene_id', 'unknown')}. Fallback to scene VLM caption: "
            f"{scene_metadata.get('vlm_caption', 'No caption available.')}"
        )

    vision_model = (
        model
        or os.environ.get("VISION_MODEL_ID")
        or os.environ.get("DEEPSEEK_VISION_MODEL")
        or "deepseek-flash"
    )
    vision_key = (
        api_key
        or os.environ.get("VISION_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "sk-*******"
    )
    vision_base_url = (
        base_url
        or os.environ.get("VISION_API_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com"
    )

    prompt_text = _build_vision_subtool_prompt(
        visual_question=visual_question,
        scene_metadata=scene_metadata,
        character_glossary=character_glossary,
    )

    content_items: List[Dict[str, Any]] = []
    for p in valid_paths:
        content_items.append(
            {
                "type": "image_url",
                "image_url": {"url": _encode_keyframe_data_uri(p, max_width=max_image_width)},
            }
        )
    content_items.append({"type": "text", "text": prompt_text})

    messages = [
        {
            "role": "system",
            "content": (
                "You are a frame-accurate visual inspection assistant for an animated movie RAG system. "
                "Describe only what is directly visible in the attached keyframes. Never mention synthetic "
                "tracking boxes or box colors."
            ),
        },
        {"role": "user", "content": content_items},
    ]

    client = get_deepseek_async_client_instance(api_key=vision_key, base_url=vision_base_url)
    is_deepseek_vlm = "deepseek" in vision_model.lower()

    if client is not None:
        try:
            extra_kwargs = (
                {"extra_body": {"thinking": {"type": "disabled"}}} if is_deepseek_vlm else {}
            )
            response = await client.chat.completions.create(
                model=vision_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                **extra_kwargs,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            response = await client.chat.completions.create(
                model=vision_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return (response.choices[0].message.content or "").strip()

    import httpx

    url = _normalize_chat_completions_url(vision_base_url)
    payload: Dict[str, Any] = {
        "model": vision_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    if is_deepseek_vlm:
        payload["thinking"] = {"type": "disabled"}

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            url,
            headers={
                "Authorization": f"Bearer {vision_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        return (result["choices"][0]["message"].get("content") or "").strip()


# DeepSeek + Local/API Embeddings (default 768-dim nomic-embed-text, deepseek-flash)
deepseek_config = LLMConfig(
    embedding_func_raw=local_or_api_embedding,
    embedding_model_name=os.environ.get("MOVIERAG_EMBEDDING_MODEL", "nomic-embed-text"),
    embedding_dim=int(os.environ.get("MOVIERAG_EMBEDDING_DIM", "768")),
    embedding_max_token_size=8192,
    embedding_batch_num=32,
    embedding_func_max_async=16,
    query_better_than_threshold=0.2,
    best_model_func_raw=deepseek_complete,
    best_model_name=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
    best_model_max_token_size=32768,
    best_model_max_async=16,
    cheap_model_func_raw=deepseek_complete,
    cheap_model_name=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
    cheap_model_max_token_size=32768,
    cheap_model_max_async=16,
)

# DeepSeek + SiliconFlow BAAI/bge-m3 (1024-dim, deepseek-flash)
deepseek_bge_config = LLMConfig(
    embedding_func_raw=bge_m3_embedding,
    embedding_model_name="BAAI/bge-m3",
    embedding_dim=1024,
    embedding_max_token_size=8192,
    embedding_batch_num=32,
    embedding_func_max_async=16,
    query_better_than_threshold=0.2,
    best_model_func_raw=deepseek_complete,
    best_model_name=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
    best_model_max_token_size=32768,
    best_model_max_async=16,
    cheap_model_func_raw=deepseek_complete,
    cheap_model_name=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
    cheap_model_max_token_size=32768,
    cheap_model_max_async=16,
)
