from typing import Any, Sequence, Mapping, Union
from functools import partial
import json
from frozendict import frozendict
from filelock import FileLock

import asyncio
import langchain_core.messages.base
import numpy as np
import pickle
import os
import os.path
import hashlib

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks import get_openai_callback
import inverse_cai.local_secrets  # required to load env vars
from loguru import logger

NULL_LOGPROB_VALUE = -1000000


# Custom OpenRouter headers
# https://openrouter.ai/docs/api-reference/overview#headers
OPENROUTER_HEADERS = {
    "X-Title": "ICAI",
    # We need to set HTTP-Referer in addition to X-Title since otherwise openrouter does
    # not show an App name in the activity overview. If we set both, it shows the X-Title
    # and links to HTTP-Referer.
    "HTTP-Referer": "https://github.com/rdnfn/icai",
}


def serializable(obj):
    if isinstance(obj, Union[str, int, float]):
        return obj
    elif isinstance(obj, Sequence):
        return tuple(serializable(x) for x in obj)
    elif isinstance(obj, Mapping):
        return frozendict({k: serializable(v) for k, v in obj.items()})
    elif isinstance(obj, langchain_core.messages.base.BaseMessage):
        return serializable(langchain_core.messages.base.message_to_dict(obj))
    else:
        logger.warning(f"{obj} ({type(obj).__name__}) is not serializable, converting to string for cache key")
        return str(obj)


def hash_obj(obj):
    m = hashlib.sha256()
    m.update(json.dumps(serializable(obj)).encode())
    return m.hexdigest()


def partial_hash(obj):
    """
    This function is for humans, to make it easier to inspect the differences
    between objects
    """
    if isinstance(obj, str):
        if len(obj) < 20:
            return obj
        else:
            return f"hash:{hash_obj(obj)[:8]}"
    elif isinstance(obj, Sequence):
        return tuple(partial_hash(x) for x in obj)
    elif isinstance(obj, Mapping):
        return frozendict({k: partial_hash(v) for k, v in obj.items()})
    else:
        return f"hash:{hash_obj(obj)[:8]}"


class CachedModel:

    def __init__(self, cache_dir, seed=0):
        self.cache_dir = cache_dir
        self.cache = None
        self.cached_sync_funcs = tuple()
        self.cached_async_funcs = tuple()
        self.seed = seed

    @property
    def cache_path(self):
        return f"{self.cache_dir}/{self.seed}.pkl"

    def load_cache(self):
        try:
            with open(self.cache_path, "rb") as cache_file:
                file_cache = pickle.load(cache_file)
        except (FileNotFoundError, EOFError):
            file_cache = {}

        if self.cache is None:
            self.cache = {}

        self.cache = file_cache | self.cache

    def save_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)

        with FileLock(self.cache_path + ".lock"):
            self.load_cache()
            with open(self.cache_path, "wb") as cache_file:
                pickle.dump(self.cache, cache_file)

    @staticmethod
    async def _arun(coroutine):
        await coroutine
        return coroutine

    async def async_cache_run(self, func: str, *args, **kwargs):
        if self.cache is None:
            self.load_cache()

        h = hash_obj((func, args, kwargs))
        result = self.cache.get(h)

        if result is None:
            logger.debug(f"cache {self.seed} miss ({h[:8]})")
            result = self.model.__getattr__(func)(*args, **kwargs)

            if asyncio.iscoroutine(result):
                result = await result

            self.cache[h] = result
            self.save_cache()

        else:
            logger.debug(f"cache {self.seed} hit ({h[:8]})")

        return result

    def cache_run(self, *args, **kwargs):
        return asyncio.run(self.async_cache_run(*args, **kwargs))

    def __getattr__(self, attr):
        if attr in self.cached_async_funcs:
            return partial(self.async_cache_run, attr)
        elif attr in self.cached_sync_funcs:
            return partial(self.cache_run, attr)
        else:
            return self.model.__getattr__(attr)


class CachedLLM(CachedModel):
    def __init__(
        self,
        name: str,
        temp: float = 0.0,
        enable_logprobs: bool = False,
        max_tokens: int = 1000,
        **kwargs,
    ) -> Any:

        """Get a language model instance.

        Args:
            name: Model name with provider prefix (e.g. "openai/gpt-4o-2024-05-13")
            temp: Temperature for generation (default: 0.0)
            enable_logprobs: Whether to enable logprobs for token probabilities (default: False)
            max_tokens: Maximum tokens to generate (default: 1000)

        Returns:
            CachedModel-wrapped LogWrapper-wrapped language model instance
        """
        super().__init__(cache_dir=f"exp/cache/models/{name}_{temp}_{enable_logprobs}_{max_tokens}", **kwargs)
        self.cached_sync_funcs = ("invoke",)
        self.cached_async_funcs = ("ainvoke",)

        if enable_logprobs:
            model_kwargs = {"logprobs": True, "top_logprobs": 10}
        else:
            model_kwargs = {}

        if name.startswith("openai"):
            self.model = ChatOpenAI(
                model=name.split("/")[1],
                max_tokens=max_tokens,
                temperature=temp,
                model_kwargs=model_kwargs,
            )

        if name.startswith("anthropic"):
            self.model = ChatAnthropic(
                model=name.split("/")[1],
                max_tokens=max_tokens,
                temperature=temp,
                model_kwargs=model_kwargs,
            )

        if name.startswith("openrouter"):

            # Extract the actual model from openrouter/provider/model format
            parts = name.split("/", 2)
            if len(parts) < 3:
                raise ValueError(
                    "OpenRouter model format should be 'openrouter/provider/model'"
                )

            # Use the provider/model as the model name for OpenRouter
            model_id = "/".join(parts[1:])

            # Get the OpenRouter API key from environment variable
            openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable must be set for OpenRouter models"
                )

            self.model = ChatOpenAI(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temp,
                openai_api_key=openrouter_api_key,  # Use the OpenRouter API key
                openai_api_base="https://openrouter.ai/api/v1",
                default_headers=OPENROUTER_HEADERS,
                model_kwargs=model_kwargs,
            )


class CachedEmbeddingsModel(CachedModel):
    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        # TODO: support other embeddings?

        super().__init__(cache_dir=f"exp/cache/models/{name}", **kwargs)

        openrouter = name.startswith("openrouter")
        openai_api_key = os.environ.get("OPENAI_API_KEY")

        if openrouter and not openai_api_key:
            raise ValueError(
                "OpenRouter doesn't support embedding models, OPENAI_API_KEY still required"
            )

        self.model = OpenAIEmbeddings()


# compatibility
def get_model(*args, **kwargs):
    return CachedLLM(*args, **kwargs)

def get_embeddings_model(*args, **kwargs):
    return CachedEmbeddingsModel(*args, **kwargs)


def get_token_probs(tokens: list[str], model: str, messages: list) -> dict[float]:
    """
    Get the token probabilities for a list of tokens.
    """

    def get_first_generation(llm_generate_return_val):
        return llm_generate_return_val.generations[0][0]

    def get_log_prob_for_token(generation, token):
        top_logprobs = generation.generation_info["logprobs"]["content"][0][
            "top_logprobs"
        ]
        for logprob in top_logprobs:
            if logprob["token"] == token:
                return logprob["logprob"]
        logger.warning(
            f"Token {token} not found in top logprobs. Returning {NULL_LOGPROB_VALUE} logprob (close to 0 probability for token)."
        )
        return NULL_LOGPROB_VALUE

    def get_norm_probs_for_tokens(generation, tokens):
        logprobs = [get_log_prob_for_token(generation, token) for token in tokens]
        errors = []

        if any(logprob == NULL_LOGPROB_VALUE for logprob in logprobs):
            for token, logprob in zip(tokens, logprobs):
                if logprob == NULL_LOGPROB_VALUE:
                    errors.append(f"token_not_found_in_top_logprobs_{token}")

        if all(logprob == NULL_LOGPROB_VALUE for logprob in logprobs):
            logger.warning(
                f"All tokens not found in top logprobs. Returning equal probabilities for all tokens ({tokens})."
            )
            normalised_probs = [0.5 for _ in tokens]
            errors.append("neither_tokens_found_in_top_logprobs")
        else:
            probs = [np.exp(logprob) for logprob in logprobs]
            normalised_probs = [prob / sum(probs) for prob in probs]
        prob_dict = dict(zip(tokens, normalised_probs))
        return prob_dict, errors

    return_val = model.generate([messages])
    generation = get_first_generation(return_val)

    token_probs, errors = get_norm_probs_for_tokens(
        generation=generation, tokens=tokens
    )

    return token_probs, errors
