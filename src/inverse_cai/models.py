from typing import Any, Hashable, Sequence, Mapping
from functools import partial
import json
from frozendict import frozendict

import asyncio
import langchain_core.messages.base
import numpy as np
import pickle
import os
import os.path

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


def hashable(obj):
    if isinstance(obj, str):
        # Strings are sequences, and all their subelements are also sequences, on and on forever
        return obj
    if isinstance(obj, Sequence):
        return tuple(hashable(x) for x in obj)
    elif isinstance(obj, Mapping):
        return frozendict({k: hashable(v) for k, v in obj.items()})
    elif isinstance(obj, langchain_core.messages.base.BaseMessage):
        return hashable(langchain_core.messages.base.message_to_dict(obj))
    elif isinstance(obj, Hashable):
        return obj
    else:
        logger.warning(f"{obj} ({type(obj).__name__}) is not hashable, converting to string for cache key")
        return str(obj)


class CachedModel:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.cache = None
        self.cached_sync_funcs = tuple()
        self.cached_async_funcs = tuple()

    def load_cache(self):
        try:
            with open(self.cache_path, "rb") as cache_file:
                self.cache = pickle.load(cache_file)
        except (FileNotFoundError, EOFError):
            pass

        if self.cache is None:
            self.cache = {}

    def save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as cache_file:
            pickle.dump(self.cache, cache_file)

    @staticmethod
    async def _arun(coroutine):
        await coroutine
        return coroutine

    async def acache_run(self, func: str, *args, **kwargs):
        if self.cache is None:
            self.load_cache()

        result = self.cache.get(hashable((func, args, kwargs)))

        if result is None:
            logger.debug(f"returning without cache: {(func, args, kwargs)}")
            result = self.model.__getattr__(func)(*args, **kwargs)

            if asyncio.iscoroutine(result):
                result = await result

            self.cache[hashable((func, args, kwargs))] = result
            self.save_cache()

        return result

    def cache_run(self, *args, **kwargs):
        return asyncio.run(self.acache_run(*args, **kwargs))

    def __getattr__(self, attr):
        if attr in self.cached_async_funcs:
            return partial(self.acache_run, attr)
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
        seed: int = 0,
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
        super().__init__(cache_path=f"exp/cache/models/{name}_{temp}_{enable_logprobs}_{max_tokens}_{seed}")
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
        seed: int = 0,
    ):
        # TODO: support other embeddings?

        super().__init__(cache_path=f"exp/cache/models/{name}_{seed}")

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
