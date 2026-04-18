"""
LLM Module
Sets up a HuggingFace language model for text generation.
Uses google/flan-t5-small by default (free, runs locally, no API key needed).

Fix: Loads the model directly via AutoTokenizer + AutoModelForSeq2SeqLM
to avoid "Unknown task text2text-generation" errors in older transformers versions.
"""

from typing import Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from langchain_core.language_models.llms import LLM
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
except ImportError:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun

# Default model - small, fast, and free (runs locally)
DEFAULT_MODEL = "google/flan-t5-small"


class FlanT5LLM(LLM):
    """
    A LangChain-compatible LLM wrapper for Flan-T5 seq2seq models.
    Loads the model directly without relying on the transformers pipeline
    task name, fixing compatibility issues with older transformers versions.
    """

    model_name: str = DEFAULT_MODEL
    max_new_tokens: int = 256
    tokenizer: Any = None
    model: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model_name: str = DEFAULT_MODEL, max_new_tokens: int = 256, **kwargs):
        super().__init__(model_name=model_name, max_new_tokens=max_new_tokens, **kwargs)
        print("  -> Loading tokenizer for '{}'...".format(model_name))
        object.__setattr__(self, "tokenizer", AutoTokenizer.from_pretrained(model_name))
        print("  -> Loading model for '{}'...".format(model_name))
        object.__setattr__(
            self,
            "model",
            AutoModelForSeq2SeqLM.from_pretrained(model_name)
        )
        print("  -> LLM loaded successfully")

    @property
    def _llm_type(self) -> str:
        return "flan-t5-direct"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=2,
                early_stopping=True
            )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()


def get_llm(model_name: str = DEFAULT_MODEL, max_new_tokens: int = 256) -> FlanT5LLM:
    """
    Create and return a HuggingFace LLM instance for text generation.

    Args:
        model_name (str): The HuggingFace model name (Flan-T5 family).
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        FlanT5LLM: A LangChain-compatible LLM instance.

    Raises:
        RuntimeError: If the model fails to load.
    """
    try:
        return FlanT5LLM(model_name=model_name, max_new_tokens=max_new_tokens)
    except Exception as e:
        raise RuntimeError("Failed to load LLM '{}': {}".format(model_name, str(e)))

