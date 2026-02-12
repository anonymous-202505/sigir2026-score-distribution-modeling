from functools import partial
from typing import Tuple

from .AbstractModelWrapper import AbstractModelWrapper
from .GenericModelWrapper import GenericModelWrapper

MODEL_WRAPPERS = {
    "e5": (
        partial(
            GenericModelWrapper,
            pretrained_model_name="intfloat/multilingual-e5-large-instruct",
            query_instruct="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        ),
        1024,
    ),
    "snowflakev2": (
        partial(
            GenericModelWrapper,
            pretrained_model_name="Snowflake/snowflake-arctic-embed-m-v2.0",
            query_instruct="query: ",
        ),
        768,
    ),
    "snowflake": (
        partial(
            GenericModelWrapper,
            pretrained_model_name="Snowflake/snowflake-arctic-embed-m",
            query_instruct="Represent this sentence for searching ",
        ),
        768,
    ),
}


def get_model_wrapper(model_name: str) -> Tuple[AbstractModelWrapper, int]:
    """
    Attempts to return the model wrapper used to create embeddings to be evaluated.

    Args:
        model_name: The model to use.

    Returns:
        The model wrapper and its maximum embedding dimension, if implemented.
    """
    if model_name in MODEL_WRAPPERS:
        wrapper_class, max_dim = MODEL_WRAPPERS[model_name]
        return wrapper_class(), max_dim
    else:
        raise NotImplementedError(f"Model {model_name} not supported!")
