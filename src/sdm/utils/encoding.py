import os
from typing import Any, Dict, List, Union

import click
import torch

from sdm.config import *
from sdm.model_wrappers import AbstractModelWrapper


def load_embeddings(
    save_path: Union[str, None],
    model: AbstractModelWrapper,
    queries_or_corpus: Union[List[str], List[Dict[str, str]]],
    query_instruction: Union[str, None] = None,
    encode_kwargs: Dict[str, Any] = {},
) -> torch.Tensor:
    """
    Attempts to load query or corpus embeddings at the given path.

    Args:
        save_path: The path where the embeddings are stored.
        model: The model used to encode the queries or corpus.
        queries_or_corpus: The queries or corpus to encode.
        encode_kwargs: The keyword arguments passed to the encoding function.
    """
    if save_path:
        try:
            embeddings = torch.load(save_path, weights_only=False)
            # click.echo(f"Loaded embeddings from {save_path}")
            return embeddings
        except OSError:
            click.echo(f"Could not find any embeddings at {save_path}")

    if all(isinstance(q, str) for q in queries_or_corpus):
        embeddings = _encode_queries(
            save_path, queries_or_corpus, model, query_instruction, encode_kwargs  # type: ignore
        )
    else:
        embeddings = _encode_corpus(
            save_path, queries_or_corpus, model, encode_kwargs  # type: ignore
        )
    return embeddings


def _encode_queries(
    save_path: Union[str, None],
    queries: List[str],
    model: AbstractModelWrapper,
    query_instruction: Union[str, None] = None,
    encode_kwargs: Dict[str, Any] = {},
) -> torch.Tensor:
    """
    Encodes the given queries and stores them as pytorch tensors.

    Args:
        save_path: The path where the embeddings are stored.
        queries: The queries to encode.
        model: The model used to encode the queries.
        encode_kwargs: The keyword arguments passed to the encoding function.

    Returns:
        The generated query embeddings.
    """
    if query_instruction:
        queries = [f"{query_instruction}\n{q}" for q in queries]
    query_embeddings = model.encode_queries(
        queries,
        **encode_kwargs,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(query_embeddings, save_path)

    return query_embeddings


def _encode_corpus(
    save_path: Union[str, None],
    corpus: List[Dict[str, str]],
    model: AbstractModelWrapper,
    encode_kwargs: Dict[str, Any] = {},
) -> torch.Tensor:
    """
    Encodes the given corpus documents and stores them as pytorch tensors.

    Args:
        save_path: The path where the embeddings are stored.
        corpus: The documents to encode.
        model: The model used to encode the queries.
        encode_kwargs: The keyword arguments passed to the encoding function.
    """
    sub_corpus_embeddings = model.encode_corpus(
        corpus,
        **encode_kwargs,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            sub_corpus_embeddings,
            save_path,
        )

    return sub_corpus_embeddings
