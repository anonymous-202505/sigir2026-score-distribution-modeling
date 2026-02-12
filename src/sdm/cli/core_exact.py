import heapq
import json
import os
import random

import click
from datasets import load_dataset
from tqdm import tqdm

from sdm.config import *
from sdm.model_wrappers import get_model_wrapper
from sdm.utils.encoding import load_embeddings
from sdm.utils.similarity import cos_sim

# Set random seed for reproducibility
random.seed(42)

splits = {
    "passage": [
        "pass_core",
        "pass_10k",
        "pass_100k",
        "pass_1M",
        "pass_10M",
        "pass_100M",
    ],
    "document": [
        "doc_core",
        "doc_10k",
        "doc_100k",
        "doc_1M",
        "doc_10M",
    ],
}


@click.command()
@click.argument("model_name", type=str, default="snowflakev2")
@click.argument("dataset", type=str, default="passage")
def core_exact(model_name: str, dataset: str):
    """
    Evaluates the full ranking results for a given model and dataset.
    """
    click.echo(f"Model: {model_name}")
    click.echo(f"Dataset: {dataset}")
    click.echo()

    total_corpus_size = 100_000_000 if dataset == "passage" else 10_000_000
    click.echo(f"Total corpus size: {total_corpus_size:,}")
    click.echo()

    # Instaniate results heap
    results = {}

    # Load qrels
    qrels = set()
    qrels_file = os.path.join(DATASETS_FOLDER, dataset, "qrels.jsonl")
    with open(qrels_file, "r") as f:
        for line in f:
            item = json.loads(line)
            qid = item["query-id"]
            qrels.add(qid)

    # Load queries
    query_ids = []
    queries = []
    queries_file = os.path.join(DATASETS_FOLDER, "queries.jsonl")
    with open(queries_file, "r") as f:
        for line in f:
            item = json.loads(line)
            qid = item["_id"]
            if qid not in qrels:
                continue
            query_ids.append(item["_id"])
            queries.append(item["text"])

    # Load relevant and distractor documents
    core_docids = set()
    core_corpus_file = os.path.join(
        DATASETS_FOLDER, dataset, "relevant_distractor.jsonl"
    )
    with open(core_corpus_file, "r") as f:
        for line in f:
            item = json.loads(line)
            docid = item["_id"]
            core_docids.add(docid)
    click.echo(
        f"Collected {len(core_docids):,} relevant and distractor documents' texts from file {core_corpus_file}"
    )

    # Load model
    model, _ = get_model_wrapper(model_name)

    # Encode queries
    save_path = os.path.join(
        EMBEDDINGS_FOLDER, model_name, dataset, "query_embeddings.pt"
    )
    query_embeddings = load_embeddings(
        save_path,
        model,
        queries,
    )

    def yield_non_relevant_texts():
        # Load non-relevant documents
        _texts = []
        for split in splits[dataset]:
            hf_dataset = load_dataset("PaDaS-Lab/CoRE", "corpus")[split]
            for doc in hf_dataset:
                docid = doc["_id"]  # type: ignore
                if docid in core_docids:
                    continue
                _texts.append(
                    {
                        "title": doc["title"],  # type: ignore
                        "text": doc["text"],  # type: ignore
                    }
                )

                if len(_texts) >= 50_000:
                    yield _texts
                    _texts = []

        if len(_texts) > 0:
            yield _texts

    # Process non-relevant documents in batches
    for _texts in tqdm(
        yield_non_relevant_texts(), desc="Processing non-relevant documents"
    ):

        # Encode non-relevant texts
        non_relevant_embeddings = load_embeddings(
            None,
            model,
            _texts,
        )
        _texts = []

        # Compute cosine similarity scores for non-relevant documents
        for i, query_embedding in enumerate(query_embeddings):
            qid = query_ids[i]
            if qid not in results:
                results[qid] = []

            scores = (
                cos_sim(
                    query_embedding.unsqueeze(0),
                    non_relevant_embeddings,
                )
                .squeeze(0)
                .tolist()
            )
            for score in scores:
                if len(results[qid]) < 1_000:
                    heapq.heappush(results[qid], score)
                else:
                    heapq.heappushpop(results[qid], score)

    # Save updated results table
    results_path = os.path.join(RESULTS_FOLDER, "exact", model_name, f"{dataset}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    click.echo("Done")
