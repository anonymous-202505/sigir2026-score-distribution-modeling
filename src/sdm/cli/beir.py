import gzip
import heapq
import json
import os
import random
from typing import Dict

import click
import ir_datasets
import numpy as np
import pandas as pd
import torch
from scipy import stats as st
from tqdm import tqdm

from sdm.config import *
from sdm.model_wrappers import get_model_wrapper
from sdm.utils.encoding import load_embeddings
from sdm.utils.score_distribution import *
from sdm.utils.similarity import cos_sim

# Set random seed for reproducibility
random.seed(42)


K_NDCG = [3, 5, 10, 20, 30, 50, 100]
K_RECALL = [10, 20, 30, 50, 100, 500, 1000]


@click.command()
@click.argument(
    "model_name", type=click.Choice(MODEL_NAMES + ["all"]), default="snowflakev2"
)
@click.argument("dataset", type=click.Choice(BEIR_DATASETS), default="fiqa")
@click.argument("num_pooling", type=int, default=100)
def beir(model_name: str, dataset: str, num_pooling: int):
    """
    Compute cosine similarity scores for relevant, distractor, and non-relevant documents for the
    given model and dataset, and predict NDCG@k and Recall@k metrics based on the computed scores.
    Results are stored in a results table CSV file.
    """
    model_names = MODEL_NAMES if model_name == "all" else [model_name]

    for model_name in model_names:
        _beir(model_name, dataset, num_pooling)

    click.echo("Done")


def _beir(model_name: str, dataset: str, num_pooling: int):
    """
    Function for processing a single model and dataset.
    """
    click.echo(f"Model: {model_name}")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Num Pooling: {num_pooling}")
    click.echo()

    # Read results table
    results_table_path = os.path.join(RESOURCES_FOLDER, "tables", "beir.csv")
    if os.path.exists(results_table_path):
        df_results = pd.read_csv(results_table_path, index_col=0)
    else:
        df_results = pd.DataFrame()

    # Load cosine similarity scores if existing
    cosine_similarity_scores_save_path = os.path.join(
        RESOURCES_FOLDER,
        "cosine_similarity_scores",
        model_name,
        dataset,
        f"{num_pooling}.json",
    )
    if os.path.exists(cosine_similarity_scores_save_path):
        with open(cosine_similarity_scores_save_path, "r") as f:
            cosine_similarity_scores = json.load(f)
        total_corpus_size = cosine_similarity_scores["total_corpus_size"]
        qrels = cosine_similarity_scores["qrels"]
        relevant_cosine_similarity_scores = cosine_similarity_scores["relevant"]
        distractor_cosine_similarity_scores = cosine_similarity_scores["distractor"]
        non_relevant_fitting_params = cosine_similarity_scores["non_relevant"]
        top_non_relevant_cosine_similarity_scores = cosine_similarity_scores[
            "top_non_relevant"
        ]
        click.echo(
            f"Loaded cosine similarity scores from {cosine_similarity_scores_save_path}"
        )
        click.echo()
    else:
        click.echo(
            f"Cosine similarity scores file {cosine_similarity_scores_save_path} not found. Computing cosine similarity scores..."
        )

        # Initialize dataset
        if dataset == "dbpedia":
            ir_dataset_path = f"beir/dbpedia-entity/test"
        elif dataset == "nq":
            ir_dataset_path = f"beir/nq"
        elif dataset == "climate-fever":
            ir_dataset_path = f"beir/climate-fever"
        else:
            ir_dataset_path = f"beir/{dataset}/test"
        ir_dataset = ir_datasets.load(ir_dataset_path)
        total_corpus_size = ir_dataset.docs_count()

        # Load qrels
        qrels = {}
        relevant_doc_ids = set()
        for qrel in ir_dataset.qrels_iter():
            query_id = qrel.query_id
            doc_id = qrel.doc_id
            score = qrel.relevance
            if score > 0:
                qrels[query_id] = qrels.get(query_id, {})
                qrels[query_id][doc_id] = score
                relevant_doc_ids.add(doc_id)

        # Filter queries to only those in qrels
        query_ids = []
        queries = []
        for query in ir_dataset.queries_iter():
            if query.query_id not in qrels:
                continue
            query_id = query.query_id
            query_ids.append(query_id)
            queries.append(query.text)
        click.echo(f"Loaded {len(queries):,} queries")

        # Pooling with jinav3 and BM25 to get distractor documents

        # Read BM25 and Jina full results file
        distractor_doc_ids = set()
        bm25_results_file = os.path.join(RESULTS_FOLDER, "BM25", f"{dataset}.json.gz")
        jina_results_file = os.path.join(RESULTS_FOLDER, "jinav3", f"{dataset}.json.gz")
        with gzip.open(bm25_results_file, "rt") as bm25_file, gzip.open(
            jina_results_file, "rt"
        ) as jina_file:
            bm25_data = json.load(bm25_file)
            jina_data = json.load(jina_file)

            for (bm25_qid, bm25_ranking), (jina_qid, jina_ranking) in zip(
                bm25_data["rankings"].items(), jina_data["rankings"].items()
            ):
                assert bm25_qid == jina_qid
                distractor_doc_ids_for_query: Dict[str, int] = {}
                for docid, rank in bm25_ranking["random"].items():
                    if docid in relevant_doc_ids:
                        continue
                    if (
                        docid not in distractor_doc_ids_for_query
                        or distractor_doc_ids_for_query[docid] > rank
                    ):
                        distractor_doc_ids_for_query[docid] = rank
                for docid, rank in jina_ranking["random"].items():
                    if docid in relevant_doc_ids:
                        continue
                    if (
                        docid not in distractor_doc_ids_for_query
                        or distractor_doc_ids_for_query[docid] > rank
                    ):
                        distractor_doc_ids_for_query[docid] = rank
                _set = {
                    docid
                    for docid, _ in sorted(
                        distractor_doc_ids_for_query.items(), key=lambda item: item[1]
                    )[:num_pooling]
                }
                distractor_doc_ids.update(_set)
        click.echo(
            f"Pooled {len(distractor_doc_ids):,} distractor documents from BM25 and jinav3 rankings"
        )

        # Load non-relevant documents
        core_texts = {}
        non_relevant_texts = []
        for doc in ir_dataset.docs_iter():
            docid = doc.doc_id
            if docid in relevant_doc_ids or docid in distractor_doc_ids:
                core_texts[docid] = {
                    "title": doc.title if hasattr(doc, "title") else "",
                    "text": doc.text,
                }
            else:
                non_relevant_texts.append(
                    {
                        "title": doc.title if hasattr(doc, "title") else "",
                        "text": doc.text,
                    }
                )
        click.echo(
            f"Collected {len(core_texts):,} relevant and distractor documents' texts and {len(non_relevant_texts):,} non-relevant documents' texts"
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

        # Encode core texts
        core_docids = sorted(core_texts.keys())
        core_texts_list = [core_texts[docid] for docid in core_docids]
        save_path = os.path.join(
            EMBEDDINGS_FOLDER,
            model_name,
            dataset,
            str(num_pooling),
            "core_embeddings.pt",
        )
        core_embeddings = load_embeddings(
            save_path,
            model,
            core_texts_list,
        )

        # Encode non-relevant texts
        save_path = os.path.join(
            EMBEDDINGS_FOLDER,
            model_name,
            dataset,
            str(num_pooling),
            "non_relevant_embeddings.pt",
        )
        non_relevant_embeddings = load_embeddings(
            save_path,
            model,
            non_relevant_texts,
        )

        # Evaluate all non-relevant embeddings
        top_non_relevant_cosine_similarity_scores = {}
        for i, query_embedding in enumerate(query_embeddings):
            qid = query_ids[i]
            if qid not in top_non_relevant_cosine_similarity_scores:
                top_non_relevant_cosine_similarity_scores[qid] = []
            scores = (
                cos_sim(
                    query_embedding.unsqueeze(0),
                    non_relevant_embeddings,
                )
                .squeeze(0)
                .tolist()
            )
            for score in scores:
                if len(top_non_relevant_cosine_similarity_scores[qid]) < 1_000:
                    heapq.heappush(
                        top_non_relevant_cosine_similarity_scores[qid], score
                    )
                else:
                    heapq.heappushpop(
                        top_non_relevant_cosine_similarity_scores[qid], score
                    )

        # Compute cosine similarity scores for relevant and distractor documents
        relevant_cosine_similarity_scores = {}
        distractor_cosine_similarity_scores = {}
        for i, query_embedding in tqdm(
            enumerate(query_embeddings),
            desc="Computing cos sim scores (relevant and distractor)",
            total=len(query_embeddings),
        ):
            qid = query_ids[i]
            relevant_cosine_similarity_scores[qid] = []
            distractor_cosine_similarity_scores[qid] = []
            scores = cos_sim(
                query_embedding.unsqueeze(0),
                core_embeddings,
            )
            for docid, score in zip(core_docids, scores.squeeze(0).tolist()):
                if docid in qrels[qid] and qrels[qid][docid] > 0:
                    relevant_cosine_similarity_scores[qid].append(score)
                else:
                    distractor_cosine_similarity_scores[qid].append(score)

        # Sample non-relevant embeddings
        non_relevant_embeddings_sample = torch.tensor(
            random.sample(non_relevant_embeddings.tolist(), 10_000)
        )
        click.echo(
            f"Sampled {len(non_relevant_embeddings_sample):,} non-relevant documents' embeddings for encoding"
        )

        # Compute cosine similarity scores for non-relevant documents
        non_relevant_sample_cosine_similarity_scores = {}
        for i, query_embedding in tqdm(
            enumerate(query_embeddings),
            desc="Computing cos sim scores (non-relevant)",
            total=len(query_embeddings),
        ):
            qid = query_ids[i]
            non_relevant_sample_cosine_similarity_scores[qid] = []
            scores = cos_sim(
                query_embedding.unsqueeze(0),
                non_relevant_embeddings_sample,
            )
            for score in scores.squeeze(0).tolist():
                non_relevant_sample_cosine_similarity_scores[qid].append(score)

        # Fit score distribution
        non_relevant_fitting_params = {}
        for qid, scores in tqdm(
            non_relevant_sample_cosine_similarity_scores.items(),
            desc="Fitting score distributions",
        ):
            non_relevant_fitting_params[qid] = st.skewnorm.fit(scores)

        # Store cosine similarity scores
        os.makedirs(os.path.dirname(cosine_similarity_scores_save_path), exist_ok=True)
        with open(cosine_similarity_scores_save_path, "w") as f:
            json.dump(
                {
                    "total_corpus_size": total_corpus_size,
                    "qrels": qrels,
                    "relevant": relevant_cosine_similarity_scores,
                    "distractor": distractor_cosine_similarity_scores,
                    "non_relevant": non_relevant_fitting_params,
                    "top_non_relevant": top_non_relevant_cosine_similarity_scores,
                },
                f,
                indent=4,
            )
        click.echo(
            f"Stored cosine similarity scores to {cosine_similarity_scores_save_path}"
        )
        click.echo()

    # Turn scores lists into numpy arrays for faster processing
    for qid in relevant_cosine_similarity_scores.keys():
        relevant_cosine_similarity_scores[qid] = np.array(
            relevant_cosine_similarity_scores[qid]
        )
        distractor_cosine_similarity_scores[qid] = np.array(
            distractor_cosine_similarity_scores[qid]
        )
        _temp = np.array(top_non_relevant_cosine_similarity_scores[qid])
        top_non_relevant_cosine_similarity_scores[qid] = np.concatenate(
            [distractor_cosine_similarity_scores[qid], _temp]
        )

    # Predict NDCG@k (Full)
    click.echo("Full Results Metrics:")
    for k in K_NDCG:
        predicted_ndcg_at_k = {}
        for qid in relevant_cosine_similarity_scores.keys():
            N = 0
            tau_k_ndcg = compute_ndcg_at_k_for_q(
                k,
                relevant_cosine_similarity_scores[qid],
                top_non_relevant_cosine_similarity_scores[qid],
                non_relevant_fitting_params[qid],
                N,
            )
            predicted_ndcg_at_k[qid] = tau_k_ndcg
        avg_ndcg_at_k = sum(predicted_ndcg_at_k.values()) / len(predicted_ndcg_at_k)
        click.echo(f"NDCG@{k}: {avg_ndcg_at_k:.4f}")
        df_results.loc[dataset + "_" + model_name + "_Full", f"NDCG@{k}"] = (
            avg_ndcg_at_k
        )

    # Predict Recall@k (Full)
    for k in K_RECALL:
        predicted_recall_at_k = {}
        for qid in relevant_cosine_similarity_scores.keys():
            N = 0
            tau_k_recall = compute_recall_at_k_for_q(
                k,
                relevant_cosine_similarity_scores[qid],
                top_non_relevant_cosine_similarity_scores[qid],
                non_relevant_fitting_params[qid],
                N,
            )
            predicted_recall_at_k[qid] = tau_k_recall
        avg_recall_at_k = sum(predicted_recall_at_k.values()) / len(
            predicted_recall_at_k
        )
        click.echo(f"Recall@{k}: {avg_recall_at_k:.4f}")
        df_results.loc[dataset + "_" + model_name + "_Full", f"Recall@{k}"] = (
            avg_recall_at_k
        )
    click.echo()

    # Predict NDCG@k (Base)
    click.echo("Base Results Metrics:")
    for k in K_NDCG:
        predicted_ndcg_at_k = {}
        for qid in relevant_cosine_similarity_scores.keys():
            N = 0
            tau_k_ndcg = compute_ndcg_at_k_for_q(
                k,
                relevant_cosine_similarity_scores[qid],
                distractor_cosine_similarity_scores[qid],
                non_relevant_fitting_params[qid],
                N,
            )
            predicted_ndcg_at_k[qid] = tau_k_ndcg
        avg_ndcg_at_k = sum(predicted_ndcg_at_k.values()) / len(predicted_ndcg_at_k)
        click.echo(f"NDCG@{k}: {avg_ndcg_at_k:.4f}")
        df_results.loc[dataset + "_" + model_name + "_Pool", f"NDCG@{k}"] = (
            avg_ndcg_at_k
        )

    # Predict Recall@k (Base)
    for k in K_RECALL:
        predicted_recall_at_k = {}
        for qid in relevant_cosine_similarity_scores.keys():
            N = 0
            tau_k_recall = compute_recall_at_k_for_q(
                k,
                relevant_cosine_similarity_scores[qid],
                distractor_cosine_similarity_scores[qid],
                non_relevant_fitting_params[qid],
                N,
            )
            predicted_recall_at_k[qid] = tau_k_recall
        avg_recall_at_k = sum(predicted_recall_at_k.values()) / len(
            predicted_recall_at_k
        )
        click.echo(f"Recall@{k}: {avg_recall_at_k:.4f}")
        df_results.loc[dataset + "_" + model_name + "_Pool", f"Recall@{k}"] = (
            avg_recall_at_k
        )
    click.echo()

    # Predict NDCG@k
    click.echo(f"Predicted NDCG@k Metrics for corpus size {total_corpus_size:,}:")
    for k in K_NDCG:
        predicted_ndcg_at_k = {}
        for qid in relevant_cosine_similarity_scores.keys():
            R = len(relevant_cosine_similarity_scores[qid])
            D = len(distractor_cosine_similarity_scores[qid])
            N = total_corpus_size - R - D
            tau_k_ndcg = compute_ndcg_at_k_for_q(
                k,
                relevant_cosine_similarity_scores[qid],
                distractor_cosine_similarity_scores[qid],
                non_relevant_fitting_params[qid],
                N,
            )
            predicted_ndcg_at_k[qid] = tau_k_ndcg
        avg_ndcg_at_k = sum(predicted_ndcg_at_k.values()) / len(predicted_ndcg_at_k)
        click.echo(f"NDCG@{k}: {avg_ndcg_at_k:.4f}")
        df_results.loc[dataset + "_" + model_name + "_SDM", f"NDCG@{k}"] = avg_ndcg_at_k
    click.echo()

    # Predict Recall@k
    click.echo(f"Predicted Recall@k Metrics for corpus size {total_corpus_size:,}:")
    for k in K_RECALL:
        predicted_recall_at_k = {}
        for qid in relevant_cosine_similarity_scores.keys():
            R = len(relevant_cosine_similarity_scores[qid])
            D = len(distractor_cosine_similarity_scores[qid])
            N = total_corpus_size - R - D
            tau_k_recall = compute_recall_at_k_for_q(
                k,
                relevant_cosine_similarity_scores[qid],
                distractor_cosine_similarity_scores[qid],
                non_relevant_fitting_params[qid],
                N,
            )
            predicted_recall_at_k[qid] = tau_k_recall
        avg_recall_at_k = sum(predicted_recall_at_k.values()) / len(
            predicted_recall_at_k
        )
        click.echo(f"Recall@{k}: {avg_recall_at_k:.4f}")
        df_results.loc[dataset + "_" + model_name + "_SDM", f"Recall@{k}"] = (
            avg_recall_at_k
        )
    click.echo()

    # Save updated results table
    os.makedirs(os.path.dirname(results_table_path), exist_ok=True)
    df_results.to_csv(results_table_path, index=True)
    click.echo(f"Updated results table saved to {results_table_path}")
