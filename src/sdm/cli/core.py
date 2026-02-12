import json
import os
import random

import click
import numpy as np
import pandas as pd
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
@click.argument("dataset", type=click.Choice(CORE_DATASETS), default="passage")
def core(model_name: str, dataset: str):
    """
    Compute cosine similarity scores for relevant, distractor, and non-relevant documents for the
    given model and dataset, and predict NDCG@k and Recall@k metrics based on the computed scores.
    Results are stored in a results table CSV file.
    """
    model_names = MODEL_NAMES if model_name == "all" else [model_name]

    for model_name in model_names:
        _core(model_name, dataset)

    click.echo("Done")


def _core(model_name: str, dataset: str):
    """
    Function for processing a single model and dataset.
    """
    click.echo(f"Model: {model_name}")
    click.echo(f"Dataset: {dataset}")
    click.echo()

    total_corpus_size = 100_000_000 if dataset == "passage" else 10_000_000
    click.echo(f"Total corpus size: {total_corpus_size:,}")
    click.echo()

    # Read results table
    results_table_path = os.path.join(RESOURCES_FOLDER, "tables", "core.csv")
    if os.path.exists(results_table_path):
        df_results = pd.read_csv(results_table_path, index_col=0)
    else:
        df_results = pd.DataFrame()

    # Load cosine similarity scores if existing
    cosine_similarity_scores_save_path = os.path.join(
        RESOURCES_FOLDER, "cosine_similarity_scores", model_name, f"{dataset}.json"
    )
    if os.path.exists(cosine_similarity_scores_save_path):
        # if os.path.exists(cosine_similarity_scores_save_path) and False:
        with open(cosine_similarity_scores_save_path, "r") as f:
            cosine_similarity_scores = json.load(f)
        relevant_cosine_similarity_scores = cosine_similarity_scores["relevant"]
        distractor_cosine_similarity_scores = cosine_similarity_scores["distractor"]
        non_relevant_fitting_params = cosine_similarity_scores["non_relevant"]
        click.echo(
            f"Loaded cosine similarity scores from {cosine_similarity_scores_save_path}"
        )
        click.echo()
    else:
        click.echo(
            f"Cosine similarity scores file {cosine_similarity_scores_save_path} not found. Computing cosine similarity scores..."
        )

        # Load qrels
        relevant_docids = set()
        distractor_docids = set()
        qrels = {}
        qrels_file = os.path.join(DATASETS_FOLDER, dataset, "qrels.jsonl")
        with open(qrels_file, "r") as f:
            for line in f:
                item = json.loads(line)
                qid = item["query-id"]
                docid = item["corpus-id"]
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = item["type"]
                if item["type"] == "relevant":
                    relevant_docids.add(docid)
                elif item["type"] == "distractor":
                    distractor_docids.add(docid)

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
        core_texts = {}
        core_corpus_file = os.path.join(
            DATASETS_FOLDER, dataset, "relevant_distractor.jsonl"
        )
        with open(core_corpus_file, "r") as f:
            for line in f:
                item = json.loads(line)
                core_texts[item["_id"]] = {"title": item["title"], "text": item["text"]}
        click.echo(
            f"Collected {len(core_texts):,} relevant and distractor documents' texts from file {core_corpus_file}"
        )

        # Load non-relevant documents
        non_relevant_texts = []
        non_relevant_corpus_file = os.path.join(
            DATASETS_FOLDER, dataset, "non_relevant.jsonl"
        )
        with open(non_relevant_corpus_file, "r") as f:
            for line in f:
                item = json.loads(line)
                non_relevant_texts.append(
                    {"title": item["title"], "text": item["text"]}
                )
        click.echo(
            f"Collected {len(non_relevant_texts):,} non-relevant documents' texts from file {non_relevant_corpus_file}"
        )
        non_relevant_texts = random.sample(non_relevant_texts, 10_000)
        click.echo(
            f"Sampled {len(non_relevant_texts):,} non-relevant documents' texts for score distribution fitting"
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
        core_docids = list(core_texts.keys())
        core_texts_list = [core_texts[docid] for docid in core_docids]
        save_path = os.path.join(
            EMBEDDINGS_FOLDER, model_name, dataset, "core_embeddings.pt"
        )
        core_embeddings = load_embeddings(
            save_path,
            model,
            core_texts_list,
        )

        # Encode non-relevant texts
        save_path = os.path.join(
            EMBEDDINGS_FOLDER, model_name, dataset, "non_relevant_embeddings.pt"
        )
        non_relevant_embeddings = load_embeddings(
            save_path,
            model,
            non_relevant_texts,
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
            for docid, doc_embedding in zip(core_docids, core_embeddings):
                score = cos_sim(
                    query_embedding.unsqueeze(0),
                    doc_embedding.unsqueeze(0),
                ).item()
                if docid in qrels[qid] and qrels[qid][docid] == "relevant":
                    relevant_cosine_similarity_scores[qid].append(score)
                else:
                    distractor_cosine_similarity_scores[qid].append(score)

        # Compute cosine similarity scores for non-relevant documents
        non_relevant_cosine_similarity_scores = {}
        for i, query_embedding in tqdm(
            enumerate(query_embeddings),
            desc="Computing cos sim scores (non-relevant)",
            total=len(query_embeddings),
        ):
            qid = query_ids[i]
            non_relevant_cosine_similarity_scores[qid] = []
            for doc_embedding in non_relevant_embeddings:
                score = cos_sim(
                    query_embedding.unsqueeze(0),
                    doc_embedding.unsqueeze(0),
                ).item()
                non_relevant_cosine_similarity_scores[qid].append(score)

        # Fit score distribution
        non_relevant_fitting_params = {}
        for qid, scores in tqdm(
            non_relevant_cosine_similarity_scores.items(),
            desc="Fitting score distributions",
        ):
            non_relevant_fitting_params[qid] = st.skewnorm.fit(scores)

        # Store cosine similarity scores
        os.makedirs(os.path.dirname(cosine_similarity_scores_save_path), exist_ok=True)
        with open(cosine_similarity_scores_save_path, "w") as f:
            json.dump(
                {
                    "relevant": relevant_cosine_similarity_scores,
                    "distractor": distractor_cosine_similarity_scores,
                    "non_relevant": non_relevant_fitting_params,
                },
                f,
                indent=4,
            )
        click.echo(
            f"Stored cosine similarity scores to {cosine_similarity_scores_save_path}"
        )
        click.echo()

    # Read full results JSON file
    full_results_json_file = os.path.join(
        RESULTS_FOLDER, "exact", model_name, f"{dataset}.json"
    )
    with open(full_results_json_file, "r") as f:
        full_results_json = json.load(f)

    # Turn scores lists into numpy arrays for faster processing
    for qid in relevant_cosine_similarity_scores.keys():
        relevant_cosine_similarity_scores[qid] = np.array(
            relevant_cosine_similarity_scores[qid]
        )
        distractor_cosine_similarity_scores[qid] = np.array(
            distractor_cosine_similarity_scores[qid]
        )
        _temp = np.array(full_results_json[qid])
        full_results_json[qid] = np.concatenate(
            [distractor_cosine_similarity_scores[qid], _temp]
        )

    # Predict NDCG@k (Full)
    click.echo(f"Full Results Metrics:")
    for k in K_NDCG:
        predicted_ndcg_at_k = {}
        for qid in relevant_cosine_similarity_scores.keys():
            N = 0
            tau_k_ndcg = compute_ndcg_at_k_for_q(
                k,
                relevant_cosine_similarity_scores[qid],
                full_results_json[qid],
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
                full_results_json[qid],
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
    click.echo(f"Base Results Metrics:")
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
