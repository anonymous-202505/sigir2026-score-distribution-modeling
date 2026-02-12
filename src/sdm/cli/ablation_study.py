import os
import pickle
import random

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from tqdm import tqdm

from sdm.config import *
from sdm.utils.similarity import cos_sim

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)


MODEL_NAME = "snowflakev2"
DATASET = "hotpotqa"
NUM_POOLING = 100
NUM_ITERATIONS = 20


def evaluate_distribution_fit(
    original_data: np.ndarray, sampled_data: np.ndarray
) -> dict:

    results = {}

    # =========================
    # 1. NORMAL DISTRIBUTION
    # =========================
    mu, sigma = stats.norm.fit(sampled_data)

    # KS Test
    D_norm, p_norm = stats.kstest(original_data, "norm", args=(mu, sigma))

    # Log-Likelihood
    ll_norm = np.sum(stats.norm.logpdf(original_data, mu, sigma))
    ll_norm /= len(original_data)  # Average log-likelihood per data point

    results["normal"] = {
        "params": {"mu": mu, "sigma": sigma},
        "KS_statistic": D_norm,
        "KS_pvalue": p_norm,
        "log_likelihood": ll_norm,
    }

    # =========================
    # 2. SKEW-NORMAL DISTRIBUTION
    # =========================
    a, loc, scale = stats.skewnorm.fit(sampled_data)

    # KS Test
    D_skew, p_skew = stats.kstest(original_data, "skewnorm", args=(a, loc, scale))

    # Log-Likelihood
    ll_skew = np.sum(stats.skewnorm.logpdf(original_data, a, loc, scale))
    ll_skew /= len(original_data)  # Average log-likelihood per data point

    results["skew_normal"] = {
        "params": {"a": a, "loc": loc, "scale": scale},
        "KS_statistic": D_skew,
        "KS_pvalue": p_skew,
        "log_likelihood": ll_skew,
    }

    return results


@click.command()
def ablation_study():
    """
    Conduct an ablation study to analyze how the distribution of cosine similarity scores between
    query and non-relevant passage embeddings changes as we vary the number of non-relevant samples
    used for fitting the distribution. We will evaluate the fit using both the Kolmogorov-Smirnov
    test and log-likelihood, comparing normal and skew-normal distributions.
    """
    # Load results if they already exist to avoid recomputation
    results_path = os.path.join(RESOURCES_FOLDER, "ablation_results.pkl")
    if not os.path.exists(results_path):

        # Instantiate the results dictionary
        results = {}
        for num_samples in [
            10,
            20,
            30,
            50,
            70,
            100,
            200,
            300,
            500,
            700,
            1000,
            2000,
            3000,
            5000,
            7000,
            10000,
            20000,
            30000,
            50000,
            70000,
            100000,
        ]:
            results[num_samples] = []

        # Load embeddings
        query_embeddings_path = os.path.join(
            "embeddings", MODEL_NAME, DATASET, "query_embeddings.pt"
        )
        if not os.path.exists(query_embeddings_path):
            raise FileNotFoundError(
                f"Query embeddings not found at {query_embeddings_path}. Please run the encoding step first."
            )
        non_relevant_embeddings_path = os.path.join(
            "embeddings",
            MODEL_NAME,
            DATASET,
            str(NUM_POOLING),
            "non_relevant_embeddings.pt",
        )
        if not os.path.exists(non_relevant_embeddings_path):
            raise FileNotFoundError(
                f"Non-relevant embeddings not found at {non_relevant_embeddings_path}. Please run the encoding step first."
            )
        query_embeddings = torch.load(query_embeddings_path, weights_only=False)
        non_relevant_embeddings = torch.load(
            non_relevant_embeddings_path, weights_only=False
        )

        # Sample embeddings of non-relevant passages
        for _ in tqdm(range(NUM_ITERATIONS)):  # Run multiple iterations for robustness
            # Randomly sample one query embedding and a subset of non-relevant embeddings
            query_sample = random.choice(query_embeddings)
            non_relevant_samples = list(non_relevant_embeddings)

            # Compute similarity scores (e.g., cosine similarity)
            scores = cos_sim(query_sample, non_relevant_samples).cpu().numpy().flatten()

            for num_samples in results.keys():
                sample = scores[:num_samples]
                fit_results = evaluate_distribution_fit(scores, sample)
                results[num_samples].append(fit_results)

        for num_samples, fit_results_list in results.items():
            print(f"\n=== Num Samples: {num_samples} ===")
            ks_stats_normal = [r["normal"]["KS_statistic"] for r in fit_results_list]
            ks_stats_skew = [r["skew_normal"]["KS_statistic"] for r in fit_results_list]
            ks_p_values_normal = [r["normal"]["KS_pvalue"] for r in fit_results_list]
            ks_p_values_skew = [r["skew_normal"]["KS_pvalue"] for r in fit_results_list]
            ll_normal = [r["normal"]["log_likelihood"] for r in fit_results_list]
            ll_skew = [r["skew_normal"]["log_likelihood"] for r in fit_results_list]

            print(
                f"Normal KS: mean={np.mean(ks_stats_normal):.4f}, std={np.std(ks_stats_normal):.4f}, p-value mean={np.mean(ks_p_values_normal):.4f} p-value std={np.std(ks_p_values_normal):.4f}"
            )
            print(
                f"Skew-Normal KS: mean={np.mean(ks_stats_skew):.4f}, std={np.std(ks_stats_skew):.4f}, p-value mean={np.mean(ks_p_values_skew):.4f} p-value std={np.std(ks_p_values_skew):.4f}"
            )
            print(
                f"Normal Log-Likelihood: mean={np.mean(ll_normal):.2f}, std={np.std(ll_normal):.2f}"
            )
            print(
                f"Skew-Normal Log-Likelihood: mean={np.mean(ll_skew):.2f}, std={np.std(ll_skew):.2f}"
            )

        with open(results_path, "wb") as f:
            pickle.dump(results, f)

    else:
        with open(results_path, "rb") as f:
            results = pickle.load(f)

    df_ll_normal = pd.DataFrame(columns=["num_samples", "iteration", "log_likelihood"])
    df_ll_normal["num_samples"] = df_ll_normal["num_samples"].astype(int)
    df_ll_normal["iteration"] = df_ll_normal["iteration"].astype(int)
    df_ll_normal["log_likelihood"] = df_ll_normal["log_likelihood"].astype(float)
    df_ll_normal.set_index(["num_samples", "iteration"], inplace=True)

    for num_samples, fit_results_list in sorted(results.items()):
        ll_normal = [r["normal"]["log_likelihood"] for r in fit_results_list]
        ll_skew = [r["skew_normal"]["log_likelihood"] for r in fit_results_list]

        for i, ll in enumerate(ll_normal):
            df_ll_normal.loc[(num_samples, i), "log_likelihood"] = ll

    sns.set_style("darkgrid")
    plt.figure(figsize=(5, 3))
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    sns.lineplot(data=df_ll_normal.reset_index(), x="num_samples", y="log_likelihood")

    plt.xscale("log")
    plt.xticks(fontsize=12)
    plt.xlabel("Number of samples (log scale)", fontsize=14)
    plt.yticks(fontsize=12)
    plt.ylabel("Norm. Log-likelihood", fontsize=14)
    plt.tight_layout()
    plt.savefig("log_likelihood_comparison.pdf", bbox_inches="tight", dpi=300)

    click.echo("Done")
