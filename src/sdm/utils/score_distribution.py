import math
from typing import Callable

import numpy as np
from scipy import stats as st

from sdm.config import *


def alpha_skewnorm(params: tuple, tau: float) -> float:
    """
    Tail probability P[S >= tau] for S ~ SkewNormal(a, loc, scale).
    """
    a, loc, scale = params
    assert scale > 0.0
    return 1.0 - st.skewnorm.cdf(tau, a, loc=loc, scale=scale)  # type: ignore


def alpha_empirical(scores, tau: float) -> float:
    """
    Tail probability P[S >= tau] estimated empirically from a list of scores.
    scores: list/array of cosine similarity scores for relevant or distractor documents (one query).
    """
    count_ge = (scores >= tau).sum()
    return count_ge / len(scores)


def compute_recall_at_k_for_q(
    k: int,
    relevant_cosine_similarity_scores: np.ndarray,
    distractor_cosine_similarity_scores: np.ndarray,
    params_n: tuple,
    N: int,
    fun_alpha_n: Callable = alpha_skewnorm,
    tol: float = 1e-10,
    max_iter: int = 100,
):
    """
    Numerically exact Recall@k by solving:
        R * alpha_r(tau_k) + D * alpha_d(tau_k) + N * alpha_n(tau_k) = k
    then Recall@k = alpha_r(tau_k).
    """
    R = len(relevant_cosine_similarity_scores)
    D = len(distractor_cosine_similarity_scores)

    assert k > 0 and R > 0 and D >= 0 and N >= 0

    # Define g(tau) = expected count >= tau - k
    def g(tau):
        sum = R * alpha_empirical(relevant_cosine_similarity_scores, tau)
        if N > 0:
            sum += N * fun_alpha_n(params_n, tau)
        if D > 0:
            sum += D * alpha_empirical(distractor_cosine_similarity_scores, tau)
        return sum - k

    # Bracket where g(lo) >= 0 and g(hi) <= 0
    lo = -1.0
    hi = 1.0
    glo, ghi = g(lo), g(hi)

    # If k is extreme, clamp
    if glo <= 0:
        tau_k = lo
        return alpha_empirical(relevant_cosine_similarity_scores, tau_k)
    if ghi >= 0:
        tau_k = hi
        return alpha_empirical(relevant_cosine_similarity_scores, tau_k)

    # Robust bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if abs(gm) <= tol:
            break
        if gm > 0:
            lo = mid
        else:
            hi = mid
    tau_k = 0.5 * (lo + hi)
    return alpha_empirical(relevant_cosine_similarity_scores, tau_k)


def compute_ndcg_at_k_for_q(
    k: int,
    relevant_cosine_similarity_scores: np.ndarray,
    distractor_cosine_similarity_scores: np.ndarray,
    params_n: tuple,
    N: int,
    fun_alpha_n: Callable = alpha_skewnorm,
    tie_inclusive: bool = True,
) -> float:
    """
    Compute expected NDCG@k for ONE query WITHOUT Monte Carlo (binary relevance).

    Assumptions:
      - all docs in relevant_cosine_similarity_scores are relevant (gain=1)
      - distractors and modeled nonrelevants are nonrelevant (gain=0)
      - modeled nonrelevants (count N) are i.i.d. with tail prob p(t)=fun_alpha_n(params_n, t)=P[S_n>=t]

    Method:
      E[DCG@k] = sum_{relevant doc with score s} sum_{x=0..xmax} w(rank(x)) * P(X=x)
      where rank(x)=A(s)+B(s)+x+1 and w(r)=1/log2(r+1), X~Binomial(N, p(s)).
      Then NDCG@k = E[DCG@k] / IDCG@k, with IDCG@k deterministic.

    Notes:
      - R and D are ignored for counts; list lengths are used.
      - N can be float; we round to nearest int.
      - tie_inclusive controls whether equal scores count as "beating" s (>= vs >).
    """
    R = len(relevant_cosine_similarity_scores)
    D = len(distractor_cosine_similarity_scores)

    assert k > 0 and R > 0 and D >= 0 and N >= 0

    # Ideal DCG for binary relevance: top min(R,k) ranks are relevant
    m = min(R, k)
    idcg = 0.0
    for r in range(1, m + 1):
        idcg += 1.0 / math.log2(r + 1)
    if idcg == 0.0:
        return 0.0

    def safe_p(p: float) -> float:
        if p < 0.0:
            return 0.0
        if p > 1.0:
            return 1.0
        return p

    edcg = 0.0

    for A, s in enumerate(sorted(relevant_cosine_similarity_scores, reverse=True)):
        # A(s): other relevants that beat s
        # B(s): distractors that beat s
        if tie_inclusive:
            B = (distractor_cosine_similarity_scores >= s).sum()
        else:
            B = (distractor_cosine_similarity_scores > s).sum()

        x_max = k - 1 - A - B
        if x_max < 0:
            continue  # can never enter top-k

        # Deterministic if no modeled nonrelevants
        if N == 0:
            rank = A + B + 1
            if rank <= k:
                edcg += 1.0 / math.log2(rank + 1)
            continue

        p = safe_p(float(fun_alpha_n(params_n, s)))

        # If x_max covers all possible x, doc always in top-k
        if x_max >= N:
            # Need full expectation of discount over X in [0..N]
            x_max_eff = N
        else:
            x_max_eff = x_max

        # Handle p==0 or p==1 cheaply
        if p == 0.0:
            rank = A + B + 1
            if rank <= k:
                edcg += 1.0 / math.log2(rank + 1)
            continue
        if p == 1.0:
            rank = A + B + N + 1
            if rank <= k:
                edcg += 1.0 / math.log2(rank + 1)
            continue

        # Sum pmf iteratively for stability:
        # pmf(0)=(1-p)^N; pmf(x+1)=pmf(x)*(N-x)/(x+1)*p/(1-p)
        pmf = (1.0 - p) ** N
        ratio = p / (1.0 - p)

        # x=0 term
        rank0 = A + B + 0 + 1
        if rank0 <= k:
            edcg += (1.0 / math.log2(rank0 + 1)) * pmf

        for x in range(0, x_max_eff):
            pmf = pmf * (N - x) / (x + 1) * ratio
            rank = A + B + (x + 1) + 1
            if rank <= k:
                edcg += (1.0 / math.log2(rank + 1)) * pmf

    return edcg / idcg
