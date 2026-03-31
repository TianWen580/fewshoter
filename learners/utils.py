import importlib
from typing import Any


torch = importlib.import_module("torch")


def l2_normalize(embeddings: Any, eps: float = 1e-12):
    """L2-normalize embedding rows.

    Args:
        embeddings: Tensor-like matrix with shape ``[N, D]``.
        eps: Small epsilon to avoid divide-by-zero.

    Returns:
        Any: Row-wise normalized embeddings.
    """
    norms = embeddings.norm(dim=1, keepdim=True).clamp_min(eps)
    return embeddings / norms


def squared_euclidean_distance(queries: Any, prototypes: Any):
    """Compute pairwise squared Euclidean distance.

    Args:
        queries: Query embedding matrix ``[N, D]``.
        prototypes: Prototype embedding matrix ``[C, D]``.

    Returns:
        Any: Distance matrix ``[N, C]``.

    Raises:
        ValueError: If input tensors are not rank-2 or dimensions mismatch.
    """
    if queries.ndim != 2:
        raise ValueError("query embeddings must be a 2D tensor")
    if prototypes.ndim != 2:
        raise ValueError("prototype embeddings must be a 2D tensor")
    if queries.shape[1] != prototypes.shape[1]:
        raise ValueError("support and query embeddings must share embedding dimension")
    return ((queries[:, None, :] - prototypes[None, :, :]) ** 2).sum(dim=-1)


def validate_episode_tensors(
    support_embeddings: Any,
    support_labels: Any,
    query_embeddings: Any | None = None,
    query_labels: Any | None = None,
) -> None:
    """Validate support/query tensor shapes for episodic learning.

    Args:
        support_embeddings: Support embedding matrix ``[S, D]``.
        support_labels: Support labels vector ``[S]``.
        query_embeddings: Optional query embedding matrix ``[Q, D]``.
        query_labels: Optional query labels vector ``[Q]``.

    Raises:
        ValueError: If tensor ranks, lengths, or embedding dimensions are invalid.
    """
    if support_embeddings.ndim != 2:
        raise ValueError("support embeddings must be a 2D tensor")
    if support_labels.ndim != 1:
        raise ValueError("support labels must be a 1D tensor")
    if support_embeddings.shape[0] != support_labels.shape[0]:
        raise ValueError("support embeddings and support labels must have the same length")

    if query_embeddings is not None:
        if query_embeddings.ndim != 2:
            raise ValueError("query embeddings must be a 2D tensor")
        if support_embeddings.shape[1] != query_embeddings.shape[1]:
            raise ValueError("support and query embeddings must share embedding dimension")

    if query_labels is not None:
        if query_labels.ndim != 1:
            raise ValueError("query labels must be a 1D tensor")
        if query_embeddings is None:
            raise ValueError("query embeddings are required when query labels are provided")
        if query_embeddings.shape[0] != query_labels.shape[0]:
            raise ValueError("query embeddings and query labels must have the same length")
