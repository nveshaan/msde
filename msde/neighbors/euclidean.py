from .chunked import ChunkedKNN


class EuclideanKNN(ChunkedKNN):
    """Exact top-k neighbours by Euclidean distance."""

    metric = "euclidean"
