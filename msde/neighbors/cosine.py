from .chunked import ChunkedKNN


class CosineKNN(ChunkedKNN):
    """Exact top-k neighbours by cosine distance ``1 - cos(x, y)``.

    ``Neighbors.dist`` holds cosine distances. Only the neighbour choice
    changes; the density weights and the movement step still work in the
    original feature space.
    """

    metric = "cosine"
