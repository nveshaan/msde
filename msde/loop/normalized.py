import torch

from .early_stop import EarlyStopLoop


def _unit(X, eps):
    """Rows scaled to unit length (a zero row stays zero)."""
    return X / X.norm(dim=1, keepdim=True).clamp_min(eps)


class NormalizedLoop(EarlyStopLoop):
    """``EarlyStopLoop`` on the unit sphere: the points are normalised to unit
    length before every shift.

    The input is normalised first (``preprocess``), so the density weights are
    computed on the sphere too, and again at the start of each iteration, so
    neighbour selection and the step always see unit vectors. A step moves a
    point off the sphere (a convex combination of unit vectors is shorter than
    one); the next iteration puts it back, which keeps only its direction.

    tangent_projection : project each point's shift onto the tangent plane of the
        sphere at that point (remove the radial component) before applying it,
        and scale the reported movement by the fraction of the shift that was kept.
        The step then stays on
        the sphere to first order instead of cutting inside it. After
        renormalisation the two differ only in how far the point moves: the
        projected step keeps weight ``1 - eta * (mu . x)`` on x where the plain
        step keeps ``1 - eta``, so it moves slightly less, by an amount that
        grows with the angular spread of the neighbourhood.
    normalize_output : also normalise the returned points (and the trajectory),
        which otherwise end up just inside the sphere after the last shift.
    """

    def __init__(self, max_iters_shift=5, shift_threshold=0.0001, recompute_neighbors=0,
                 keep_trajectory=False, tangent_projection=False, normalize_output=False, eps=1e-12):
        super().__init__(max_iters_shift, shift_threshold, recompute_neighbors, keep_trajectory)
        if not eps > 0:
            raise ValueError("eps must be > 0")
        self.tangent_projection = tangent_projection
        self.normalize_output = normalize_output
        self.eps = eps

    def preprocess(self, X):
        return _unit(X, self.eps)

    def before_shift(self, X, iter_count, ctx):
        return _unit(X, self.eps)

    def after_shift(self, X, X_new, change, ctx):
        if not self.tangent_projection:
            return X_new, change
        shift = X_new - X                                          # X is on the sphere here
        tangent = shift - (shift * X).sum(dim=1, keepdim=True) * X   # drop the radial component
        # Keep the step's own notion of movement (PlainBarycenter reports the distance to the
        # barycenter, GravityBarycenter the displacement) and shrink it by what the projection removed.
        length = shift.norm(dim=1)
        kept = torch.where(length > self.eps, tangent.norm(dim=1) / length.clamp_min(self.eps),
                           torch.ones_like(length))
        return X + tangent, change * kept

    def postprocess(self, result, ctx):
        if not self.normalize_output:
            return result
        shifted, total_distance, trajectory = result
        return _unit(shifted, self.eps), total_distance, [_unit(t, self.eps) for t in trajectory]
