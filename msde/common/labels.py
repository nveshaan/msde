import torch

LABEL_MODES = ("same_label", "zero_label")


class LabelPolicy:
    """How labels passed to forward() restrict neighbour search.

    - "same_label": a point's neighbours must share its label.
    - "zero_label": a point's neighbours must have label 0, whatever its own
      label -- points labelled 1, 2, 3, ... are shifted towards label-0 points.
    """

    def __init__(self, mode="same_label"):
        if mode not in LABEL_MODES:
            raise ValueError(f"label_mode must be 'same_label' or 'zero_label'; got {mode!r}")
        self.mode = mode

    def query_labels(self, labels):
        """Labels the query side compares against the neighbour-side labels."""
        if self.mode == "same_label":
            return labels
        return torch.zeros_like(labels)


def as_label_tensor(labels, n, device, what="X"):
    """Validate ``labels`` as a detached 1D long tensor of length ``n``."""
    labels = torch.as_tensor(labels, device=device).detach().long()
    if labels.dim() != 1 or labels.shape[0] != n:
        raise ValueError(
            f"labels must be 1D with length matching {what} (n={n}); "
            f"got shape {tuple(labels.shape)}"
        )
    return labels
