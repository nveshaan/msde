# ways to integrate dmsl into neural networks
1. differentiable dmsl layer: replace emperical denisty weights with attention scores, and shift with aggregated values
    reconstruction + classification loss gives most promising results, an interesting behaviour when adding a simplical set weights loss
    might be useful when there is imbalance of anomalies and normal samples for anomaly detection; weighing based on the features instead of density might lead to better separation even when imbalanced

2. dmsl + ste

3. dmsl as target (consistency): try different kind of losses which have the same goal
    when and where?


 - combine 1 and 3?
    either distill dmsl into a gat, or force an encoder to match the shifted output


> Does "the shift" need to reflect the current population, or just the training-time population? If a future single sample arrives and the surrounding density has genuinely shifted (batch effects, a new subpopulation, distribution drift), the encoder-only approach has no way to know that — it outputs whatever it memorized during training. A live GAT recomputes against whatever's actually there, so it adapts automatically — at the cost of not being single-sample.

So, concretely:

1. **GAT distillation:** output for sample x changes depending on batch composition. Same input, different embedding on different days — a real problem for reproducibility, caching, and any API that promises one-sample-in-one-vector-out. To make it usable for single-sample inference you'd need GraphSAGE's trick: keep a fixed reference/atlas set and attend the new sample against that, not against other query samples. That's workable, but now you're maintaining a reference index at serving time, not doing pure feed-forward inference.

2. **Encoder-matches-shifted-target**: the encoder itself becomes a plain function z = f(x), no neighbors needed at all, genuinely single-sample. But you're asking a feed-forward function to memorize, for every point in the input space, what an average-over-neighbors would have been during training — i.e. you're baking a population-level statistic into per-sample weights.


# plan
1. dmsl as target (consistency), only encoder: experiment on when and where and batching (dec, deepcluster, swav), try different kind of losses (parametric umap compares, direct mse on embeddings, and umap loss).

2. gat distillation on top of learned embeddings, same experiments as above

3. any loss that captures dmsl's goal? to make differential dmsl work; make observations on ideal datasets, compare different stuff (like adjacency matrices, fuzzy weights, clustering, ...) before and after dmsl