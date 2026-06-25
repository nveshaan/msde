# CellFlow
1. what does each part of msde/dmsl do?
2. what can we infer in trajectories of anomalies (or loners) merging into a cluster (or majority)?
3. is it assumed that cells undergoing differentiation to be in minorites and cells which are functional adults to be majorities for msde/dmsl to infer the trajectories?
4. how are msde/dmsl trajectory genes different from the real trajectory genes? if vastly different, how useful are the fake trajectory genes, and if similar, how does msde/dmsl compare to other methods such as optimal transport and flow matching (which can also predict the perturbation effects of drugs or gene knockouts on gene expression levels)
5. trajectories on hvg space vs embedding space. is there any benefit?

# Optimal Transport
1. instead of moving towards the mean, can we use optimal transport (towards the nearest neighbors) to find a path which is more ideal to real cell trajectory?
2. how dependent is the algorithm on the data points? do we need sufficient data points for every stage of the cell to accurately infer the real trajectory?

# Manifold Refinement for Representation Learning
1. what are the properties of a good embedding?
2. what are the current challanges in representation learning in producing good embeddings?
3. can msde/dmsl be applied to the embeddings in hopes of refining the learned manifold?
4. can msde/dmsl itself be improved or changed to achieve better objective? 
    - need to understand msde/dmsl from ground up by challenging every decision made to use a certain component in the algorithm
    - experiment the behaviors on different manifolds, noise levels, etc. to profile its performance on clustering, manifold refinement, etc.

## Notes
1. Trajectory genes has to renamed and even CellFlow might need to renamed (there already exists a tool called CellFlow which uses flow matching to predict the perturbation of gene expression levels)
2. Saptarshi sir told to see what will happen if we keep one cluster fixed (how would the data points around the cluster interact with it)