# Source: https://github.com/D6nam853/medi-msde/blob/main/msde/msde.py

import numpy as np
from scipy.spatial import cKDTree
from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.special import expit
from numba import njit, prange
from pynndescent import NNDescent
from tqdm import tqdm
import os
from annoy import AnnoyIndex


def count_points_within_radius(X, tree, epsilon):
    neighbors = tree.query_ball_tree(tree, epsilon)
    counts = np.array([len(pts) - 1 for pts in neighbors])
    return counts


def max_min_distances_kdtree(X):
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=X.shape[0])
    all_distances = dists[:, 1:].flatten()
    return np.max(all_distances), np.min(all_distances)


def binary_search_condition(low, high, condition, tol=1e-4, max_iter=50):
    result = None
    for _ in range(max_iter):
        mid = (low + high) / 2
        if condition(mid):
            result = mid
            high = mid
        else:
            low = mid
        if abs(high - low) < tol:
            break
    return result


def condition_formulation(point_in_radius_counts, nbd_sample_count_threshold,
                          satisfiability_proportion):
    if_satisfied = (np.where(
        point_in_radius_counts > nbd_sample_count_threshold)[0]) * 1
    return np.sum(if_satisfied) >= satisfiability_proportion


def get_empirical_weights(
    X,
    nbd_sample_count_threshold=5,
    max_iters_weight_count=4,
    satisfiability_proportion=0.3,
    n_neighbors=15,
    metric='euclidean',
    random_state=42,
    batch_size=1000
):
    def umap_graph_similarity(X_batch):
        knn_indices, knn_dists, _ = nearest_neighbors(
            X_batch, n_neighbors=n_neighbors, metric=metric,
            metric_kwds={}, angular=False, random_state=random_state,
            low_memory=True, use_pynndescent=True
        )
        G, _, _ = fuzzy_simplicial_set(
            X_batch, n_neighbors=n_neighbors, random_state=random_state,
            metric=metric, knn_indices=knn_indices, knn_dists=knn_dists,
            angular=False, set_op_mix_ratio=1.0, local_connectivity=1.0,
            apply_set_operations=True, verbose=False
        )
        return G.toarray() if isinstance(G, csr_matrix) else G

    def compute_weights_from_similarity(sim, X_batch):
        tree = cKDTree(sim)
        max_dist, min_dist = max_min_distances_kdtree(sim)
        threshold = (
            max(1, len(X_batch) - 1)
            if nbd_sample_count_threshold >= len(X_batch)
            else nbd_sample_count_threshold
        )
        effective_required = int(satisfiability_proportion * len(X_batch))

        eps = binary_search_condition(
            min_dist, max_dist,
            lambda mid: condition_formulation(
                count_points_within_radius(sim, tree, mid),
                threshold, effective_required
            )
        )
        if eps is None:
            eps = binary_search_condition(
                min_dist, max_dist,
                lambda mid: condition_formulation(
                    count_points_within_radius(sim, tree, mid),
                    max(1, threshold // 2), effective_required // 2
                )
            )
        if eps is None:
            eps = max_dist

        delta = (eps - 1e-6) / max_iters_weight_count
        all_counts = []
        for _ in range(max_iters_weight_count):
            counts = count_points_within_radius(sim, tree, eps)
            all_counts.append(counts)
            eps -= delta
        return np.mean(all_counts, axis=0)

    if len(X) <= 3 * n_neighbors:
        sim = umap_graph_similarity(X)
        return compute_weights_from_similarity(sim, X)

    effective_batch_size = min(batch_size, max(n_neighbors * 3, 100))
    total_batches = (len(X) + effective_batch_size - 1) // effective_batch_size
    weights_all = []
    for batch_idx in tqdm(range(total_batches), desc="Calculating Emperical Weights"):
        start = batch_idx * effective_batch_size
        end   = min(len(X), start + effective_batch_size)
        X_batch = X[start:end]
        sim = umap_graph_similarity(X_batch)
        weights_all.append(compute_weights_from_similarity(sim, X_batch))
    return np.concatenate(weights_all)


# TODO: consider vectorizing the loops
@njit(fastmath=True, parallel=True)
def shift_data(X, indices, weights, learning_rate):
    n, k = indices.shape
    d = X.shape[1]
    revised_d = np.empty_like(X)
    total_change = np.empty(n)

    for i in prange(n):
        denom = 0.0
        for j in range(k):
            denom += weights[indices[i, j]]
        if denom < 1e-6:
            denom = 1e-6

        for t in range(d):
            acc = 0.0
            for j in range(k):
                acc += weights[indices[i, j]] * X[indices[i, j], t]
            acc /= denom
            revised_d[i, t] = acc

        dist = 0.0
        for t in range(d):
            diff = revised_d[i, t] - X[i, t]
            dist += diff * diff
        dist = np.sqrt(dist)
        total_change[i] = dist

        if dist > 1e-6:
            scale = learning_rate * dist
            for t in range(d):
                revised_d[i, t] = (X[i, t]
                                   + scale * (revised_d[i, t] - X[i, t]) / dist) # BUG: add 1e-6 for numerical stability?
        else:
            for t in range(d):
                revised_d[i, t] = X[i, t]

    # BUG: total_change is inconsistent with the papers. is it intended change or is it applied change?
    return revised_d, total_change


def get_shift_fast(X, k, nbd_sample_count_threshold, learning_rate, nn,
                   max_iters_shift, shift_threshold, batch_size, path):
    if path is not None and os.path.exists(path):
        print("Loading saved weights.")
        weights = np.load(path)
    else:
        weights = get_empirical_weights(
            X,
            nbd_sample_count_threshold=nbd_sample_count_threshold,
            max_iters_weight_count=4,
            satisfiability_proportion=0.3,
            batch_size=batch_size,
        )
        if path is not None:
            np.save(path, weights)
            print("Saved weights to avoid computation in future.")

    shifted_dataset  = X.copy()
    total_distance   = np.zeros(X.shape[0])
    feature_distance = np.zeros(X.shape[1])
    trajectory = [shifted_dataset.copy()]

    pbar = tqdm(range(max_iters_shift), desc="MSDE")
    for i in pbar:
        if nn=="nndescent":
            index = NNDescent(
                shifted_dataset, n_neighbors=k, n_jobs=-1,
                metric="euclidean", random_state=42
            )
            indices, _ = index.neighbor_graph
        elif nn=="annoy":
            # n_trees, search_k are the two main params to tune Annoy.
            d = shifted_dataset.shape[1]
            annoy_index = AnnoyIndex(d, 'euclidean')
            for idx in range(shifted_dataset.shape[0]):
                annoy_index.add_item(idx, shifted_dataset[idx])
                
            annoy_index.build(n_trees=50, n_jobs=-1) 
            indices = np.empty((shifted_dataset.shape[0], k), dtype=np.int64)
            for idx in range(shifted_dataset.shape[0]):
                indices[idx] = annoy_index.get_nns_by_item(idx, k, search_k=-1)

        indices = indices.astype(np.int64)

        revised_d, total_change = shift_data(
            shifted_dataset, indices, weights, learning_rate
        )
        feature_change = np.sum(np.abs(revised_d - shifted_dataset), axis=0)
        total_distance += total_change
        feature_distance += feature_change
        shifted_dataset = revised_d
        trajectory.append(shifted_dataset.copy())

        if total_change.mean() < shift_threshold:
            pbar.total = i+1
            pbar.refresh()
            pbar.close()
            print(f"Total change converged after iteration {i+1}. Exiting the loop.")
            break

    return shifted_dataset, total_distance, feature_distance, trajectory


def mean_shift_density_enhancement(X, k=50, nbd_sample_count_threshold=70,
                                 learning_rate=0.33, max_iters_shift=8,
                                 shift_threshold=0.01, batch_size=1000,
                                 path=None, nn="nndescent"):
    return get_shift_fast(
        X, k, nbd_sample_count_threshold, learning_rate, nn,
        max_iters_shift, shift_threshold, batch_size, path,
    )


class _GDEScorer:
    """
    Gaussian Density Estimator on PCA-256 projected shifted features.
    Anomaly score = Mahalanobis distance from the normal cluster.
    Higher score → more anomalous.
    """

    GDE_PCA_DIM = 256
    REG         = 1e-4

    def __init__(self):
        self._pca     = None
        self._mean    = None
        self._cov_inv = None

    def fit(self, X_shifted_train: np.ndarray):
        X = X_shifted_train.copy()

        dim = min(self.GDE_PCA_DIM, X.shape[1], X.shape[0] - 1)
        self._pca = PCA(n_components=dim, random_state=42)
        X = self._pca.fit_transform(X)

        self._mean = X.mean(axis=0)
        X_c = X - self._mean
        cov = (X_c.T @ X_c) / max(len(X) - 1, 1)
        cov += np.eye(dim) * self.REG
        self._cov_inv = np.linalg.inv(cov)
        return self

    def score(self, X_shifted_test: np.ndarray) -> np.ndarray:
        X    = self._pca.transform(X_shifted_test.copy())
        diff = X - self._mean
        maha2 = np.einsum('ij,jk,ik->i', diff, self._cov_inv, diff)
        return np.sqrt(np.clip(maha2, 0, None))


class MSDE:
    """
    Manifold-Shift Manifold Learning anomaly detector.

    Scoring is fixed to GDE (Mahalanobis distance in PCA-256 shifted space).
    Only the five shift hyperparameters need to be tuned.
    """

    # TODO: add cumulative displacement anamoly scoring
    def __init__(
        self,
        seed: int,
        model_name: str = 'MSDE',
        k: int = 50,
        nbd_sample_count_threshold: int = 70,
        learning_rate: float = 0.33,
        max_iters_shift: int = 8,
        shift_threshold: float = 0.01,
        anomalyThreshold: float = 0.22,
        scaler=None,
        anomalyScore = 'GDE',
        batch_size: int = 1000,
        path: str = None,
        nn: str = "nndescent",
    ):
        self.k                          = k
        self.nbd_sample_count_threshold = nbd_sample_count_threshold
        self.learning_rate              = learning_rate
        self.max_iters_shift            = max_iters_shift
        self.shift_threshold            = shift_threshold
        self.anomalyThreshold           = anomalyThreshold
        self.scaler                     = scaler or StandardScaler()
        self.seed                       = seed
        self.model_name                 = model_name
        self.X_train_ref                = None
        self._gde                       = None
        self.anomalyScore               = anomalyScore
        self.batch_size                 = batch_size
        self.path                       = path
        self.nn                         = nn

    def fit(self, X_train, y_train=None):
        self.X_train_ref = np.asarray(X_train).copy()

        # Shift training normals and fit GDE on the shifted positions
        X_shifted, X_total_dist, X_feature_dist, trajectory = mean_shift_density_enhancement(
            self.X_train_ref,
            k=self.k,
            nbd_sample_count_threshold=self.nbd_sample_count_threshold,
            learning_rate=self.learning_rate,
            max_iters_shift=self.max_iters_shift,
            shift_threshold=self.shift_threshold,
            batch_size=self.batch_size,
            path=self.path,
            nn=self.nn,
        )
        if self.anomalyScore == 'GDE':
            self._gde = _GDEScorer().fit(X_shifted)
        return X_shifted, X_total_dist, X_feature_dist, trajectory
    
    # TODO: implement partial_fit
    def partial_fit(self):
        raise NotImplementedError

    def predict_score(self, X):
        X = np.asarray(X)

        if self.X_train_ref is None:
            raise ValueError("Model not fitted. Call fit(X_train) first.")
        
        if self._gde is None:
            raise ValueError("GDE scorer not fitted. Pass 'GDE' to anomalyScore.")

        # Shift train + test together to preserve neighbourhood context
        X_all = np.vstack([self.X_train_ref, X])
        n_train = len(self.X_train_ref)

        X_shifted_all, _, _, _ = mean_shift_density_enhancement(
            X_all,
            k=self.k,
            nbd_sample_count_threshold=self.nbd_sample_count_threshold,
            learning_rate=self.learning_rate,
            max_iters_shift=self.max_iters_shift,
            shift_threshold=self.shift_threshold,
            batch_size=self.batch_size,
            path=self.path,
            nn=self.nn,
        )

        # Score test points via GDE fitted on training normals
        X_shifted_test = X_shifted_all[n_train:]
        gde_scores     = self._gde.score(X_shifted_test)

        scores = self.scaler.fit_transform(gde_scores.reshape(-1, 1))
        return expit(scores).squeeze()