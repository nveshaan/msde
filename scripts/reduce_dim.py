import hydra
from omegaconf import DictConfig, OmegaConf
import scanpy as sc
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../configs", config_name="msde")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    file_path = "data/" + cfg.data + ".h5ad"
    adata = sc.read_h5ad(file_path, backed='r')

    batch_size = cfg.pca.batch_size
    n = adata.X.shape[0]
    
    pca = IncrementalPCA(cfg.pca.n_components, batch_size=batch_size)
    for i in tqdm(range(0, n, batch_size), desc="Fitting PCA"):
        pca.partial_fit(adata.X[i:i+batch_size].toarray())
        
    X_pca = []
    for i in tqdm(range(0, n, batch_size), desc="Transforming X"):
        X_pca.append(pca.transform(adata.X[i:i+batch_size].toarray()))

    X_pca = np.stack(X_pca)
    save_path = "data/" + cfg.pca.save_file + ".npy"
    np.save(save_path, X_pca)
    print("Saved PCA Components")


if __name__ == "__main__":
    main()